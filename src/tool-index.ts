import { MemoryVectorBackend } from "./backends/memory.js";
import { normalizeTools } from "./normalize.js";
import { toolText } from "./text.js";
import type {
  FilterOptions,
  IndexedTool,
  SelectionTrace,
  StickyConfig,
  ToolIndexOptions,
  ToolScore,
} from "./types.js";

const cosine = (a: number[], b: number[]) => {
  if (a.length === 0 || a.length !== b.length) {
    throw new Error("Embedding dimension mismatch");
  }
  let dot = 0,
    aa = 0,
    bb = 0;
  for (let i = 0; i < a.length; i++) {
    if (!Number.isFinite(a[i]) || !Number.isFinite(b[i])) {
      throw new Error("Embedding vectors must contain only finite numbers");
    }
    dot += a[i]! * b[i]!;
    aa += a[i]! * a[i]!;
    bb += b[i]! * b[i]!;
  }
  return dot / (Math.sqrt(aa) * Math.sqrt(bb) || 1);
};
const words = (s: string) =>
  new Set(
    s
      .toLowerCase()
      .split(/[^a-z0-9]+/)
      .filter(Boolean),
  );
const lexical = (q: string, d: string) => {
  const a = words(q),
    b = words(d);
  let hit = 0;
  for (const x of a) if (b.has(x)) hit++;
  return hit / (Math.sqrt(a.size * b.size) || 1);
};
function validateVectors(vectors: number[][], expectedCount: number, expectedDimension?: number) {
  if (vectors.length !== expectedCount) {
    throw new Error("Embedder returned the wrong vector count");
  }
  const dimension = expectedDimension ?? vectors[0]?.length;
  for (const vector of vectors) {
    if (!dimension || vector.length !== dimension) {
      throw new Error("Embedding dimension mismatch");
    }
    if (vector.some((value) => !Number.isFinite(value))) {
      throw new Error("Embedding vectors must contain only finite numbers");
    }
  }
}
export function messagesToQueryText(messages: unknown): string {
  if (typeof messages === "string") return messages;
  if (Array.isArray(messages))
    return messages
      .map((m) => messagesToQueryText(m))
      .filter(Boolean)
      .join("\n");
  if (messages && typeof messages === "object") {
    const x = messages as Record<string, unknown>;
    if (typeof x.content === "string") return x.content;
    if (Array.isArray(x.content))
      return x.content
        .map((v) =>
          typeof v === "string"
            ? v
            : v && typeof v === "object" && typeof (v as Record<string, unknown>).text === "string"
              ? String((v as Record<string, unknown>).text)
              : "",
        )
        .join(" ");
  }
  return String(messages ?? "");
}
interface Session {
  at: number;
  query: number[];
  ids: string[];
}
export class ToolIndex {
  private sessions = new Map<string, Session>();
  private sticky: Required<StickyConfig>;
  private textConfig;
  private sinks;
  readonly backend;
  readonly embedder;
  constructor(options: ToolIndexOptions) {
    this.embedder = options.embedder;
    this.backend = options.backend ?? new MemoryVectorBackend();
    this.textConfig = options.text ?? {};
    this.sinks = options.traceSinks ?? [];
    this.sticky = {
      enabled: options.sticky?.enabled ?? false,
      reuseThreshold: options.sticky?.reuseThreshold ?? 0.95,
      refreshThreshold: options.sticky?.refreshThreshold ?? 0.8,
      keep: options.sticky?.keep ?? 2,
      ttlMs: options.sticky?.ttlMs ?? 1_800_000,
      maxSessions: Math.max(0, options.sticky?.maxSessions ?? 1_000),
    };
  }
  get size() {
    return this.backend.size;
  }
  get sessionCount() {
    return this.sessions.size;
  }
  async add(inputs: readonly unknown[]) {
    const tools = normalizeTools(inputs);
    if (tools.length === 0) return this;
    const texts = tools.map((t) => toolText(t, this.textConfig));
    const vectors = await this.embedder.embed(texts);
    validateVectors(vectors, tools.length, this.backend.list()[0]?.vector.length);
    await this.backend.upsert(
      tools.map((tool, i) => ({ tool, vector: vectors[i]!, text: texts[i]! })),
    );
    return this;
  }
  async sync(inputs: readonly unknown[]) {
    const normalized = normalizeTools(inputs);
    const keep = new Set(normalized.map((t) => t.id));
    await this.backend.remove(
      this.backend
        .list()
        .filter((r) => !keep.has(r.tool.id))
        .map((r) => r.tool.id),
    );
    // Tags and executable identity can change without changing the semantic ID.
    await this.add(normalized.map((t) => t.original));
    return this;
  }
  async remove(idsOrNames: string[]) {
    const ids = new Set(idsOrNames);
    await this.backend.remove(
      this.backend
        .list()
        .filter((r) => ids.has(r.tool.id) || ids.has(r.tool.name))
        .map((r) => r.tool.id),
    );
  }
  async clear() {
    await this.backend.clear();
  }
  async filter(messages: unknown, options: FilterOptions = {}) {
    return (await this.filterWithTrace(messages, options)).tools;
  }
  async filterWithTrace(
    messages: unknown,
    options: FilterOptions = {},
  ): Promise<{ tools: unknown[]; trace: SelectionTrace }> {
    const start = performance.now();
    const query = messagesToQueryText(messages);
    const k = Math.max(0, options.k ?? 12);
    const allow = new Set(options.allowTags ?? []),
      deny = new Set(options.denyTags ?? []);
    let records = this.backend.list();
    const allRecords = records;
    const candidateCount = records.length;
    const checked: IndexedTool[] = [];
    for (const r of records) {
      if (options.namespace && r.tool.namespace !== options.namespace) continue;
      if (allow.size && !r.tool.tags.some((t) => allow.has(t))) continue;
      if (r.tool.tags.some((t) => deny.has(t))) continue;
      if (options.policy && !(await options.policy(r.tool))) continue;
      checked.push(r);
    }
    records = checked;
    const embAt = performance.now();
    const [qv] = await this.embedder.embed([query]);
    if (!qv) throw new Error("Embedder returned no query vector");
    validateVectors([qv], 1);
    validateVectors(
      allRecords.map((record) => record.vector),
      allRecords.length,
      allRecords.length ? qv.length : undefined,
    );
    const embeddingMs = performance.now() - embAt;
    const semW = options.semanticWeight ?? 0.8,
      lexW = options.lexicalWeight ?? 0.2;
    let scores = records.map((r) => {
      const semanticScore = cosine(qv, r.vector),
        lexicalScore = lexical(query, r.text);
      return {
        record: r,
        semanticScore,
        lexicalScore,
        score: semanticScore * semW + lexicalScore * lexW,
      };
    });
    scores = scores
      .filter((x) => x.score >= (options.minScore ?? 0.01))
      .sort((a, b) => b.score - a.score || a.record.tool.name.localeCompare(b.record.tool.name));
    const searchDone = performance.now();
    let rerankMs = 0;
    if (options.reranker && scores.length) {
      const at = performance.now();
      const pool = scores.slice(0, options.rerankPoolSize ?? Math.max(k, 20));
      const values = await options.reranker.rerank(
        query,
        pool.map((x) => x.record.tool),
      );
      if (values.length !== pool.length || values.some((value) => !Number.isFinite(value))) {
        throw new Error("Reranker returned invalid scores");
      }
      pool.forEach((x, i) => {
        if (Number.isFinite(values[i])) {
          (x as typeof x & { rerankScore?: number }).rerankScore = values[i];
          x.score = values[i]!;
        }
      });
      pool.sort((a, b) => b.score - a.score);
      scores = [...pool, ...scores.slice(pool.length)];
      rerankMs = performance.now() - at;
    }
    let chosen = scores.slice(0, k);
    if ((options.diversity ?? 0) > 0 && scores.length > k) {
      const lambda = 1 - Math.min(1, Math.max(0, options.diversity!));
      const remaining = [...scores],
        diverse: typeof scores = [];
      while (diverse.length < k && remaining.length) {
        let best = 0,
          bestScore = -Infinity;
        for (let i = 0; i < remaining.length; i++) {
          const relevance = remaining[i]!.score;
          const similarity = diverse.length
            ? Math.max(...diverse.map((x) => cosine(x.record.vector, remaining[i]!.record.vector)))
            : 0;
          const mmr = lambda * relevance - (1 - lambda) * similarity;
          if (mmr > bestScore) {
            bestScore = mmr;
            best = i;
          }
        }
        diverse.push(remaining.splice(best, 1)[0]!);
      }
      chosen = diverse;
    }
    if (this.sticky.enabled && options.sessionId) {
      const now = Date.now();
      for (const [id, session] of this.sessions) {
        if (now - session.at > this.sticky.ttlMs) this.sessions.delete(id);
      }
      const old = this.sessions.get(options.sessionId);
      if (old && now - old.at <= this.sticky.ttlMs) {
        const sim = cosine(qv, old.query);
        const oldItems = old.ids
          .map((id) => scores.find((x) => x.record.tool.id === id))
          .filter((item): item is NonNullable<typeof item> => !!item);
        if (sim >= this.sticky.reuseThreshold) {
          const ids = new Set(oldItems.map((x) => x.record.tool.id));
          chosen = [
            ...oldItems.slice(0, k),
            ...scores
              .filter((x) => !ids.has(x.record.tool.id))
              .slice(0, Math.max(0, k - oldItems.length)),
          ];
        } else if (sim >= this.sticky.refreshThreshold) {
          const selectedIds = new Set(chosen.map((x) => x.record.tool.id));
          for (const item of oldItems.slice(0, this.sticky.keep)) {
            const id = item.record.tool.id;
            if (!selectedIds.has(id) && k > 0) {
              if (chosen.length >= k) {
                const removed = chosen.pop();
                if (removed) selectedIds.delete(removed.record.tool.id);
              }
              chosen.push(item);
              selectedIds.add(id);
            }
          }
        }
      }
      if (this.sticky.maxSessions > 0) {
        if (!old && this.sessions.size >= this.sticky.maxSessions) {
          const oldest = [...this.sessions.entries()].sort(([, a], [, b]) => a.at - b.at)[0];
          if (oldest) this.sessions.delete(oldest[0]);
        }
        this.sessions.set(options.sessionId, {
          at: now,
          query: qv,
          ids: chosen.map((x) => x.record.tool.id),
        });
      }
    }
    const selectedIds = new Set(chosen.map((x) => x.record.tool.id));
    const mapScore = (x: (typeof scores)[number]): ToolScore => ({
      id: x.record.tool.id,
      name: x.record.tool.name,
      score: x.score,
      semanticScore: x.semanticScore,
      lexicalScore: x.lexicalScore,
      rerankScore: (x as typeof x & { rerankScore?: number }).rerankScore,
      selected: selectedIds.has(x.record.tool.id),
    });
    const mapped = scores.map(mapScore);
    const trace: SelectionTrace = {
      id: crypto.randomUUID(),
      timestamp: new Date().toISOString(),
      sessionId: options.sessionId,
      query,
      candidateCount,
      filteredCount: records.length,
      selected: chosen.map(mapScore),
      scores: mapped,
      timings: {
        embeddingMs,
        searchMs: searchDone - embAt - embeddingMs,
        rerankMs,
        totalMs: performance.now() - start,
      },
      config: {
        k,
        allowTags: [...allow],
        denyTags: [...deny],
        semanticWeight: semW,
        lexicalWeight: lexW,
      },
    };
    await Promise.all(this.sinks.map((s) => s.emit(trace)));
    return { tools: chosen.map((x) => x.record.tool.original), trace };
  }
}
