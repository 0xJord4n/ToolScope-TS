import { messagesToQueryText, ToolIndex } from "../tool-index.js";
import type { FilterOptions, IndexedTool, Reranker } from "../types.js";

export const PAPER_2026_RETRIEVER_DEFAULTS = {
  alpha: 1,
  candidatePoolSize: 50,
  epsilon: 1e-12,
} as const;

export interface QueryDecomposer {
  decompose(query: string): string[] | Promise<string[]>;
}

type RetrievalFilters = Pick<FilterOptions, "allowTags" | "denyTags" | "namespace" | "policy">;

export interface MultiQueryRetrieverOptions extends RetrievalFilters {
  index: ToolIndex;
  decomposer: QueryDecomposer;
  reranker: Reranker;
  k: number;
  candidatePoolSize?: number;
  epsilon?: number;
  alpha?: number;
}

export interface MultiQueryRetrieveOptions extends RetrievalFilters {
  k?: number;
  candidatePoolSize?: number;
  epsilon?: number;
  alpha?: number;
}

export interface MultiQueryCandidateTrace {
  id: string;
  name: string;
  query: string;
  stepIndex: number;
  rank: number;
  denseScore: number;
  bm25Score: number;
  hybridScore: number;
  rerankScore: number;
  normalizedScore?: number;
  reserved: boolean;
  selected: boolean;
}

export interface MultiQueryStepTrace {
  query: string;
  candidateCount: number;
  filteredCount: number;
  candidates: MultiQueryCandidateTrace[];
  reserved?: MultiQueryCandidateTrace;
}

export interface MultiQueryRetrievalTrace {
  query: string;
  k: number;
  candidatePoolSize: number;
  alpha: number;
  epsilon: number;
  normalizationScope: "per-query";
  steps: MultiQueryStepTrace[];
  reserved: MultiQueryCandidateTrace[];
  remaining: MultiQueryCandidateTrace[];
  selected: MultiQueryCandidateTrace[];
}

export interface MultiQueryRetrievalResult<T = unknown> {
  tools: T[];
  trace: MultiQueryRetrievalTrace;
}

export interface Bm25Options {
  k1?: number;
  b?: number;
}

const tokenize = (text: string): string[] =>
  text
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter(Boolean);

export function bm25Scores(
  query: string,
  documents: readonly string[],
  options: Bm25Options = {},
): number[] {
  const k1 = options.k1 ?? 1.2;
  const b = options.b ?? 0.75;
  if (!Number.isFinite(k1) || k1 < 0) throw new RangeError("BM25 k1 must be non-negative");
  if (!Number.isFinite(b) || b < 0 || b > 1) {
    throw new RangeError("BM25 b must be between 0 and 1");
  }
  if (documents.length === 0) return [];

  const tokenized = documents.map(tokenize);
  const averageLength =
    tokenized.reduce((total, document) => total + document.length, 0) / tokenized.length || 1;
  const terms = [...new Set(tokenize(query))];
  const documentFrequencies = new Map<string, number>();
  for (const term of terms) {
    documentFrequencies.set(
      term,
      tokenized.reduce((count, document) => count + (document.includes(term) ? 1 : 0), 0),
    );
  }

  return tokenized.map((document) => {
    const frequencies = new Map<string, number>();
    for (const term of document) frequencies.set(term, (frequencies.get(term) ?? 0) + 1);
    let score = 0;
    for (const term of terms) {
      const frequency = frequencies.get(term) ?? 0;
      if (frequency === 0) continue;
      const documentFrequency = documentFrequencies.get(term) ?? 0;
      const inverseDocumentFrequency = Math.log(
        1 + (documents.length - documentFrequency + 0.5) / (documentFrequency + 0.5),
      );
      const lengthNormalization = frequency + k1 * (1 - b + b * (document.length / averageLength));
      score += inverseDocumentFrequency * ((frequency * (k1 + 1)) / lengthNormalization);
    }
    return score;
  });
}

function cosine(left: number[], right: number[]): number {
  if (left.length === 0 || left.length !== right.length) {
    throw new Error("Embedding dimension mismatch");
  }
  let dot = 0;
  let leftMagnitude = 0;
  let rightMagnitude = 0;
  for (let index = 0; index < left.length; index++) {
    const leftValue = left[index]!;
    const rightValue = right[index]!;
    if (!Number.isFinite(leftValue) || !Number.isFinite(rightValue)) {
      throw new Error("Embedding vectors must contain only finite numbers");
    }
    dot += leftValue * rightValue;
    leftMagnitude += leftValue * leftValue;
    rightMagnitude += rightValue * rightValue;
  }
  return dot / (Math.sqrt(leftMagnitude) * Math.sqrt(rightMagnitude) || 1);
}

function validateConfiguration(
  k: number,
  candidatePoolSize: number,
  epsilon: number,
  alpha: number,
): void {
  if (!Number.isInteger(k) || k < 1) throw new RangeError("k must be at least 1");
  if (!Number.isInteger(candidatePoolSize) || candidatePoolSize < 1) {
    throw new RangeError("candidatePoolSize must be at least 1");
  }
  if (!Number.isFinite(epsilon) || epsilon <= 0) {
    throw new RangeError("epsilon must be greater than zero");
  }
  if (!Number.isFinite(alpha) || alpha < 0 || alpha > 1) {
    throw new RangeError("alpha must be between 0 and 1");
  }
}

async function filterRecords(
  records: IndexedTool[],
  options: RetrievalFilters,
): Promise<IndexedTool[]> {
  const allowTags = new Set(options.allowTags ?? []);
  const denyTags = new Set(options.denyTags ?? []);
  const filtered: IndexedTool[] = [];
  for (const record of records) {
    const { tool } = record;
    if (options.namespace && tool.namespace !== options.namespace) continue;
    if (allowTags.size > 0 && !tool.tags.some((tag) => allowTags.has(tag))) continue;
    if (tool.tags.some((tag) => denyTags.has(tag))) continue;
    if (options.policy && !(await options.policy(tool))) continue;
    filtered.push(record);
  }
  return filtered;
}

interface RankedCandidate {
  record: IndexedTool;
  poolRank: number;
  denseScore: number;
  bm25Score: number;
  hybridScore: number;
  rerankScore: number;
}

export class MultiQueryRetriever<T = unknown> {
  private readonly options: MultiQueryRetrieverOptions;

  constructor(options: MultiQueryRetrieverOptions) {
    validateConfiguration(
      options.k,
      options.candidatePoolSize ?? PAPER_2026_RETRIEVER_DEFAULTS.candidatePoolSize,
      options.epsilon ?? PAPER_2026_RETRIEVER_DEFAULTS.epsilon,
      options.alpha ?? PAPER_2026_RETRIEVER_DEFAULTS.alpha,
    );
    this.options = options;
  }

  async retrieve(
    messages: unknown,
    overrides: MultiQueryRetrieveOptions = {},
  ): Promise<MultiQueryRetrievalResult<T>> {
    const k = overrides.k ?? this.options.k;
    const candidatePoolSize =
      overrides.candidatePoolSize ??
      this.options.candidatePoolSize ??
      PAPER_2026_RETRIEVER_DEFAULTS.candidatePoolSize;
    const epsilon =
      overrides.epsilon ?? this.options.epsilon ?? PAPER_2026_RETRIEVER_DEFAULTS.epsilon;
    const alpha = overrides.alpha ?? this.options.alpha ?? PAPER_2026_RETRIEVER_DEFAULTS.alpha;
    validateConfiguration(k, candidatePoolSize, epsilon, alpha);

    const query = messagesToQueryText(messages);
    const decomposition = await this.options.decomposer.decompose(query);
    if (
      !Array.isArray(decomposition) ||
      decomposition.length === 0 ||
      decomposition.some((step) => typeof step !== "string" || step.trim().length === 0)
    ) {
      throw new Error("Invalid query decomposition");
    }
    if (decomposition.length > k) {
      throw new Error("Query decomposition has more steps than k");
    }

    const filters: RetrievalFilters = {
      allowTags: overrides.allowTags ?? this.options.allowTags,
      denyTags: overrides.denyTags ?? this.options.denyTags,
      namespace: overrides.namespace ?? this.options.namespace,
      policy: overrides.policy ?? this.options.policy,
    };
    const catalog = this.options.index.backend.list();
    const records = await filterRecords(catalog, filters);
    const steps: MultiQueryStepTrace[] = [];
    const recordById = new Map(catalog.map((record) => [record.tool.id, record]));

    for (let stepIndex = 0; stepIndex < decomposition.length; stepIndex++) {
      const stepQuery = decomposition[stepIndex]!;
      const vectors = await this.options.index.embedder.embed([stepQuery]);
      if (vectors.length !== 1 || !vectors[0]) {
        throw new Error("Embedder returned the wrong vector count");
      }
      const queryVector = vectors[0];
      if (queryVector.length === 0 || queryVector.some((value) => !Number.isFinite(value))) {
        throw new Error("Embedding vectors must contain only finite numbers");
      }
      const sparseScores = bm25Scores(
        stepQuery,
        records.map((record) => record.text),
      );
      const hybrid = records
        .map((record, index) => {
          const denseScore = cosine(queryVector, record.vector);
          const bm25Score = sparseScores[index]!;
          return {
            record,
            denseScore,
            bm25Score,
            hybridScore: alpha * denseScore + (1 - alpha) * bm25Score,
          };
        })
        .sort(
          (left, right) =>
            right.hybridScore - left.hybridScore ||
            left.record.tool.id.localeCompare(right.record.tool.id),
        )
        .slice(0, candidatePoolSize);
      const rerankerScores = await this.options.reranker.rerank(
        stepQuery,
        hybrid.map((candidate) => candidate.record.tool),
      );
      if (
        rerankerScores.length !== hybrid.length ||
        rerankerScores.some((score) => !Number.isFinite(score))
      ) {
        throw new Error("Reranker returned invalid scores");
      }
      const ranked: RankedCandidate[] = hybrid
        .map((candidate, poolRank) => ({
          ...candidate,
          poolRank,
          rerankScore: rerankerScores[poolRank]!,
        }))
        .sort(
          (left, right) =>
            right.rerankScore - left.rerankScore ||
            left.poolRank - right.poolRank ||
            left.record.tool.id.localeCompare(right.record.tool.id),
        );
      const candidates = ranked.map<MultiQueryCandidateTrace>((candidate, rank) => ({
        id: candidate.record.tool.id,
        name: candidate.record.tool.name,
        query: stepQuery,
        stepIndex,
        rank,
        denseScore: candidate.denseScore,
        bm25Score: candidate.bm25Score,
        hybridScore: candidate.hybridScore,
        rerankScore: candidate.rerankScore,
        reserved: rank === 0,
        selected: false,
      }));
      const remaining = candidates.slice(1);
      if (remaining.length > 0) {
        const scores = remaining.map((candidate) => candidate.rerankScore);
        const minimum = Math.min(...scores);
        const maximum = Math.max(...scores);
        for (const candidate of remaining) {
          candidate.normalizedScore =
            (candidate.rerankScore - minimum) / (maximum - minimum + epsilon);
        }
      }
      steps.push({
        query: stepQuery,
        candidateCount: catalog.length,
        filteredCount: records.length,
        candidates,
        reserved: candidates[0],
      });
    }

    const reserved = steps.flatMap((step) => (step.reserved ? [step.reserved] : []));
    const remaining = steps
      .flatMap((step) => step.candidates.slice(1))
      .sort(
        (left, right) =>
          (right.normalizedScore ?? 0) - (left.normalizedScore ?? 0) ||
          left.stepIndex - right.stepIndex ||
          left.rank - right.rank ||
          left.id.localeCompare(right.id),
      );
    const selected: MultiQueryCandidateTrace[] = [];
    const selectedIds = new Set<string>();
    for (const candidate of [...reserved, ...remaining]) {
      if (selected.length >= k) break;
      if (selectedIds.has(candidate.id)) continue;
      selectedIds.add(candidate.id);
      candidate.selected = true;
      selected.push(candidate);
    }

    const tools = selected.map((candidate) => {
      const record = recordById.get(candidate.id);
      if (!record) throw new Error(`Selected unknown tool id: ${candidate.id}`);
      return record.tool.original as T;
    });
    return {
      tools,
      trace: {
        query,
        k,
        candidatePoolSize,
        alpha,
        epsilon,
        normalizationScope: "per-query",
        steps,
        reserved,
        remaining,
        selected,
      },
    };
  }
}
