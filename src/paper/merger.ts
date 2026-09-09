import { fingerprintTool, stableJson } from "../fingerprint.js";
import { cloneStrictJsonData } from "../json.js";
import { normalizeTools } from "../normalize.js";
import { toolText } from "../text.js";
import type { CanonicalTool, EmbeddingProvider, JsonSchema, ToolTextConfig } from "../types.js";

export const PAPER_2026_DEFAULTS = Object.freeze({
  candidateNeighbors: 30,
  candidateThreshold: 0.82,
  autoCorrectionPasses: 1,
});

export const PAPER_2026_MERGER_RESOURCE_DEFAULTS = Object.freeze({
  modelConcurrency: 8,
  maxCatalogSize: 1000,
  maxEmbeddingDimensions: 4096,
  maxDescriptorChars: 16384,
  maxClassifierCalls: 30000,
});

export interface RelationshipClassification {
  similar: boolean;
  reason?: string;
}

export interface RelationshipClassifier {
  classify(
    target: CanonicalTool,
    candidate: CanonicalTool,
  ): Promise<RelationshipClassification> | RelationshipClassification;
}

export type ClusterValidation =
  | { merge: true; reason?: string }
  | { merge: false; clusters: string[][]; reason?: string };

export interface ClusterValidator {
  validate(cluster: readonly CanonicalTool[]): Promise<ClusterValidation> | ClusterValidation;
}

export interface SynthesizedDescriptor {
  description: string;
}

export interface DescriptorSynthesizer {
  synthesize(
    representative: CanonicalTool,
    members: readonly CanonicalTool[],
  ): Promise<SynthesizedDescriptor> | SynthesizedDescriptor;
}

export interface ToolMergerOptions {
  embedder: EmbeddingProvider;
  classifier: RelationshipClassifier;
  validator?: ClusterValidator;
  synthesizer?: DescriptorSynthesizer;
  candidateCount?: number;
  similarityThreshold?: number;
  autoCorrectionPasses?: number;
  allowCrossNamespaceCandidates?: boolean;
  text?: ToolTextConfig;
  modelConcurrency?: number;
  maxCatalogSize?: number;
  maxEmbeddingDimensions?: number;
  maxDescriptorChars?: number;
  maxClassifierCalls?: number;
}

export interface CandidatePair {
  ids: [string, string];
  similarity: number;
}

export interface MergedToolDescriptor {
  id: string;
  name: string;
  description: string;
  inputSchema: JsonSchema;
  tags: string[];
  namespace?: string;
  representativeId: string;
  memberIds: string[];
}

export interface MergeManifestEntry<T> {
  mergedId: string;
  originals: readonly T[];
}

export class MergeManifest<T> {
  readonly size: number;
  readonly #originalsByMergedId = new Map<string, readonly T[]>();

  constructor(entries: readonly MergeManifestEntry<T>[]) {
    for (const entry of entries) {
      if (this.#originalsByMergedId.has(entry.mergedId)) {
        throw new Error(`Merge manifest id collision: ${entry.mergedId}`);
      }
      this.#originalsByMergedId.set(entry.mergedId, [...entry.originals]);
    }
    this.size = this.#originalsByMergedId.size;
  }

  resolve(mergedId: string): readonly T[] {
    const originals = this.#originalsByMergedId.get(mergedId);
    if (!originals) throw new Error(`Unknown merged descriptor id: ${mergedId}`);
    return [...originals];
  }
}

export interface MergeResult<T> {
  originals: CanonicalTool<T>[];
  candidatePairs: CandidatePair[];
  components: CanonicalTool<T>[][];
  merged: MergedToolDescriptor[];
  manifest: MergeManifest<T>;
}

interface IndexedPair {
  left: number;
  right: number;
  similarity: number;
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === "object" && !Array.isArray(value);

function assertSafeInteger(value: number, name: string, minimum: number): void {
  if (!Number.isSafeInteger(value) || value < minimum) {
    const qualification = minimum === 0 ? "a non-negative" : "a positive";
    throw new RangeError(`${name} must be ${qualification} safe integer`);
  }
}

function validateEmbeddings(
  vectors: number[][],
  expectedCount: number,
  maxEmbeddingDimensions: number,
): void {
  if (!Array.isArray(vectors) || vectors.length !== expectedCount) {
    throw new TypeError("Invalid embedding output: vector count does not match tool count");
  }
  if (expectedCount === 0) return;
  const dimension = vectors[0]?.length ?? 0;
  if (dimension === 0) throw new TypeError("Invalid embedding output: vectors must not be empty");
  if (dimension > maxEmbeddingDimensions) {
    throw new RangeError(
      `Invalid embedding output: vector dimensions exceed maxEmbeddingDimensions (${maxEmbeddingDimensions})`,
    );
  }
  for (const vector of vectors) {
    if (!Array.isArray(vector) || vector.length !== dimension) {
      throw new TypeError("Invalid embedding output: vector dimensions must match");
    }
    if (!vector.every(Number.isFinite)) {
      throw new TypeError("Invalid embedding output: vector values must be finite");
    }
    const normSquared = vector.reduce((sum, value) => sum + value * value, 0);
    if (!Number.isFinite(normSquared) || normSquared === 0) {
      throw new TypeError("Invalid embedding output: vectors must have a finite non-zero norm");
    }
  }
}

function cosine(left: readonly number[], right: readonly number[]): number {
  let dot = 0;
  let leftNorm = 0;
  let rightNorm = 0;
  for (let index = 0; index < left.length; index += 1) {
    const leftValue = left[index]!;
    const rightValue = right[index]!;
    dot += leftValue * rightValue;
    leftNorm += leftValue * leftValue;
    rightNorm += rightValue * rightValue;
  }
  const score = dot / Math.sqrt(leftNorm * rightNorm);
  if (!Number.isFinite(score)) throw new TypeError("Invalid embedding cosine similarity");
  return Math.max(-1, Math.min(1, score));
}

function compareNeighbors(a: IndexedPair, b: IndexedPair): number {
  return b.similarity - a.similarity || a.left - b.left || a.right - b.right;
}

function candidatePairs<T>(
  tools: readonly CanonicalTool<T>[],
  vectors: readonly number[][],
  count: number,
  threshold: number,
  allowCrossNamespaceCandidates: boolean,
): IndexedPair[] {
  if (count === 0) return [];
  const deduplicated = new Map<string, IndexedPair>();
  for (let target = 0; target < tools.length; target += 1) {
    const targetTool = tools[target]!;
    const neighbors: IndexedPair[] = [];
    for (let candidate = 0; candidate < tools.length; candidate += 1) {
      if (candidate === target) continue;
      const candidateTool = tools[candidate]!;
      if (!allowCrossNamespaceCandidates && targetTool.namespace !== candidateTool.namespace)
        continue;
      const similarity = cosine(vectors[target]!, vectors[candidate]!);
      if (similarity < threshold) continue;
      const pair = {
        left: Math.min(target, candidate),
        right: Math.max(target, candidate),
        similarity,
      };
      let insertion = neighbors.length;
      while (insertion > 0 && compareNeighbors(pair, neighbors[insertion - 1]!) < 0) {
        insertion -= 1;
      }
      if (insertion < count) {
        neighbors.splice(insertion, 0, pair);
        if (neighbors.length > count) neighbors.pop();
      }
    }
    for (const pair of neighbors) {
      const key = `${pair.left}:${pair.right}`;
      const previous = deduplicated.get(key);
      if (!previous || pair.similarity > previous.similarity) deduplicated.set(key, pair);
    }
  }
  return [...deduplicated.values()].sort((a, b) => a.left - b.left || a.right - b.right);
}

type OrderedOutcome<T> = { ok: true; value: T } | { ok: false; error: unknown };

async function mapConcurrentOrdered<T, U>(
  values: readonly T[],
  concurrency: number,
  map: (value: T, index: number) => Promise<U> | U,
): Promise<U[]> {
  if (values.length === 0) return [];
  const outcomes = Array.from<OrderedOutcome<U> | undefined>({ length: values.length });
  let nextIndex = 0;
  const workers = Array.from({ length: Math.min(concurrency, values.length) }, async () => {
    while (nextIndex < values.length) {
      const index = nextIndex;
      nextIndex += 1;
      try {
        outcomes[index] = { ok: true, value: await map(values[index]!, index) };
      } catch (error) {
        outcomes[index] = { ok: false, error };
      }
    }
  });
  await Promise.all(workers);
  const results: U[] = [];
  for (const outcome of outcomes) {
    if (!outcome) throw new Error("Ordered concurrency task did not complete");
    if (!outcome.ok) throw outcome.error;
    results.push(outcome.value);
  }
  return results;
}

function assertCanonicalDescriptorSize<T>(
  tool: CanonicalTool<T>,
  maxDescriptorChars: number,
): void {
  const payload = stableJson({
    id: tool.id,
    name: tool.name,
    description: tool.description,
    inputSchema: tool.inputSchema,
    tags: tool.tags,
    ...(tool.namespace === undefined ? {} : { namespace: tool.namespace }),
  });
  if (payload.length > maxDescriptorChars) {
    throw new RangeError(
      `Canonical tool descriptor exceeds maxDescriptorChars (${maxDescriptorChars}): ${tool.name}`,
    );
  }
}

function embeddingText<T>(
  tool: CanonicalTool<T>,
  config: ToolTextConfig,
  maxDescriptorChars: number,
): string {
  const base = toolText(tool, {
    ...config,
    preprocessors: [],
    truncate: null as unknown as number,
  });
  if (base.length > maxDescriptorChars) {
    throw new RangeError(
      `Tool descriptor exceeds maxDescriptorChars (${maxDescriptorChars}): ${tool.name}`,
    );
  }
  const preprocessors = (config.preprocessors ?? []).map((preprocessor) => (text: string) => {
    const output = preprocessor(text);
    if (typeof output !== "string") {
      throw new TypeError("Tool text preprocessor must return a string");
    }
    if (output.length > maxDescriptorChars) {
      throw new RangeError(
        `Preprocessed tool text exceeds maxDescriptorChars (${maxDescriptorChars}): ${tool.name}`,
      );
    }
    return output;
  });
  const text = toolText(tool, { ...config, preprocessors });
  if (text.length > maxDescriptorChars) {
    throw new RangeError(
      `Embedding text exceeds maxDescriptorChars (${maxDescriptorChars}): ${tool.name}`,
    );
  }
  return text;
}

function connectedComponents<T>(tools: readonly CanonicalTool<T>[], edges: readonly IndexedPair[]) {
  const adjacency = Array.from({ length: tools.length }, () => new Set<number>());
  for (const edge of edges) {
    adjacency[edge.left]!.add(edge.right);
    adjacency[edge.right]!.add(edge.left);
  }
  const visited = new Set<number>();
  const components: CanonicalTool<T>[][] = [];
  for (let start = 0; start < tools.length; start += 1) {
    if (visited.has(start)) continue;
    const pending = [start];
    const indices: number[] = [];
    visited.add(start);
    while (pending.length > 0) {
      const current = pending.shift()!;
      indices.push(current);
      for (const neighbor of [...adjacency[current]!].sort((a, b) => a - b)) {
        if (!visited.has(neighbor)) {
          visited.add(neighbor);
          pending.push(neighbor);
        }
      }
    }
    indices.sort((a, b) => a - b);
    components.push(indices.map((index) => tools[index]!));
  }
  return components;
}

function validateCorrection<T>(
  cluster: readonly CanonicalTool<T>[],
  output: unknown,
): CanonicalTool<T>[][] | null {
  if (!isRecord(output) || typeof output.merge !== "boolean") {
    throw new TypeError("Invalid correction output: expected a merge boolean");
  }
  if (output.reason !== undefined && typeof output.reason !== "string") {
    throw new TypeError("Invalid correction output: reason must be a string");
  }
  if (output.merge) return null;
  if (!Array.isArray(output.clusters)) {
    throw new TypeError("Invalid correction output: clusters must be an array");
  }
  const byId = new Map(cluster.map((tool) => [tool.id, tool]));
  const seen = new Set<string>();
  const corrected: CanonicalTool<T>[][] = [];
  for (const proposed of output.clusters) {
    if (!Array.isArray(proposed) || proposed.length === 0) {
      throw new TypeError("Invalid correction partition: groups must be non-empty arrays");
    }
    const group: CanonicalTool<T>[] = [];
    for (const id of proposed) {
      if (typeof id !== "string" || !byId.has(id)) {
        throw new TypeError(`Invalid correction partition: unknown id ${String(id)}`);
      }
      if (seen.has(id)) throw new TypeError(`Invalid correction partition: duplicate id ${id}`);
      seen.add(id);
      group.push(byId.get(id)!);
    }
    corrected.push(group);
  }
  for (const tool of cluster) {
    if (!seen.has(tool.id)) corrected.push([tool]);
  }
  return corrected;
}

function representative<T>(cluster: readonly CanonicalTool<T>[]): CanonicalTool<T> {
  return [...cluster].sort(
    (a, b) =>
      a.name.length - b.name.length || a.name.localeCompare(b.name) || a.id.localeCompare(b.id),
  )[0]!;
}

function defineEnumerable(target: Record<string, unknown>, key: string, value: unknown): void {
  Object.defineProperty(target, key, {
    value,
    enumerable: true,
    configurable: true,
    writable: true,
  });
}

function validateClonedObjectSchema(
  schema: unknown,
  context: string,
): asserts schema is JsonSchema {
  if (!isRecord(schema) || (schema.type !== undefined && schema.type !== "object")) {
    throw new TypeError(`Invalid ${context}: only object JSON Schemas are supported`);
  }
  if (schema.properties !== undefined && !isRecord(schema.properties)) {
    throw new TypeError(`Invalid ${context}: properties must be an object`);
  }
  if (
    schema.required !== undefined &&
    (!Array.isArray(schema.required) || !schema.required.every((name) => typeof name === "string"))
  ) {
    throw new TypeError(`Invalid ${context}: required must contain only strings`);
  }
}

function cloneObjectSchema(schema: unknown, context: string): JsonSchema {
  const clone = cloneStrictJsonData(schema, context);
  validateClonedObjectSchema(clone, context);
  return clone;
}

function consolidateSchemas<T>(cluster: readonly CanonicalTool<T>[]): JsonSchema {
  const schemas = cluster.map((tool) =>
    cloneObjectSchema(tool.inputSchema, "member descriptor schema"),
  );
  if (schemas.length === 1) return schemas[0]!;

  const definitions = new Map<string, Map<string, unknown>>();
  const requiredSets: Set<string>[] = [];
  const hoisted = new Map<"$defs" | "definitions", Map<string, [string, unknown]>>([
    ["$defs", new Map()],
    ["definitions", new Map()],
  ]);
  for (const schema of schemas) {
    const properties = (schema.properties ?? {}) as Record<string, unknown>;
    for (const [name, definition] of Object.entries(properties)) {
      const serialized = stableJson(definition);
      let variants = definitions.get(name);
      if (!variants) {
        variants = new Map();
        definitions.set(name, variants);
      }
      variants.set(serialized, definition);
    }
    requiredSets.push(new Set((schema.required as string[] | undefined) ?? []));
    for (const keyword of ["$defs", "definitions"] as const) {
      const source = schema[keyword];
      if (source === undefined) continue;
      if (!isRecord(source)) {
        throw new TypeError(`Invalid member descriptor schema: ${keyword} must be an object`);
      }
      for (const [name, definition] of Object.entries(source)) {
        const serialized = stableJson(definition);
        const previous = hoisted.get(keyword)!.get(name);
        if (previous && previous[0] !== serialized) {
          throw new TypeError(`Conflicting ${keyword} definition: ${name}`);
        }
        hoisted.get(keyword)!.set(name, [serialized, definition]);
      }
    }
  }

  const properties: Record<string, unknown> = {};
  for (const name of [...definitions.keys()].sort((a, b) => a.localeCompare(b))) {
    const variants = [...definitions.get(name)!.entries()]
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([, definition]) => definition);
    defineEnumerable(properties, name, variants.length === 1 ? variants[0] : { anyOf: variants });
  }
  const required = [...definitions.keys()]
    .filter((name) => requiredSets.every((set) => set.has(name)))
    .sort((a, b) => a.localeCompare(b));
  const consolidated: JsonSchema = { type: "object", properties, required };
  for (const keyword of ["$defs", "definitions"] as const) {
    const entries = hoisted.get(keyword)!;
    if (entries.size === 0) continue;
    const target: Record<string, unknown> = {};
    for (const name of [...entries.keys()].sort((a, b) => a.localeCompare(b))) {
      defineEnumerable(target, name, entries.get(name)![1]);
    }
    consolidated[keyword] = target;
  }
  consolidated.anyOf = [...new Map(schemas.map((schema) => [stableJson(schema), schema])).entries()]
    .sort(
      ([leftSerialized, left], [rightSerialized, right]) =>
        stableJson(left.properties ?? {}).localeCompare(stableJson(right.properties ?? {})) ||
        leftSerialized.localeCompare(rightSerialized),
    )
    .map(([, schema]) => schema);
  return consolidated;
}

function validateSynthesizedDescriptor(output: unknown): SynthesizedDescriptor {
  const clone = cloneStrictJsonData(output, "synthesized descriptor");
  if (!isRecord(clone)) throw new TypeError("Invalid synthesized descriptor: expected an object");
  const supportedDescriptorKeys = new Set(["description"]);
  for (const key of Object.keys(clone)) {
    if (!supportedDescriptorKeys.has(key)) {
      throw new TypeError(`Invalid synthesized descriptor: unsupported field ${key}`);
    }
  }
  if (typeof clone.description !== "string" || clone.description.trim().length === 0) {
    throw new TypeError("Invalid synthesized descriptor: description must be non-empty");
  }
  return { description: clone.description };
}

function assertFinalIntegrity<T>(
  originals: readonly CanonicalTool<T>[],
  components: readonly (readonly CanonicalTool<T>[])[],
): void {
  const expected = new Set(originals.map((tool) => tool.id));
  const actual = new Set<string>();
  for (const component of components) {
    if (component.length === 0) throw new Error("Final merge integrity failure: empty component");
    for (const tool of component) {
      if (!expected.has(tool.id) || actual.has(tool.id)) {
        throw new Error(`Final merge integrity failure for tool ${tool.id}`);
      }
      actual.add(tool.id);
    }
  }
  if (actual.size !== expected.size)
    throw new Error("Final merge integrity failure: omitted tools");
}

export class ToolMerger {
  readonly #options: Required<
    Pick<
      ToolMergerOptions,
      | "candidateCount"
      | "similarityThreshold"
      | "autoCorrectionPasses"
      | "allowCrossNamespaceCandidates"
      | "text"
      | "modelConcurrency"
      | "maxCatalogSize"
      | "maxEmbeddingDimensions"
      | "maxDescriptorChars"
      | "maxClassifierCalls"
    >
  > &
    Omit<
      ToolMergerOptions,
      | "candidateCount"
      | "similarityThreshold"
      | "autoCorrectionPasses"
      | "allowCrossNamespaceCandidates"
      | "text"
      | "modelConcurrency"
      | "maxCatalogSize"
      | "maxEmbeddingDimensions"
      | "maxDescriptorChars"
      | "maxClassifierCalls"
    >;

  constructor(options: ToolMergerOptions) {
    const candidateCount = options.candidateCount ?? PAPER_2026_DEFAULTS.candidateNeighbors;
    const similarityThreshold =
      options.similarityThreshold ?? PAPER_2026_DEFAULTS.candidateThreshold;
    const autoCorrectionPasses =
      options.autoCorrectionPasses ?? PAPER_2026_DEFAULTS.autoCorrectionPasses;
    const modelConcurrency =
      options.modelConcurrency ?? PAPER_2026_MERGER_RESOURCE_DEFAULTS.modelConcurrency;
    const maxCatalogSize =
      options.maxCatalogSize ?? PAPER_2026_MERGER_RESOURCE_DEFAULTS.maxCatalogSize;
    const maxEmbeddingDimensions =
      options.maxEmbeddingDimensions ?? PAPER_2026_MERGER_RESOURCE_DEFAULTS.maxEmbeddingDimensions;
    const maxDescriptorChars =
      options.maxDescriptorChars ?? PAPER_2026_MERGER_RESOURCE_DEFAULTS.maxDescriptorChars;
    const maxClassifierCalls =
      options.maxClassifierCalls ?? PAPER_2026_MERGER_RESOURCE_DEFAULTS.maxClassifierCalls;
    assertSafeInteger(candidateCount, "candidateCount", 0);
    assertSafeInteger(autoCorrectionPasses, "autoCorrectionPasses", 0);
    assertSafeInteger(modelConcurrency, "modelConcurrency", 1);
    assertSafeInteger(maxCatalogSize, "maxCatalogSize", 1);
    assertSafeInteger(maxEmbeddingDimensions, "maxEmbeddingDimensions", 1);
    assertSafeInteger(maxDescriptorChars, "maxDescriptorChars", 0);
    assertSafeInteger(maxClassifierCalls, "maxClassifierCalls", 0);
    if (options.text?.truncate !== undefined) {
      assertSafeInteger(options.text.truncate, "text.truncate", 0);
    }
    if (
      !Number.isFinite(similarityThreshold) ||
      similarityThreshold < -1 ||
      similarityThreshold > 1
    ) {
      throw new RangeError("similarityThreshold must be a finite number between -1 and 1");
    }
    this.#options = {
      ...options,
      candidateCount,
      similarityThreshold,
      autoCorrectionPasses,
      allowCrossNamespaceCandidates: options.allowCrossNamespaceCandidates ?? false,
      text: options.text ?? {},
      modelConcurrency,
      maxCatalogSize,
      maxEmbeddingDimensions,
      maxDescriptorChars,
      maxClassifierCalls,
    };
  }

  async merge<T>(tools: readonly T[]): Promise<MergeResult<T>> {
    if (tools.length > this.#options.maxCatalogSize) {
      throw new RangeError(`Tool catalog exceeds maxCatalogSize (${this.#options.maxCatalogSize})`);
    }
    const originals = normalizeTools(tools);
    const ids = new Set<string>();
    for (const tool of originals) {
      if (ids.has(tool.id)) throw new Error(`Canonical tool id collision: ${tool.id}`);
      ids.add(tool.id);
    }
    for (const tool of originals) {
      assertCanonicalDescriptorSize(tool, this.#options.maxDescriptorChars);
    }
    if (originals.length === 0) {
      return {
        originals,
        candidatePairs: [],
        components: [],
        merged: [],
        manifest: new MergeManifest([]),
      };
    }

    const texts = originals.map((tool) =>
      embeddingText(tool, this.#options.text, this.#options.maxDescriptorChars),
    );
    const vectors = await this.#options.embedder.embed(texts);
    validateEmbeddings(vectors, originals.length, this.#options.maxEmbeddingDimensions);
    const candidates = candidatePairs(
      originals,
      vectors,
      this.#options.candidateCount,
      this.#options.similarityThreshold,
      this.#options.allowCrossNamespaceCandidates,
    );
    if (candidates.length > this.#options.maxClassifierCalls) {
      throw new RangeError(
        `Candidate pairs exceed maxClassifierCalls (${this.#options.maxClassifierCalls})`,
      );
    }
    const classifications = await mapConcurrentOrdered(
      candidates,
      this.#options.modelConcurrency,
      async (pair) => {
        const output: unknown = await this.#options.classifier.classify(
          originals[pair.left]!,
          originals[pair.right]!,
        );
        if (
          !isRecord(output) ||
          typeof output.similar !== "boolean" ||
          (output.reason !== undefined && typeof output.reason !== "string")
        ) {
          throw new TypeError(
            "Invalid classifier output: expected { similar: boolean, reason?: string }",
          );
        }
        return output.similar;
      },
    );
    const edges = candidates.filter((_pair, index) => classifications[index]);

    let components = connectedComponents(originals, edges);
    if (this.#options.validator) {
      for (let pass = 0; pass < this.#options.autoCorrectionPasses; pass += 1) {
        const corrections = await mapConcurrentOrdered(
          components,
          this.#options.modelConcurrency,
          async (cluster) => {
            if (cluster.length === 1) return [cluster];
            const output: unknown = await this.#options.validator!.validate(cluster);
            return validateCorrection(cluster, output) ?? [cluster];
          },
        );
        components = corrections.flat();
      }
    }
    assertFinalIntegrity(originals, components);

    const prepared = components.map((cluster) => ({
      cluster,
      selected: representative(cluster),
      consolidatedSchema: consolidateSchemas(cluster),
    }));
    const synthesized = await mapConcurrentOrdered(
      prepared,
      this.#options.modelConcurrency,
      async ({ cluster, selected }) => {
        const descriptor = this.#options.synthesizer
          ? validateSynthesizedDescriptor(
              await this.#options.synthesizer.synthesize(selected, cluster),
            )
          : { description: selected.description };
        if (descriptor.description.length > this.#options.maxDescriptorChars) {
          throw new RangeError(
            `Synthesized descriptor exceeds maxDescriptorChars (${this.#options.maxDescriptorChars})`,
          );
        }
        return descriptor;
      },
    );
    const merged: MergedToolDescriptor[] = [];
    const manifestEntries: MergeManifestEntry<T>[] = [];
    const mergedIds = new Set<string>();
    for (let index = 0; index < prepared.length; index += 1) {
      const { cluster, selected, consolidatedSchema } = prepared[index]!;
      const descriptor = synthesized[index]!;
      const tags = [...new Set(cluster.flatMap((tool) => tool.tags))].sort((a, b) =>
        a.localeCompare(b),
      );
      const namespace = cluster.every((tool) => tool.namespace === cluster[0]!.namespace)
        ? cluster[0]!.namespace
        : undefined;
      const id = fingerprintTool(
        selected.name,
        descriptor.description,
        consolidatedSchema,
        namespace,
      );
      if (mergedIds.has(id)) throw new Error(`Merged descriptor id collision: ${id}`);
      mergedIds.add(id);
      merged.push({
        id,
        name: selected.name,
        description: descriptor.description,
        inputSchema: consolidatedSchema,
        tags: [...tags],
        ...(namespace === undefined ? {} : { namespace }),
        representativeId: selected.id,
        memberIds: cluster.map((tool) => tool.id),
      });
      manifestEntries.push({ mergedId: id, originals: cluster.map((tool) => tool.original) });
    }

    return {
      originals,
      candidatePairs: candidates.map((pair) => ({
        ids: [originals[pair.left]!.id, originals[pair.right]!.id],
        similarity: pair.similarity,
      })),
      components,
      merged,
      manifest: new MergeManifest(manifestEntries),
    };
  }
}
