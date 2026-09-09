import { fingerprintTool, stableJson } from "../fingerprint.js";
import { normalizeTools } from "../normalize.js";
import { toolText } from "../text.js";
import type { CanonicalTool, EmbeddingProvider, JsonSchema, ToolTextConfig } from "../types.js";

export const PAPER_2026_DEFAULTS = Object.freeze({
  candidateNeighbors: 30,
  candidateThreshold: 0.82,
  autoCorrectionPasses: 1,
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
  inputSchema: JsonSchema;
  tags?: string[];
  namespace?: string;
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

function assertNonNegativeInteger(value: number, name: string): void {
  if (!Number.isInteger(value) || value < 0) {
    throw new RangeError(`${name} must be a non-negative integer`);
  }
}

function validateEmbeddings(vectors: number[][], expectedCount: number): void {
  if (!Array.isArray(vectors) || vectors.length !== expectedCount) {
    throw new TypeError("Invalid embedding output: vector count does not match tool count");
  }
  if (expectedCount === 0) return;
  const dimension = vectors[0]?.length ?? 0;
  if (dimension === 0) throw new TypeError("Invalid embedding output: vectors must not be empty");
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

function candidatePairs<T>(
  tools: readonly CanonicalTool<T>[],
  vectors: readonly number[][],
  count: number,
  threshold: number,
  allowCrossNamespaceCandidates: boolean,
): IndexedPair[] {
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
      neighbors.push({
        left: Math.min(target, candidate),
        right: Math.max(target, candidate),
        similarity,
      });
    }
    neighbors.sort((a, b) => b.similarity - a.similarity || a.left - b.left || a.right - b.right);
    for (const pair of neighbors.slice(0, count)) {
      const key = `${pair.left}:${pair.right}`;
      const previous = deduplicated.get(key);
      if (!previous || pair.similarity > previous.similarity) deduplicated.set(key, pair);
    }
  }
  return [...deduplicated.values()].sort((a, b) => a.left - b.left || a.right - b.right);
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

const forbiddenDescriptorKeys = new Set([
  "code",
  "execute",
  "handler",
  "implementation",
  "function",
]);

function assertJsonSafe(value: unknown, context: string, seen = new Set<object>()): void {
  if (
    value === null ||
    typeof value === "string" ||
    typeof value === "boolean" ||
    (typeof value === "number" && Number.isFinite(value))
  ) {
    return;
  }
  if (typeof value !== "object") {
    throw new TypeError(`Invalid ${context}: values must be finite JSON data`);
  }
  if (seen.has(value)) throw new TypeError(`Invalid ${context}: cyclic data is forbidden`);
  seen.add(value);
  if (Array.isArray(value)) {
    for (const item of value) assertJsonSafe(item, context, seen);
  } else {
    for (const [key, item] of Object.entries(value)) {
      if (forbiddenDescriptorKeys.has(key.toLowerCase())) {
        throw new TypeError(`Invalid ${context}: executable field ${key} is forbidden`);
      }
      assertJsonSafe(item, context, seen);
    }
  }
  seen.delete(value);
}

function validateObjectSchema(schema: unknown, context: string): asserts schema is JsonSchema {
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
  assertJsonSafe(schema, context);
}

function consolidateSchemas<T>(cluster: readonly CanonicalTool<T>[]): JsonSchema {
  const definitions = new Map<string, Map<string, unknown>>();
  const requiredSets: Set<string>[] = [];
  for (const tool of cluster) {
    validateObjectSchema(tool.inputSchema, "member descriptor schema");
    const properties = (tool.inputSchema.properties ?? {}) as Record<string, unknown>;
    for (const [name, definition] of Object.entries(properties)) {
      const serialized = stableJson(definition);
      let variants = definitions.get(name);
      if (!variants) {
        variants = new Map();
        definitions.set(name, variants);
      }
      variants.set(serialized, definition);
    }
    requiredSets.push(new Set((tool.inputSchema.required as string[] | undefined) ?? []));
  }

  const properties: Record<string, unknown> = {};
  for (const name of [...definitions.keys()].sort((a, b) => a.localeCompare(b))) {
    const variants = [...definitions.get(name)!.entries()]
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([, definition]) => definition);
    properties[name] = variants.length === 1 ? variants[0] : { anyOf: variants };
  }
  const required = [...definitions.keys()]
    .filter((name) => requiredSets.every((set) => set.has(name)))
    .sort((a, b) => a.localeCompare(b));
  return { type: "object", properties, required };
}

function validateSynthesizedDescriptor(output: unknown): SynthesizedDescriptor {
  if (!isRecord(output)) throw new TypeError("Invalid synthesized descriptor: expected an object");
  for (const key of Object.keys(output)) {
    if (forbiddenDescriptorKeys.has(key.toLowerCase())) {
      throw new TypeError(`Invalid synthesized descriptor: executable field ${key} is forbidden`);
    }
  }
  if (typeof output.description !== "string" || output.description.trim().length === 0) {
    throw new TypeError("Invalid synthesized descriptor: description must be non-empty");
  }
  validateObjectSchema(output.inputSchema, "synthesized descriptor schema");
  assertJsonSafe(output, "synthesized descriptor");
  if (
    output.tags !== undefined &&
    (!Array.isArray(output.tags) || !output.tags.every((tag) => typeof tag === "string"))
  ) {
    throw new TypeError("Invalid synthesized descriptor: tags must contain only strings");
  }
  if (
    output.namespace !== undefined &&
    (typeof output.namespace !== "string" || output.namespace.length === 0)
  ) {
    throw new TypeError("Invalid synthesized descriptor: namespace must be a non-empty string");
  }
  return {
    description: output.description,
    inputSchema: output.inputSchema,
    ...(output.tags === undefined ? {} : { tags: [...output.tags] as string[] }),
    ...(output.namespace === undefined ? {} : { namespace: output.namespace }),
  };
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
    >
  > &
    Omit<
      ToolMergerOptions,
      | "candidateCount"
      | "similarityThreshold"
      | "autoCorrectionPasses"
      | "allowCrossNamespaceCandidates"
      | "text"
    >;

  constructor(options: ToolMergerOptions) {
    const candidateCount = options.candidateCount ?? PAPER_2026_DEFAULTS.candidateNeighbors;
    const similarityThreshold =
      options.similarityThreshold ?? PAPER_2026_DEFAULTS.candidateThreshold;
    const autoCorrectionPasses =
      options.autoCorrectionPasses ?? PAPER_2026_DEFAULTS.autoCorrectionPasses;
    assertNonNegativeInteger(candidateCount, "candidateCount");
    assertNonNegativeInteger(autoCorrectionPasses, "autoCorrectionPasses");
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
    };
  }

  async merge<T>(tools: readonly T[]): Promise<MergeResult<T>> {
    const originals = normalizeTools(tools);
    const ids = new Set<string>();
    for (const tool of originals) {
      if (ids.has(tool.id)) throw new Error(`Canonical tool id collision: ${tool.id}`);
      ids.add(tool.id);
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

    const vectors = await this.#options.embedder.embed(
      originals.map((tool) => toolText(tool, this.#options.text)),
    );
    validateEmbeddings(vectors, originals.length);
    const candidates = candidatePairs(
      originals,
      vectors,
      this.#options.candidateCount,
      this.#options.similarityThreshold,
      this.#options.allowCrossNamespaceCandidates,
    );
    const edges: IndexedPair[] = [];
    for (const pair of candidates) {
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
      if (output.similar) edges.push(pair);
    }

    let components = connectedComponents(originals, edges);
    if (this.#options.validator) {
      for (let pass = 0; pass < this.#options.autoCorrectionPasses; pass += 1) {
        const corrected: CanonicalTool<T>[][] = [];
        for (const cluster of components) {
          if (cluster.length === 1) {
            corrected.push(cluster);
            continue;
          }
          const output: unknown = await this.#options.validator.validate(cluster);
          corrected.push(...(validateCorrection(cluster, output) ?? [cluster]));
        }
        components = corrected;
      }
    }
    assertFinalIntegrity(originals, components);

    const merged: MergedToolDescriptor[] = [];
    const manifestEntries: MergeManifestEntry<T>[] = [];
    const mergedIds = new Set<string>();
    for (const cluster of components) {
      const selected = representative(cluster);
      const consolidatedSchema = consolidateSchemas(cluster);
      const synthesized = this.#options.synthesizer
        ? validateSynthesizedDescriptor(
            await this.#options.synthesizer.synthesize(selected, cluster),
          )
        : {
            description: selected.description,
            inputSchema: consolidatedSchema,
            tags: selected.tags,
            ...(selected.namespace === undefined ? {} : { namespace: selected.namespace }),
          };
      const tags = synthesized.tags ?? selected.tags;
      const namespace = synthesized.namespace ?? selected.namespace;
      const id = fingerprintTool(
        selected.name,
        synthesized.description,
        consolidatedSchema,
        namespace,
      );
      if (mergedIds.has(id)) throw new Error(`Merged descriptor id collision: ${id}`);
      mergedIds.add(id);
      merged.push({
        id,
        name: selected.name,
        description: synthesized.description,
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
