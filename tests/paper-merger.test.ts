import { describe, expect, test } from "bun:test";
import {
  MergeManifest,
  PAPER_2026_DEFAULTS,
  PAPER_2026_MERGER_RESOURCE_DEFAULTS,
  ToolMerger,
  type ClusterValidator,
  type DescriptorSynthesizer,
  type RelationshipClassifier,
} from "../src/paper/merger.js";
import {
  buildCorrectionPrompt,
  buildDescriptorSynthesisPrompt,
  buildQueryDecompositionPrompt,
  buildRelationshipPrompt,
} from "../src/paper/prompts.js";
import type { CanonicalTool, EmbeddingProvider } from "../src/types.js";

interface RawTool {
  name: string;
  description: string;
  inputSchema: Record<string, unknown>;
  execute?: () => string;
  tags?: string[];
  namespace?: string;
}

const raw = (name: string, description = name): RawTool => ({
  name,
  description,
  inputSchema: { type: "object" },
  execute: () => name,
});

const embedder = (vectors: readonly (readonly number[])[]): EmbeddingProvider => ({
  async embed() {
    return vectors.map((vector) => [...vector]);
  },
});

const classifier = (
  similar: (left: CanonicalTool, right: CanonicalTool) => boolean = () => true,
): RelationshipClassifier => ({
  async classify(left, right) {
    return { similar: similar(left, right), reason: "concise" };
  },
});

describe("paper-inspired tool merger", () => {
  test("exports separate immutable resource defaults without changing paper defaults", () => {
    expect(PAPER_2026_DEFAULTS).toEqual({
      candidateNeighbors: 30,
      candidateThreshold: 0.82,
      autoCorrectionPasses: 1,
    });
    expect(PAPER_2026_MERGER_RESOURCE_DEFAULTS).toEqual({
      modelConcurrency: 8,
      maxCatalogSize: 1000,
      maxEmbeddingDimensions: 4096,
      maxDescriptorChars: 16384,
      maxClassifierCalls: 30000,
    });
    expect(Object.isFrozen(PAPER_2026_MERGER_RESOURCE_DEFAULTS)).toBe(true);
  });

  test.each([
    ["candidateCount", Number.MAX_SAFE_INTEGER + 1],
    ["autoCorrectionPasses", Number.MAX_SAFE_INTEGER + 1],
    ["modelConcurrency", 0],
    ["modelConcurrency", 1.5],
    ["maxCatalogSize", 0],
    ["maxEmbeddingDimensions", Number.MAX_SAFE_INTEGER + 1],
    ["maxDescriptorChars", -1],
    ["maxClassifierCalls", -1],
  ] as const)("rejects unsafe integer option %s=%s", (name, value) => {
    expect(
      () =>
        new ToolMerger({
          embedder: embedder([]),
          classifier: classifier(),
          [name]: value,
        }),
    ).toThrow(name);
  });

  test.each([-1, 1.5, Number.MAX_SAFE_INTEGER + 1])(
    "rejects unsafe text.truncate=%s",
    (truncate) => {
      expect(
        () =>
          new ToolMerger({
            embedder: embedder([]),
            classifier: classifier(),
            text: { truncate },
          }),
      ).toThrow("text.truncate");
    },
  );

  test("rejects an oversized catalog before normalization or provider calls", async () => {
    let getterCalls = 0;
    let embedCalls = 0;
    const oversized = Object.defineProperty({}, "name", {
      enumerable: true,
      get() {
        getterCalls += 1;
        return "unsafe";
      },
    });
    const merger = new ToolMerger({
      embedder: { embed: async () => (++embedCalls, []) },
      classifier: classifier(),
      maxCatalogSize: 1,
    });

    await expect(merger.merge([oversized, oversized])).rejects.toThrow("maxCatalogSize");
    expect(getterCalls).toBe(0);
    expect(embedCalls).toBe(0);
  });

  test("rejects an oversized canonical descriptor before preprocessors or providers", async () => {
    let preprocessorCalls = 0;
    let embedCalls = 0;
    let classifierCalls = 0;
    let validatorCalls = 0;
    let synthesizerCalls = 0;
    const merger = new ToolMerger({
      embedder: { embed: async () => (++embedCalls, [[1]]) },
      classifier: { classify: async () => (++classifierCalls, { similar: true }) },
      validator: { validate: async () => (++validatorCalls, { merge: true }) },
      synthesizer: {
        synthesize: async () => (++synthesizerCalls, { description: "synthesized" }),
      },
      maxDescriptorChars: 256,
      text: { preprocessors: [(text) => (++preprocessorCalls, text)] },
    });
    const tool = {
      ...raw("a", "b"),
      inputSchema: {
        type: "object",
        properties: { payload: { type: "string", description: "x".repeat(512) } },
      },
    };

    await expect(merger.merge([tool])).rejects.toThrow("maxDescriptorChars");
    expect(preprocessorCalls).toBe(0);
    expect(embedCalls).toBe(0);
    expect(classifierCalls).toBe(0);
    expect(validatorCalls).toBe(0);
    expect(synthesizerCalls).toBe(0);
  });

  test("rejects embedding text expanded by a preprocessor before the embedder", async () => {
    let preprocessorCalls = 0;
    let embedCalls = 0;
    const merger = new ToolMerger({
      embedder: { embed: async () => (++embedCalls, [[1]]) },
      classifier: classifier(),
      maxDescriptorChars: 256,
      text: {
        truncate: 256,
        preprocessors: [() => (++preprocessorCalls, "x".repeat(257))],
      },
    });

    await expect(merger.merge([raw("a")])).rejects.toThrow("maxDescriptorChars");
    expect(preprocessorCalls).toBe(1);
    expect(embedCalls).toBe(0);
  });

  test("rejects an oversized synthesized description", async () => {
    let synthesizerCalls = 0;
    const merger = new ToolMerger({
      embedder: embedder([[1]]),
      classifier: classifier(),
      synthesizer: {
        synthesize: async () => (++synthesizerCalls, { description: "x".repeat(257) }),
      },
      maxDescriptorChars: 256,
    });

    await expect(merger.merge([raw("a")])).rejects.toThrow("maxDescriptorChars");
    expect(synthesizerCalls).toBe(1);
  });

  test("rejects excessive embedding dimensions before classifier calls", async () => {
    let classifierCalls = 0;
    const merger = new ToolMerger({
      embedder: embedder([
        [1, 0],
        [1, 0],
      ]),
      classifier: { classify: async () => (++classifierCalls, { similar: true }) },
      maxEmbeddingDimensions: 1,
      similarityThreshold: 0,
    });

    await expect(merger.merge([raw("a"), raw("b")])).rejects.toThrow("maxEmbeddingDimensions");
    expect(classifierCalls).toBe(0);
  });

  test("fails closed on classifier-call budget before invoking the classifier", async () => {
    let calls = 0;
    const merger = new ToolMerger({
      embedder: embedder([
        [1, 0],
        [1, 0],
        [1, 0],
      ]),
      classifier: { classify: async () => (++calls, { similar: true }) },
      similarityThreshold: 0,
      maxClassifierCalls: 2,
    });

    await expect(merger.merge([raw("a"), raw("b"), raw("c")])).rejects.toThrow(
      "maxClassifierCalls",
    );
    expect(calls).toBe(0);
  });

  test("bounds classifier concurrency and reports failures in candidate order", async () => {
    let active = 0;
    let peak = 0;
    const merger = new ToolMerger({
      embedder: embedder(Array.from({ length: 4 }, () => [1, 0])),
      classifier: {
        async classify(left, right) {
          active += 1;
          peak = Math.max(peak, active);
          const pair = `${left.name}:${right.name}`;
          await Bun.sleep(pair === "a:b" ? 20 : 1);
          active -= 1;
          if (pair === "a:b" || pair === "a:c") throw new Error(pair);
          return { similar: false };
        },
      },
      similarityThreshold: 0,
      modelConcurrency: 2,
    });

    await expect(merger.merge([raw("a"), raw("b"), raw("c"), raw("d")])).rejects.toThrow("a:b");
    expect(peak).toBe(2);
  });

  test("uses pass barriers with bounded ordered validator concurrency", async () => {
    let active = 0;
    let peak = 0;
    const completedPassOne: string[] = [];
    const calls: string[] = [];
    const merger = new ToolMerger({
      embedder: embedder(
        Array.from({ length: 4 }, (_value, index) => [index < 2 ? 1 : 0, index < 2 ? 0 : 1]),
      ),
      classifier: classifier((left, right) => left.name[0] === right.name[0]),
      similarityThreshold: 0.5,
      autoCorrectionPasses: 2,
      modelConcurrency: 2,
      validator: {
        async validate(cluster) {
          const key = cluster.map((item) => item.name).join("");
          const pass = calls.filter((value) => value === key).length + 1;
          calls.push(key);
          if (pass === 2 && completedPassOne.length !== 2) throw new Error("missing pass barrier");
          active += 1;
          peak = Math.max(peak, active);
          await Bun.sleep(key === "ab" ? 10 : 1);
          active -= 1;
          if (pass === 1) completedPassOne.push(key);
          return { merge: true };
        },
      },
    });

    await merger.merge([raw("a1"), raw("a2"), raw("b1"), raw("b2")]);
    expect(calls).toEqual(["a1a2", "b1b2", "a1a2", "b1b2"]);
    expect(peak).toBe(2);
  });

  test("bounds synthesizer concurrency while preserving component order", async () => {
    let active = 0;
    let peak = 0;
    const merger = new ToolMerger({
      embedder: embedder(Array.from({ length: 4 }, (_value, index) => [index + 1, 1])),
      classifier: classifier(() => false),
      similarityThreshold: -1,
      modelConcurrency: 2,
      synthesizer: {
        async synthesize(representative) {
          active += 1;
          peak = Math.max(peak, active);
          await Bun.sleep(representative.name === "a" ? 15 : 1);
          active -= 1;
          return { description: `made-${representative.name}` };
        },
      },
    });

    const result = await merger.merge([raw("a"), raw("b"), raw("c"), raw("d")]);
    expect(peak).toBe(2);
    expect(result.merged.map((item) => item.description)).toEqual([
      "made-a",
      "made-b",
      "made-c",
      "made-d",
    ]);
  });

  test("bounded top-k candidate generation preserves exhaustive tie semantics", async () => {
    const result = await new ToolMerger({
      embedder: embedder(Array.from({ length: 5 }, () => [1, 0])),
      classifier: classifier(() => false),
      similarityThreshold: 0,
      candidateCount: 1,
    }).merge([raw("a"), raw("b"), raw("c"), raw("d"), raw("e")]);

    const names = result.candidatePairs.map(({ ids }) =>
      ids.map((id) => result.originals.find((item) => item.id === id)!.name).join(":"),
    );
    expect(names).toEqual(["a:b", "a:c", "a:d", "a:e"]);
  });
  test("generates deterministic top-neighbor candidate pairs above the threshold", async () => {
    const calls: string[] = [];
    const tools = [raw("alpha"), raw("beta"), raw("gamma")];
    const merger = new ToolMerger({
      embedder: embedder([
        [1, 0],
        [0.9, 0.1],
        [0, 1],
      ]),
      classifier: {
        async classify(left, right) {
          calls.push([left.name, right.name].sort().join(":"));
          return { similar: false };
        },
      },
      candidateCount: 1,
      similarityThreshold: 0.8,
    });

    const result = await merger.merge(tools);

    expect(result.candidatePairs).toHaveLength(1);
    expect(
      result.candidatePairs[0]?.ids.map((id) => result.originals.find((t) => t.id === id)?.name),
    ).toEqual(["alpha", "beta"]);
    expect(calls).toEqual(["alpha:beta"]);
  });

  test("exports paper-faithful defaults and uses the 0.82 candidate threshold", async () => {
    expect(PAPER_2026_DEFAULTS).toEqual({
      candidateNeighbors: 30,
      candidateThreshold: 0.82,
      autoCorrectionPasses: 1,
    });
    const result = await new ToolMerger({
      embedder: embedder([
        [1, 0],
        [0.8, 0.6],
        [0.9, Math.sqrt(0.19)],
      ]),
      classifier: classifier(() => false),
    }).merge([raw("target"), raw("below"), raw("above")]);

    const names = result.candidatePairs.map(({ ids }) =>
      ids.map((id) => result.originals.find((tool) => tool.id === id)?.name),
    );
    expect(names).toContainEqual(["target", "above"]);
    expect(names).not.toContainEqual(["target", "below"]);
  });

  test("prohibits cross-namespace candidate pairs by default", async () => {
    const left = { ...raw("left"), namespace: "one" };
    const right = { ...raw("right"), namespace: "two" };
    let calls = 0;
    const result = await new ToolMerger({
      embedder: embedder([
        [1, 0],
        [1, 0],
      ]),
      classifier: {
        async classify() {
          calls += 1;
          return { similar: true };
        },
      },
      similarityThreshold: 0,
    }).merge([left, right]);

    expect(result.candidatePairs).toEqual([]);
    expect(calls).toBe(0);
    expect(result.components).toHaveLength(2);
  });

  test("deduplicates unordered candidates and calls the classifier once per pair", async () => {
    let calls = 0;
    const merger = new ToolMerger({
      embedder: embedder([
        [1, 0],
        [1, 0],
        [1, 0],
      ]),
      classifier: {
        async classify() {
          calls += 1;
          return { similar: false };
        },
      },
      candidateCount: 30,
      similarityThreshold: 0,
    });

    const result = await merger.merge([raw("a"), raw("b"), raw("c")]);

    expect(result.candidatePairs).toHaveLength(3);
    expect(calls).toBe(3);
  });

  test("forms transitive connected components from classifier edges", async () => {
    const merger = new ToolMerger({
      embedder: embedder([
        [1, 0],
        [1, 0],
        [1, 0],
      ]),
      classifier: classifier((a, b) => new Set([a.name, b.name]).has("bridge")),
      similarityThreshold: 0,
    });

    const result = await merger.merge([raw("left"), raw("bridge"), raw("right")]);

    expect(result.components).toHaveLength(1);
    expect(result.components[0]?.map((tool) => tool.name).sort()).toEqual([
      "bridge",
      "left",
      "right",
    ]);
  });

  test("chooses shortest function name then lexical name and id", async () => {
    const merger = new ToolMerger({
      embedder: embedder([
        [1, 0],
        [1, 0],
        [1, 0],
      ]),
      classifier: classifier(),
      similarityThreshold: 0,
    });

    const result = await merger.merge([raw("zz"), raw("aa"), raw("lengthy")]);

    expect(result.merged[0]?.name).toBe("aa");
    expect(result.merged[0]?.representativeId).toBe(
      result.originals.find((tool) => tool.name === "aa")?.id,
    );
  });

  test("manifest resolves a merged id to the exact original object identities", async () => {
    const first = raw("first");
    const second = raw("second");
    const result = await new ToolMerger({
      embedder: embedder([
        [1, 0],
        [1, 0],
      ]),
      classifier: classifier(),
      similarityThreshold: 0,
    }).merge([first, second]);

    const resolved = result.manifest.resolve(result.merged[0]!.id);

    expect(resolved[0]).toBe(first);
    expect(resolved[1]).toBe(second);
    expect(() => result.manifest.resolve("unknown")).toThrow("Unknown merged descriptor id");
  });

  test("returns an empty deterministic result for an empty catalog without model calls", async () => {
    let classifierCalls = 0;
    const result = await new ToolMerger({
      embedder: embedder([]),
      classifier: {
        async classify() {
          classifierCalls += 1;
          return { similar: true };
        },
      },
    }).merge([]);

    expect(result.originals).toEqual([]);
    expect(result.candidatePairs).toEqual([]);
    expect(result.components).toEqual([]);
    expect(result.merged).toEqual([]);
    expect(classifierCalls).toBe(0);
  });

  test.each([
    { label: "wrong vector count", vectors: [[1, 0]] },
    {
      label: "inconsistent dimensions",
      vectors: [[1, 0], [1]],
    },
    {
      label: "empty vectors",
      vectors: [[], []],
    },
    {
      label: "non-finite vectors",
      vectors: [
        [1, Number.NaN],
        [1, 0],
      ],
    },
    {
      label: "zero-norm vectors",
      vectors: [
        [0, 0],
        [1, 0],
      ],
    },
  ])("rejects malformed embeddings: $label", async ({ vectors }) => {
    const merger = new ToolMerger({ embedder: embedder(vectors), classifier: classifier() });
    await expect(merger.merge([raw("a"), raw("b")])).rejects.toThrow(/embedding/i);
  });

  test("rejects invalid classifier output", async () => {
    const merger = new ToolMerger({
      embedder: embedder([
        [1, 0],
        [1, 0],
      ]),
      classifier: { classify: async () => ({ similar: "yes" }) as never },
      similarityThreshold: 0,
    });

    await expect(merger.merge([raw("a"), raw("b")])).rejects.toThrow(/classifier/i);
  });

  test("keeps a validator-approved component intact", async () => {
    const validator: ClusterValidator = { validate: async () => ({ merge: true }) };
    const result = await new ToolMerger({
      embedder: embedder([
        [1, 0],
        [1, 0],
      ]),
      classifier: classifier(),
      validator,
      similarityThreshold: 0,
    }).merge([raw("a"), raw("b")]);

    expect(result.components).toHaveLength(1);
  });

  test("accepts a strict validator split partition", async () => {
    const validator: ClusterValidator = {
      async validate(cluster) {
        return { merge: false, clusters: [[cluster[0]!.id], [cluster[1]!.id, cluster[2]!.id]] };
      },
    };
    const result = await new ToolMerger({
      embedder: embedder([
        [1, 0],
        [1, 0],
        [1, 0],
      ]),
      classifier: classifier(),
      validator,
      similarityThreshold: 0,
    }).merge([raw("a"), raw("b"), raw("c")]);

    expect(result.components.map((cluster) => cluster.map((tool) => tool.name))).toEqual([
      ["a"],
      ["b", "c"],
    ]);
  });

  test("turns validator omissions into singleton components", async () => {
    const validator: ClusterValidator = {
      async validate(cluster) {
        return { merge: false, clusters: [[cluster[0]!.id, cluster[1]!.id]] };
      },
    };
    const result = await new ToolMerger({
      embedder: embedder([
        [1, 0],
        [1, 0],
        [1, 0],
      ]),
      classifier: classifier(),
      validator,
      similarityThreshold: 0,
    }).merge([raw("a"), raw("b"), raw("c")]);

    expect(result.components.map((cluster) => cluster.map((tool) => tool.name))).toEqual([
      ["a", "b"],
      ["c"],
    ]);
  });

  test.each([
    {
      label: "unknown ids",
      clusters: (ids: string[]) => [[ids[0]!, "unknown"]],
    },
    {
      label: "duplicate ids",
      clusters: (ids: string[]) => [[ids[0]!, ids[1]!], [ids[1]!]],
    },
    { label: "empty groups", clusters: () => [[]] },
  ])("rejects correction partitions containing $label", async ({ clusters }) => {
    const validator: ClusterValidator = {
      async validate(cluster) {
        return { merge: false, clusters: clusters(cluster.map((tool) => tool.id)) };
      },
    };
    const merger = new ToolMerger({
      embedder: embedder([
        [1, 0],
        [1, 0],
      ]),
      classifier: classifier(),
      validator,
      similarityThreshold: 0,
    });

    await expect(merger.merge([raw("a"), raw("b")])).rejects.toThrow(/correction/i);
  });

  test("bounds auto-correction passes", async () => {
    let calls = 0;
    const validator: ClusterValidator = {
      async validate(cluster) {
        calls += 1;
        return cluster.length > 1
          ? { merge: false, clusters: [[cluster[0]!.id], cluster.slice(1).map((tool) => tool.id)] }
          : { merge: true };
      },
    };
    await new ToolMerger({
      embedder: embedder([
        [1, 0],
        [1, 0],
        [1, 0],
      ]),
      classifier: classifier(),
      validator,
      similarityThreshold: 0,
      autoCorrectionPasses: 1,
    }).merge([raw("a"), raw("b"), raw("c")]);

    expect(calls).toBe(1);
  });

  test("validates constructor bounds", () => {
    expect(
      () =>
        new ToolMerger({ embedder: embedder([]), classifier: classifier(), candidateCount: -1 }),
    ).toThrow(/candidateCount/);
    expect(
      () =>
        new ToolMerger({
          embedder: embedder([]),
          classifier: classifier(),
          autoCorrectionPasses: 1.5,
        }),
    ).toThrow(/autoCorrectionPasses/);
    expect(
      () =>
        new ToolMerger({
          embedder: embedder([]),
          classifier: classifier(),
          similarityThreshold: 2,
        }),
    ).toThrow(/similarityThreshold/);
  });

  test("consolidates unique object properties and only keeps universally required parameters", async () => {
    const first = {
      ...raw("alpha"),
      inputSchema: {
        type: "object",
        properties: { shared: { type: "string" }, alphaOnly: { type: "number" } },
        required: ["shared", "alphaOnly"],
      },
    };
    const second = {
      ...raw("beta"),
      inputSchema: {
        type: "object",
        properties: { shared: { type: "string" }, betaOnly: { type: "boolean" } },
        required: ["shared", "betaOnly"],
      },
    };
    const result = await new ToolMerger({
      embedder: embedder([
        [1, 0],
        [1, 0],
      ]),
      classifier: classifier(),
      similarityThreshold: 0,
    }).merge([first, second]);

    expect(result.merged[0]?.inputSchema).toEqual({
      type: "object",
      properties: {
        alphaOnly: { type: "number" },
        betaOnly: { type: "boolean" },
        shared: { type: "string" },
      },
      required: ["shared"],
      anyOf: [first.inputSchema, second.inputSchema],
    });
  });

  test("uses deterministic deduplicated anyOf for conflicting property schemas", async () => {
    const tools = [
      { ...raw("a"), inputSchema: { type: "object", properties: { value: { type: "string" } } } },
      { ...raw("b"), inputSchema: { type: "object", properties: { value: { type: "number" } } } },
      { ...raw("c"), inputSchema: { type: "object", properties: { value: { type: "string" } } } },
    ];
    const result = await new ToolMerger({
      embedder: embedder([
        [1, 0],
        [1, 0],
        [1, 0],
      ]),
      classifier: classifier(),
      similarityThreshold: 0,
    }).merge(tools);

    expect(result.merged[0]?.inputSchema).toEqual({
      type: "object",
      properties: { value: { anyOf: [{ type: "number" }, { type: "string" }] } },
      required: [],
      anyOf: [tools[1]!.inputSchema, tools[0]!.inputSchema],
    });
  });

  test("preserves a singleton's complete JSON Schema", async () => {
    const inputSchema = {
      $schema: "https://json-schema.org/draft/2020-12/schema",
      $id: "https://example.test/schemas/search.json",
      title: "Search input",
      description: "A fully constrained search request",
      type: "object",
      properties: {
        query: { $ref: "#/$defs/nonEmptyString", description: "Search terms" },
      },
      required: ["query"],
      additionalProperties: false,
      minProperties: 1,
      allOf: [{ propertyNames: { pattern: "^[a-z]+$" } }],
      $defs: {
        nonEmptyString: { type: "string", minLength: 1 },
      },
      definitions: {
        legacyString: { type: "string" },
      },
      examples: [{ query: "papers" }],
      deprecated: false,
    };
    const result = await new ToolMerger({
      embedder: embedder([[1, 0]]),
      classifier: classifier(),
    }).merge([{ ...raw("search"), inputSchema }]);

    expect(result.merged[0]?.inputSchema).toEqual(inputSchema);
  });

  test("preserves complete member schemas in deterministic deduplicated anyOf branches", async () => {
    const stringSchema = {
      type: "object",
      title: "String lookup",
      properties: { shared: { type: "string" }, query: { type: "string", minLength: 2 } },
      required: ["shared", "query"],
      additionalProperties: false,
    };
    const numericSchema = {
      type: "object",
      title: "Numeric lookup",
      properties: { shared: { type: "string" }, limit: { type: "integer", minimum: 1 } },
      required: ["shared", "limit"],
      maxProperties: 2,
    };
    const tools = [
      { ...raw("string-copy"), inputSchema: stringSchema },
      { ...raw("numeric"), inputSchema: numericSchema },
      { ...raw("string"), inputSchema: stringSchema },
    ];
    const result = await new ToolMerger({
      embedder: embedder([
        [1, 0],
        [1, 0],
        [1, 0],
      ]),
      classifier: classifier(),
      similarityThreshold: 0,
    }).merge(tools);

    expect(result.merged[0]?.inputSchema).toEqual({
      type: "object",
      properties: {
        limit: { type: "integer", minimum: 1 },
        query: { type: "string", minLength: 2 },
        shared: { type: "string" },
      },
      required: ["shared"],
      anyOf: [numericSchema, stringSchema],
    });
  });

  test("hoists compatible local definition targets so member references remain resolvable", async () => {
    const sharedDefinition = { type: "string", minLength: 1 };
    const tools = [
      {
        ...raw("modern"),
        inputSchema: {
          type: "object",
          properties: { modern: { $ref: "#/$defs/shared" } },
          $defs: { shared: sharedDefinition },
        },
      },
      {
        ...raw("legacy"),
        inputSchema: {
          type: "object",
          properties: { legacy: { $ref: "#/definitions/shared" } },
          definitions: { shared: sharedDefinition },
        },
      },
      {
        ...raw("modern-copy"),
        inputSchema: {
          type: "object",
          properties: { other: { $ref: "#/$defs/shared" } },
          $defs: { shared: { minLength: 1, type: "string" } },
        },
      },
    ];
    const result = await new ToolMerger({
      embedder: embedder([
        [1, 0],
        [1, 0],
        [1, 0],
      ]),
      classifier: classifier(),
      similarityThreshold: 0,
    }).merge(tools);

    expect(result.merged[0]?.inputSchema).toMatchObject({
      $defs: { shared: sharedDefinition },
      definitions: { shared: sharedDefinition },
    });
  });

  test.each(["$defs", "definitions"] as const)(
    "fails closed when member schemas have conflicting same-name %s",
    async (keyword) => {
      const merger = new ToolMerger({
        embedder: embedder([
          [1, 0],
          [1, 0],
        ]),
        classifier: classifier(),
        similarityThreshold: 0,
      });
      const tools = [
        {
          ...raw("string"),
          inputSchema: { type: "object", [keyword]: { shared: { type: "string" } } },
        },
        {
          ...raw("number"),
          inputSchema: { type: "object", [keyword]: { shared: { type: "number" } } },
        },
      ];

      await expect(merger.merge(tools)).rejects.toThrow(
        `Conflicting ${keyword} definition: shared`,
      );
    },
  );

  test("preserves legitimate schema properties named code, execute, tags, and namespace", async () => {
    const inputSchema = {
      type: "object",
      properties: {
        code: { type: "string" },
        execute: { type: "boolean" },
        tags: { type: "array", items: { type: "string" } },
        namespace: { type: "string" },
      },
      required: ["code", "execute", "namespace", "tags"],
    };
    const result = await new ToolMerger({
      embedder: embedder([[1, 0]]),
      classifier: classifier(),
    }).merge([{ ...raw("safe"), inputSchema }]);

    expect(result.merged[0]?.inputSchema).toEqual(inputSchema);
  });

  test("preserves a __proto__ schema property without mutating the properties prototype", async () => {
    const protoDefinition = { type: "string" };
    const schemaProperties = Object.create(null) as Record<string, unknown>;
    schemaProperties.__proto__ = protoDefinition;
    const result = await new ToolMerger({
      embedder: embedder([[1, 0]]),
      classifier: classifier(),
    }).merge([
      {
        ...raw("safe"),
        inputSchema: { type: "object", properties: schemaProperties, required: ["__proto__"] },
      },
    ]);

    const properties = result.merged[0]?.inputSchema.properties as Record<string, unknown>;
    expect(Object.hasOwn(properties, "__proto__")).toBe(true);
    expect(properties.__proto__).toEqual(protoDefinition);
    expect(Object.getPrototypeOf(properties)).toBe(Object.prototype);
  });

  test.each([
    { label: "Date values", value: () => new Date(0) },
    { label: "Map values", value: () => new Map([["key", "value"]]) },
    {
      label: "class instances",
      value: () =>
        new (class UnsafeValue {
          value = 1;
        })(),
    },
    { label: "non-plain prototypes", value: () => Object.create({}) as object },
    {
      label: "accessors",
      value: () => {
        const value = {};
        Object.defineProperty(value, "unsafe", { enumerable: true, get: () => "value" });
        return value;
      },
    },
    {
      label: "inherited enumerable state",
      value: () => Object.create({ inherited: "value" }) as object,
    },
    {
      label: "symbol-keyed values",
      value: () => ({ [Symbol("unsafe")]: "value" }),
    },
    {
      label: "sparse arrays",
      value: () => {
        const value: unknown[] = [];
        value.length = 1;
        return value;
      },
    },
    {
      label: "cycles",
      value: () => {
        const value: Record<string, unknown> = {};
        value.self = value;
        return value;
      },
    },
    { label: "NaN", value: () => Number.NaN },
    { label: "positive infinity", value: () => Number.POSITIVE_INFINITY },
    { label: "negative infinity", value: () => Number.NEGATIVE_INFINITY },
  ])("rejects non-strict JSON member schema data: $label", async ({ value }) => {
    const merger = new ToolMerger({ embedder: embedder([[1, 0]]), classifier: classifier() });

    await expect(
      merger.merge([
        {
          ...raw("unsafe"),
          inputSchema: { type: "object", properties: { value: { default: value() } } },
        },
      ]),
    ).rejects.toThrow(/strict JSON data/);
  });

  test("rejects schema accessors before fingerprinting without invoking them", async () => {
    let getterCalls = 0;
    const unsafe = {};
    Object.defineProperty(unsafe, "value", {
      enumerable: true,
      get() {
        getterCalls += 1;
        return "unsafe";
      },
    });
    const merger = new ToolMerger({ embedder: embedder([[1, 0]]), classifier: classifier() });

    await expect(
      merger.merge([
        {
          ...raw("unsafe-accessor"),
          inputSchema: { type: "object", properties: { value: { default: unsafe } } },
        },
      ]),
    ).rejects.toThrow(/strict JSON data/);
    expect(getterCalls).toBe(0);
  });

  test("deep-detaches a singleton schema from its source", async () => {
    const inputSchema = {
      type: "object",
      properties: { query: { type: "string", examples: ["original"] } },
      required: ["query"],
      $defs: { shared: { type: "string", minLength: 1 } },
    };
    const result = await new ToolMerger({
      embedder: embedder([[1, 0]]),
      classifier: classifier(),
    }).merge([{ ...raw("singleton"), inputSchema }]);
    const mergedSchema = result.merged[0]!.inputSchema;

    inputSchema.properties.query.examples[0] = "mutated";
    inputSchema.$defs.shared.minLength = 99;

    expect(mergedSchema).toEqual({
      type: "object",
      properties: { query: { type: "string", examples: ["original"] } },
      required: ["query"],
      $defs: { shared: { type: "string", minLength: 1 } },
    });
    expect(mergedSchema).not.toBe(inputSchema);
  });

  test("deep-detaches flattened and complete multi-member schemas from their sources", async () => {
    const firstSchema = {
      type: "object",
      properties: { shared: { type: "string", examples: ["first"] } },
      required: ["shared"],
    };
    const secondSchema = {
      type: "object",
      properties: { shared: { type: "string", examples: ["second"] } },
      required: ["shared"],
    };
    const result = await new ToolMerger({
      embedder: embedder([
        [1, 0],
        [1, 0],
      ]),
      classifier: classifier(),
      similarityThreshold: 0,
    }).merge([
      { ...raw("first"), inputSchema: firstSchema },
      { ...raw("second"), inputSchema: secondSchema },
    ]);
    const mergedSchema = result.merged[0]!.inputSchema;

    firstSchema.properties.shared.examples[0] = "mutated-first";
    secondSchema.properties.shared.examples[0] = "mutated-second";

    expect(mergedSchema).toEqual({
      type: "object",
      properties: {
        shared: {
          anyOf: [
            { type: "string", examples: ["first"] },
            { type: "string", examples: ["second"] },
          ],
        },
      },
      required: ["shared"],
      anyOf: [firstSchema, secondSchema].map((schema, index) => ({
        ...schema,
        properties: {
          shared: { type: "string", examples: [index === 0 ? "first" : "second"] },
        },
      })),
    });
  });

  test("accepts exactly a synthesized description while keeping schema exclusively trusted", async () => {
    const first = {
      ...raw("safe"),
      inputSchema: {
        type: "object",
        properties: { requiredInput: { type: "string" } },
        required: ["requiredInput"],
      },
    };
    const second = {
      ...raw("safer"),
      inputSchema: {
        type: "object",
        properties: { optionalInput: { type: "number" } },
      },
    };
    const result = await new ToolMerger({
      embedder: embedder([
        [1, 0],
        [1, 0],
      ]),
      classifier: classifier(),
      similarityThreshold: 0,
      synthesizer: {
        synthesize: async () => ({ description: "Synthesized description" }),
      },
    }).merge([first, second]);

    expect(result.merged[0]?.description).toBe("Synthesized description");
    expect(result.merged[0]?.inputSchema).toEqual({
      type: "object",
      properties: {
        optionalInput: { type: "number" },
        requiredInput: { type: "string" },
      },
      required: [],
      anyOf: [second.inputSchema, first.inputSchema],
    });
  });

  test.each([
    { field: "inputSchema", extra: { inputSchema: { type: "object" } } },
    { field: "tags", extra: { tags: ["allow"] } },
    { field: "namespace", extra: { namespace: "untrusted" } },
    { field: "other top-level keys", extra: { policy: "allow" } },
  ])("rejects synthesized $field", async ({ extra }) => {
    const merger = new ToolMerger({
      embedder: embedder([[1, 0]]),
      classifier: classifier(),
      synthesizer: {
        synthesize: async () =>
          ({
            description: "Advisory description",
            ...extra,
          }) as never,
      },
    });

    await expect(
      merger.merge([{ ...raw("safe"), tags: ["deny"], namespace: "trusted" }]),
    ).rejects.toThrow(/descriptor/i);
  });

  test("uses the sorted union of every cluster member's trusted tags", async () => {
    const result = await new ToolMerger({
      embedder: embedder([
        [1, 0],
        [1, 0],
        [1, 0],
      ]),
      classifier: classifier(),
      similarityThreshold: 0,
    }).merge([
      { ...raw("a"), tags: ["public", "zeta"] },
      { ...raw("b"), tags: ["deny", "public"] },
      { ...raw("c"), tags: ["security"] },
    ]);

    expect(result.merged[0]?.tags).toEqual(["deny", "public", "security", "zeta"]);
  });

  test("preserves a namespace shared by every cluster member", async () => {
    const result = await new ToolMerger({
      embedder: embedder([
        [1, 0],
        [1, 0],
      ]),
      classifier: classifier(),
      similarityThreshold: 0,
    }).merge([
      { ...raw("a"), namespace: "trusted" },
      { ...raw("b"), namespace: "trusted" },
    ]);

    expect(result.merged[0]?.namespace).toBe("trusted");
  });

  test("omits namespace when explicitly merging members from different namespaces", async () => {
    const result = await new ToolMerger({
      embedder: embedder([
        [1, 0],
        [1, 0],
      ]),
      classifier: classifier(),
      allowCrossNamespaceCandidates: true,
      similarityThreshold: 0,
    }).merge([
      { ...raw("a"), namespace: "one" },
      { ...raw("b"), namespace: "two" },
    ]);

    expect(result.merged[0]).not.toHaveProperty("namespace");
  });

  test("rejects synthesized executable code", async () => {
    const synth: DescriptorSynthesizer = {
      async synthesize() {
        return {
          description: "Unsafe descriptor",
          tags: ["unsafe"],
          implementation: "return dangerous()",
        } as never;
      },
    };
    const merger = new ToolMerger({
      embedder: embedder([[1, 0]]),
      classifier: classifier(),
      synthesizer: synth,
    });

    await expect(merger.merge([raw("unsafe")])).rejects.toThrow(/descriptor/i);
  });

  test.each([{}, { description: "" }, { description: "   " }, { description: 1 }])(
    "rejects malformed synthesized descriptors",
    async (output) => {
      const merger = new ToolMerger({
        embedder: embedder([[1, 0]]),
        classifier: classifier(),
        synthesizer: { synthesize: async () => output as never },
      });
      await expect(merger.merge([raw("a")])).rejects.toThrow(/descriptor/i);
    },
  );

  test("rejects input canonical id collisions", async () => {
    const merger = new ToolMerger({
      embedder: embedder([
        [1, 0],
        [1, 0],
      ]),
      classifier: classifier(),
    });
    await expect(merger.merge([raw("same"), raw("same")])).rejects.toThrow(/collision/i);
  });

  test("MergeManifest rejects merged id collisions", () => {
    const one = raw("one");
    expect(
      () =>
        new MergeManifest([
          { mergedId: "same", originals: [one] },
          { mergedId: "same", originals: [raw("two")] },
        ]),
    ).toThrow(/collision/i);
  });
});

describe("paper prompt builders", () => {
  test("produce provider-neutral structured prompts without chain-of-thought or implementation code", () => {
    const canonical = {
      id: "id",
      name: "search",
      description: "Search records",
      inputSchema: { type: "object" },
      tags: [],
      original: raw("search"),
    } satisfies CanonicalTool<RawTool>;
    const prompts = [
      buildRelationshipPrompt(canonical, canonical),
      buildCorrectionPrompt([canonical]),
      buildDescriptorSynthesisPrompt(canonical, [canonical]),
      buildQueryDecompositionPrompt("find and summarize"),
    ];

    for (const prompt of prompts) {
      expect(prompt).toContain("JSON");
      expect(prompt).toMatch(/concise/i);
      expect(prompt).toMatch(/do not (?:include|produce|write).*implementation code/i);
      expect(prompt).not.toMatch(/chain[- ]of[- ]thought|show your work|step by step/i);
      expect(prompt).not.toMatch(/OpenAI|Anthropic|Gemini/);
    }
  });

  test("asks the synthesizer only for JSON description data", () => {
    const canonical = {
      id: "id",
      name: "search",
      description: "Search records",
      inputSchema: { type: "object" },
      tags: ["security"],
      namespace: "trusted",
      original: raw("search"),
    } satisfies CanonicalTool<RawTool>;

    const prompt = buildDescriptorSynthesisPrompt(canonical, [canonical]);

    expect(prompt.match(/JSON shape:.*$/m)?.[0]).toBe(
      'JSON shape: {"description": string}. Keep text concise.',
    );
  });
});
