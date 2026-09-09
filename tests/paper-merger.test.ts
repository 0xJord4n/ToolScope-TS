import { describe, expect, test } from "bun:test";
import {
  MergeManifest,
  PAPER_2026_DEFAULTS,
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
    });
  });

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

  test("treats a synthesized schema as advisory and does not let it remove original parameters", async () => {
    const tool = {
      ...raw("safe"),
      inputSchema: {
        type: "object",
        properties: { requiredInput: { type: "string" } },
        required: ["requiredInput"],
      },
    };
    const result = await new ToolMerger({
      embedder: embedder([[1, 0]]),
      classifier: classifier(),
      synthesizer: {
        synthesize: async () => ({
          description: "Advisory descriptor",
          inputSchema: { type: "object", properties: {} },
        }),
      },
    }).merge([tool]);

    expect(result.merged[0]?.inputSchema).toEqual(tool.inputSchema);
  });

  test.each([
    { field: "tags", extra: { tags: ["allow"] } },
    { field: "namespace", extra: { namespace: "untrusted" } },
    { field: "arbitrary metadata", extra: { policy: "allow" } },
  ])("rejects synthesized $field instead of accepting policy metadata", async ({ extra }) => {
    const merger = new ToolMerger({
      embedder: embedder([[1, 0]]),
      classifier: classifier(),
      synthesizer: {
        synthesize: async (representative) =>
          ({
            description: "Advisory description",
            inputSchema: representative.inputSchema,
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

  test("rejects synthesized executable values nested in descriptor data", async () => {
    const merger = new ToolMerger({
      embedder: embedder([[1, 0]]),
      classifier: classifier(),
      synthesizer: {
        synthesize: async () => ({
          description: "unsafe",
          inputSchema: { type: "object", properties: { value: { default: () => "code" } } },
        }),
      },
    });

    await expect(merger.merge([raw("unsafe")])).rejects.toThrow(/descriptor/i);
  });

  test("rejects unsupported non-object synthesized schemas", async () => {
    const merger = new ToolMerger({
      embedder: embedder([[1, 0]]),
      classifier: classifier(),
      synthesizer: {
        synthesize: async () => ({ description: "invalid", inputSchema: { type: "string" } }),
      },
    });

    await expect(merger.merge([raw("invalid")])).rejects.toThrow(/descriptor/i);
  });

  test("rejects synthesized executable code", async () => {
    const synth: DescriptorSynthesizer = {
      async synthesize(representative) {
        return {
          description: "Unsafe descriptor",
          inputSchema: representative.inputSchema,
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

  test.each([
    { description: "", inputSchema: {} },
    { description: "valid", inputSchema: null },
    { description: "valid", inputSchema: {}, tags: ["ok", 1] },
    { description: "valid", inputSchema: {}, namespace: "" },
  ])("rejects malformed synthesized descriptors", async (output) => {
    const merger = new ToolMerger({
      embedder: embedder([[1, 0]]),
      classifier: classifier(),
      synthesizer: { synthesize: async () => output as never },
    });
    await expect(merger.merge([raw("a")])).rejects.toThrow(/descriptor/i);
  });

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

  test("does not ask the synthesizer to produce trusted policy metadata", () => {
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

    expect(prompt.match(/JSON shape:.*$/m)?.[0]).not.toMatch(/tags|namespace/);
  });
});
