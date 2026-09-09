import { describe, expect, test } from "bun:test";
import { ToolIndex } from "../src/tool-index";
import type { CanonicalTool, IndexedTool, Reranker, VectorBackend } from "../src/types";
import {
  bm25Scores,
  MultiQueryRetriever,
  type MultiQueryCandidateTrace,
  type QueryDecomposer,
} from "../src/paper/retriever";

type TestTool = {
  name: string;
  description: string;
  inputSchema: Record<string, unknown>;
  tags?: string[];
  namespace?: string;
};

const tool = (name: string, options: Pick<TestTool, "tags" | "namespace"> = {}): TestTool => ({
  name,
  description: `${name} description`,
  inputSchema: {},
  ...options,
});

const constantEmbedder = {
  async embed(texts: string[]) {
    return texts.map(() => [1]);
  },
};

async function fixture(
  originals: TestTool[],
  steps: string[],
  scores: Record<string, Record<string, number>>,
  options: {
    k?: number;
    rerankPoolSize?: number;
    decomposer?: QueryDecomposer;
    alpha?: number;
  } = {},
) {
  const index = new ToolIndex({ embedder: constantEmbedder });
  await index.add(originals);
  const rerankCalls: Array<{ query: string; names: string[] }> = [];
  const reranker: Reranker = {
    async rerank(query, candidates) {
      rerankCalls.push({ query, names: candidates.map((candidate) => candidate.name) });
      return candidates.map((candidate) => scores[query]?.[candidate.name] ?? 0);
    },
  };
  const decomposer =
    options.decomposer ??
    ({
      async decompose() {
        return steps;
      },
    } satisfies QueryDecomposer);
  const retriever = new MultiQueryRetriever({
    index,
    decomposer,
    reranker,
    k: options.k ?? 3,
    rerankPoolSize: options.rerankPoolSize,
    alpha: options.alpha,
  });
  return { index, rerankCalls, retriever };
}

function vectorFixture(
  queryVector: number[],
  storedVector?: number[],
): MultiQueryRetriever<TestTool> {
  const original = tool("candidate");
  const records: IndexedTool[] = storedVector
    ? [
        {
          tool: {
            id: "candidate-id",
            name: original.name,
            description: original.description,
            inputSchema: {},
            tags: [],
            original,
          },
          vector: storedVector,
          text: original.description,
        },
      ]
    : [];
  const backend: VectorBackend = {
    size: records.length,
    upsert() {},
    remove() {},
    clear() {},
    list: () => records,
  };
  const index = new ToolIndex({ backend, embedder: { embed: async () => [queryVector] } });
  return new MultiQueryRetriever({
    index,
    decomposer: { decompose: () => ["step"] },
    reranker: { rerank: async (_query, candidates) => candidates.map(() => 1) },
    k: 1,
  });
}

describe("paper-style multi-query retrieval", () => {
  test("decomposes the original query and retrieves steps in order", async () => {
    const originals = [tool("alpha"), tool("beta")];
    const decomposed: string[] = [];
    const { rerankCalls, retriever } = await fixture(
      originals,
      ["first step", "second step"],
      {
        "first step": { alpha: 2, beta: 1 },
        "second step": { alpha: 1, beta: 2 },
      },
      {
        k: 2,
        decomposer: {
          async decompose(query) {
            decomposed.push(query);
            return ["first step", "second step"];
          },
        },
      },
    );

    const result = await retriever.retrieve([{ role: "user", content: "do both things" }]);

    expect(decomposed).toEqual(["do both things"]);
    expect(rerankCalls.map((call) => call.query)).toEqual(["first step", "second step"]);
    expect(result.trace.steps.map((step) => step.query)).toEqual(["first step", "second step"]);
  });

  test("reserves each step's top result before filling to k", async () => {
    const originals = [tool("alpha"), tool("beta"), tool("gamma"), tool("delta")];
    const { retriever } = await fixture(originals, ["s1", "s2"], {
      s1: { alpha: 10, beta: 9, gamma: 1, delta: 0 },
      s2: { alpha: 0, beta: 6, gamma: 8, delta: 7 },
    });

    const result = await retriever.retrieve("compound request");

    expect(result.tools).toEqual([originals[0], originals[2], originals[1]]);
    expect(result.trace.reserved.map((candidate) => candidate.name)).toEqual(["alpha", "gamma"]);
    expect(result.trace.selected.map((candidate) => candidate.name)).toEqual([
      "alpha",
      "gamma",
      "beta",
    ]);
  });

  test("deduplicates a tool reserved by multiple steps and fills open slots", async () => {
    const originals = [tool("alpha"), tool("beta"), tool("gamma"), tool("delta")];
    const { retriever } = await fixture(originals, ["s1", "s2"], {
      s1: { alpha: 10, beta: 9, gamma: 1, delta: 0 },
      s2: { alpha: 8, beta: 0, gamma: 7, delta: 6 },
    });

    const result = await retriever.retrieve("compound request");

    expect(result.tools).toEqual([originals[0], originals[1], originals[2]]);
    expect(result.trace.reserved.map((candidate) => candidate.name)).toEqual(["alpha", "alpha"]);
    expect(new Set(result.trace.selected.map((candidate) => candidate.id)).size).toBe(3);
  });

  test("min-max normalizes non-reserved scores within each query", async () => {
    const originals = [tool("alpha"), tool("beta"), tool("gamma"), tool("delta"), tool("epsilon")];
    const { retriever } = await fixture(
      originals,
      ["s1", "s2"],
      {
        s1: { alpha: 100, beta: 20, gamma: 2, delta: 1, epsilon: 0 },
        s2: { alpha: 1, beta: 2, gamma: 90, delta: 10, epsilon: 0 },
      },
      { k: 4 },
    );

    const result = await retriever.retrieve("compound request");
    const remaining = (step: number, name: string): MultiQueryCandidateTrace | undefined =>
      result.trace.remaining.find(
        (candidate) => candidate.stepIndex === step && candidate.name === name,
      );

    expect(remaining(0, "beta")?.normalizedScore).toBeCloseTo(1, 10);
    expect(remaining(1, "delta")?.normalizedScore).toBeCloseTo(1, 10);
    expect(remaining(1, "epsilon")?.normalizedScore).toBe(0);
    expect(result.trace.normalizationScope).toBe("per-query");
    expect(result.tools).toEqual([originals[0], originals[2], originals[1], originals[3]]);
  });

  test("returns the exact original tool objects", async () => {
    const originals = [tool("alpha"), tool("beta")];
    const { retriever } = await fixture(originals, ["s1"], { s1: { alpha: 2, beta: 1 } }, { k: 2 });

    const result = await retriever.retrieve("request");

    expect(result.tools[0]).toBe(originals[0]);
    expect(result.tools[1]).toBe(originals[1]);
  });

  test("breaks equal-score ties by step order, rank, then tool id", async () => {
    const originals = [tool("alpha"), tool("beta"), tool("gamma"), tool("delta")];
    const { retriever } = await fixture(originals, ["s1", "s2"], {
      s1: { alpha: 10, beta: 5, gamma: 1, delta: 0 },
      s2: { alpha: 0, beta: 1, gamma: 10, delta: 5 },
    });

    const first = await retriever.retrieve("request");
    const second = await retriever.retrieve("request");

    expect(first.tools).toEqual([originals[0], originals[2], originals[1]]);
    expect(second.tools).toEqual(first.tools);
  });

  test("normalizes a zero score range to zero and remains deterministic", async () => {
    const originals = [tool("alpha"), tool("beta"), tool("gamma"), tool("delta")];
    const { retriever } = await fixture(originals, ["s1", "s2"], {
      s1: { alpha: 10, beta: 5, gamma: 5, delta: 5 },
      s2: { alpha: 5, beta: 5, gamma: 10, delta: 5 },
    });

    const result = await retriever.retrieve("request");

    expect(result.trace.remaining.every((candidate) => candidate.normalizedScore === 0)).toBe(true);
    expect(result.tools).toEqual([originals[0], originals[2], originals[1]]);
  });

  test("returns no tools for an empty catalog", async () => {
    const { retriever } = await fixture([], ["s1", "s2"], {}, { k: 2 });

    const result = await retriever.retrieve("request");

    expect(result.tools).toEqual([]);
    expect(result.trace.steps).toHaveLength(2);
    expect(result.trace.selected).toEqual([]);
  });

  test.each([
    ["empty", []],
    ["blank", ["valid", "  "]],
    ["non-string", ["valid", 7]],
  ])("rejects %s decomposition output", async (_label, decomposition) => {
    const originals = [tool("alpha")];
    const { retriever } = await fixture(
      originals,
      ["unused"],
      { unused: { alpha: 1 } },
      {
        k: 2,
        decomposer: {
          async decompose() {
            return decomposition as string[];
          },
        },
      },
    );

    await expect(retriever.retrieve("request")).rejects.toThrow("Invalid query decomposition");
  });

  test("rejects k below one", async () => {
    const index = new ToolIndex({ embedder: constantEmbedder });

    expect(
      () =>
        new MultiQueryRetriever({
          index,
          decomposer: { decompose: () => ["step"] },
          reranker: { rerank: async () => [] },
          k: 0,
        }),
    ).toThrow("k must be at least 1");
  });

  test("rejects decompositions with more steps than k", async () => {
    const originals = [tool("alpha"), tool("beta")];
    const { retriever } = await fixture(originals, ["one", "two", "three"], {}, { k: 2 });

    await expect(retriever.retrieve("request")).rejects.toThrow(
      "Query decomposition has more steps than k",
    );
  });

  test.each([
    ["wrong count", async () => []],
    [
      "non-finite score",
      async (_query: string, candidates: CanonicalTool[]) => candidates.map(() => NaN),
    ],
  ])("rejects reranker output with %s", async (_label, rerank) => {
    const index = new ToolIndex({ embedder: constantEmbedder });
    await index.add([tool("alpha"), tool("beta")]);
    const retriever = new MultiQueryRetriever({
      index,
      decomposer: { decompose: () => ["step"] },
      reranker: { rerank },
      k: 1,
    });

    await expect(retriever.retrieve("request")).rejects.toThrow("Reranker returned invalid scores");
  });

  test("rerankPoolSize bounds every reranker call", async () => {
    const originals = [tool("alpha"), tool("beta"), tool("gamma"), tool("delta")];
    const index = new ToolIndex({ embedder: constantEmbedder });
    await index.add(originals);
    const rerankCallSizes: number[] = [];
    const retriever = new MultiQueryRetriever({
      index,
      decomposer: { decompose: () => ["first", "second"] },
      reranker: {
        async rerank(_query, candidates) {
          rerankCallSizes.push(candidates.length);
          return candidates.map(() => 1);
        },
      },
      k: 2,
      rerankPoolSize: 2,
    });

    await retriever.retrieve("request");

    expect(rerankCallSizes).toEqual([2, 2]);
  });

  test("limits each reranking pool to rerankPoolSize", async () => {
    const originals = [tool("alpha"), tool("beta"), tool("gamma"), tool("delta")];
    const { rerankCalls, retriever } = await fixture(
      originals,
      ["step"],
      { step: { alpha: 4, beta: 3, gamma: 2, delta: 1 } },
      { k: 2, rerankPoolSize: 3 },
    );

    await retriever.retrieve("request");

    expect(rerankCalls[0]?.names).toHaveLength(3);
  });

  test("forwards tag, namespace, and policy filters to direct catalog retrieval", async () => {
    const allowed = tool("allowed", { tags: ["read"], namespace: "prod" });
    const blocked = tool("blocked", { tags: ["read", "blocked"], namespace: "prod" });
    const wrongTag = tool("wrong_tag", { tags: ["write"], namespace: "prod" });
    const wrongNamespace = tool("wrong_namespace", { tags: ["read"], namespace: "dev" });
    const index = new ToolIndex({
      embedder: constantEmbedder,
      sticky: { enabled: true, reuseThreshold: 0 },
    });
    await index.add([allowed, blocked, wrongTag, wrongNamespace]);
    const policyCalls: string[] = [];
    const reranker: Reranker = { rerank: async (_query, candidates) => candidates.map(() => 1) };
    const retriever = new MultiQueryRetriever({
      index,
      decomposer: { decompose: () => ["step"] },
      reranker,
      k: 1,
      allowTags: ["read"],
    });

    const result = await retriever.retrieve("request", {
      namespace: "prod",
      policy: async (candidate) => {
        policyCalls.push(candidate.name);
        return !candidate.tags.includes("blocked");
      },
    });

    expect(result.tools).toEqual([allowed]);
    expect(policyCalls).toEqual(["allowed", "blocked"]);
  });

  test("exports deterministic BM25 scoring with term-frequency and rarity behavior", () => {
    const documents = ["rare rare common", "common common common", "other words"];

    const scores = bm25Scores("rare common", documents);

    expect(scores).toHaveLength(3);
    expect(scores[0]).toBeGreaterThan(scores[1]!);
    expect(scores[1]).toBeGreaterThan(scores[2]!);
    expect(bm25Scores("missing", documents)).toEqual([0, 0, 0]);
  });

  test.each([
    [0, "sparse"],
    [0.5, "dense"],
    [1, "dense"],
  ] as const)("uses alpha=%s to combine dense and BM25 scores", async (alpha, expected) => {
    const dense = tool("dense");
    const sparse = tool("sparse");
    const records: IndexedTool[] = [
      {
        tool: {
          id: "dense-id",
          name: "dense",
          description: dense.description,
          inputSchema: {},
          tags: [],
          original: dense,
        },
        vector: [1, 0],
        text: "unrelated material",
      },
      {
        tool: {
          id: "sparse-id",
          name: "sparse",
          description: sparse.description,
          inputSchema: {},
          tags: [],
          original: sparse,
        },
        vector: [0, 1],
        text: "target target",
      },
    ];
    const backend: VectorBackend = {
      size: records.length,
      upsert() {},
      remove() {},
      clear() {},
      list: () => records,
    };
    const index = new ToolIndex({ backend, embedder: { embed: async () => [[1, 0]] } });
    const candidateOrders: string[][] = [];
    const retriever = new MultiQueryRetriever({
      index,
      decomposer: { decompose: () => ["target"] },
      reranker: {
        async rerank(_query, candidates) {
          candidateOrders.push(candidates.map((candidate) => candidate.name));
          return candidates.map((_candidate, rank) => candidates.length - rank);
        },
      },
      k: 1,
      rerankPoolSize: 2,
      alpha,
    });

    const result = await retriever.retrieve("request");

    expect(candidateOrders[0]?.[0]).toBe(expected);
    expect((result.tools[0] as TestTool).name).toBe(expected);
  });

  test.each([-0.1, 1.1, Number.NaN])("rejects invalid alpha %s", (alpha) => {
    const index = new ToolIndex({ embedder: constantEmbedder });

    expect(
      () =>
        new MultiQueryRetriever({
          index,
          decomposer: { decompose: () => ["step"] },
          reranker: { rerank: async () => [] },
          k: 1,
          alpha,
        }),
    ).toThrow("alpha must be between 0 and 1");
  });

  test("rejects non-finite stored vectors before ranking", async () => {
    const original = tool("bad");
    const records: IndexedTool[] = [
      {
        tool: {
          id: "bad-id",
          name: "bad",
          description: original.description,
          inputSchema: {},
          tags: [],
          original,
        },
        vector: [Number.NaN],
        text: "bad",
      },
    ];
    const backend: VectorBackend = {
      size: 1,
      upsert() {},
      remove() {},
      clear() {},
      list: () => records,
    };
    const index = new ToolIndex({ backend, embedder: constantEmbedder });
    const retriever = new MultiQueryRetriever({
      index,
      decomposer: { decompose: () => ["step"] },
      reranker: { rerank: async () => [1] },
      k: 1,
    });

    await expect(retriever.retrieve("request")).rejects.toThrow(
      "Embedding vectors must contain only finite numbers",
    );
  });

  test("rejects a zero-norm stored vector", async () => {
    const retriever = vectorFixture([1, 0], [0, 0]);

    await expect(retriever.retrieve("request")).rejects.toThrow(
      "Stored embedding vector must have a non-zero norm",
    );
  });

  test("rejects a zero-norm query embedding before catalog scoring", async () => {
    const retriever = vectorFixture([0, 0]);

    await expect(retriever.retrieve("request")).rejects.toThrow(
      "Query embedding vector must have a non-zero norm",
    );
  });

  test("rejects finite vector components when cosine arithmetic overflows", async () => {
    const retriever = vectorFixture([1e154], [1e155]);

    await expect(retriever.retrieve("request")).rejects.toThrow(
      "Embedding dot product must be finite",
    );
  });
});
