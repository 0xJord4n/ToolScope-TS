import { describe, expect, test } from "bun:test";
import { readFileSync, unlinkSync } from "node:fs";
import { HashEmbeddingProvider, JsonlTraceSink, ToolIndex, type TraceSink } from "../src/index";

const tools = [
  {
    name: "weather_now",
    description: "Get current weather",
    inputSchema: {},
    tags: ["read"],
  },
  {
    name: "weather_forecast",
    description: "Get a future weather forecast",
    inputSchema: {},
    tags: ["read"],
  },
  {
    name: "delete_account",
    description: "Permanently delete an account",
    inputSchema: {},
    tags: ["destructive"],
  },
];
describe("advanced selection", () => {
  test("returns rich traces to sinks", async () => {
    const traces: unknown[] = [];
    const sink: TraceSink = {
      emit(trace) {
        traces.push(trace);
      },
    };
    const index = new ToolIndex({
      embedder: new HashEmbeddingProvider(),
      traceSinks: [sink],
    });
    await index.add(tools);
    const { tools: selected, trace } = await index.filterWithTrace("weather tomorrow", { k: 1 });
    expect(selected).toHaveLength(1);
    expect(trace.candidateCount).toBe(3);
    expect(trace.selected).toHaveLength(1);
    expect(trace.timings.totalMs).toBeGreaterThanOrEqual(0);
    expect(traces).toHaveLength(1);
  });
  test("JSONL trace sink redacts queries", async () => {
    const path = `/tmp/toolscope-trace-${crypto.randomUUID()}.jsonl`;
    try {
      const index = new ToolIndex({
        embedder: new HashEmbeddingProvider(),
        traceSinks: [
          new JsonlTraceSink(path, (trace) => ({
            ...trace,
            query: "[REDACTED]",
          })),
        ],
      });
      await index.add(tools);
      await index.filter("secret weather request", { k: 1 });
      const saved = JSON.parse(readFileSync(path, "utf8").trim());
      expect(saved.query).toBe("[REDACTED]");
    } finally {
      try {
        unlinkSync(path);
      } catch {}
    }
  });
  test("sticky sessions retain a previously selected tool", async () => {
    const index = new ToolIndex({
      embedder: new HashEmbeddingProvider(),
      sticky: {
        enabled: true,
        reuseThreshold: 0,
        refreshThreshold: 0,
        keep: 1,
        ttlMs: 60000,
      },
    });
    await index.add(tools);
    const first = await index.filter("current weather", {
      k: 1,
      sessionId: "s1",
      minScore: -1,
    });
    const second = await index.filter("forecast tomorrow", {
      k: 2,
      sessionId: "s1",
      minScore: -1,
    });
    expect(second).toContain(first[0]);
  });
  test("sticky reuse threshold reuses the previous toolset", async () => {
    const index = new ToolIndex({
      embedder: new HashEmbeddingProvider(),
      sticky: {
        enabled: true,
        reuseThreshold: 0,
        refreshThreshold: 2,
        keep: 1,
        ttlMs: 60000,
      },
    });
    await index.add(tools);
    const first = await index.filter("current weather", {
      k: 1,
      sessionId: "reuse",
      minScore: -1,
    });
    const second = await index.filter("forecast tomorrow", {
      k: 1,
      sessionId: "reuse",
      minScore: -1,
    });
    expect(second).toEqual(first);
  });
  test("bounds sticky session memory", async () => {
    const index = new ToolIndex({
      embedder: new HashEmbeddingProvider(),
      sticky: { enabled: true, maxSessions: 2 },
    });
    await index.add(tools);
    await index.filter("weather", { k: 1, sessionId: "a" });
    await index.filter("weather", { k: 1, sessionId: "b" });
    await index.filter("weather", { k: 1, sessionId: "c" });
    expect(index.sessionCount).toBe(2);
  });
  test("policy callback can exclude tools", async () => {
    const index = new ToolIndex({ embedder: new HashEmbeddingProvider() });
    await index.add(tools);
    expect(
      await index.filter("delete my account", {
        k: 3,
        policy: (tool) => !tool.tags.includes("destructive"),
      }),
    ).not.toContain(tools[2]);
  });
  test("rejects malformed vectors from persistent backends", async () => {
    const backend = {
      size: 2,
      upsert() {},
      remove() {},
      clear() {},
      list() {
        return [
          {
            tool: {
              id: "a",
              name: "a",
              description: "a",
              inputSchema: {},
              tags: [],
              original: { name: "a" },
            },
            text: "a",
            vector: [1, 0],
          },
          {
            tool: {
              id: "b",
              name: "b",
              description: "b",
              inputSchema: {},
              tags: [],
              original: { name: "b" },
            },
            text: "b",
            vector: [1, 0, 0],
          },
        ];
      },
    };
    const index = new ToolIndex({
      backend,
      embedder: {
        async embed() {
          return [[1, 0]];
        },
      },
    });
    expect(index.filter("a")).rejects.toThrow("Embedding dimension mismatch");
  });
});
