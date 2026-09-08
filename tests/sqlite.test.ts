import { expect, test } from "bun:test";
import { unlinkSync } from "node:fs";
import { SqliteVectorBackend } from "../src/backends/sqlite";
import { HashEmbeddingProvider, ToolIndex } from "../src/index";

test("SQLite backend persists indexed tools", async () => {
  const path = `/tmp/toolscope-${crypto.randomUUID()}.db`;
  try {
    const backend = new SqliteVectorBackend(path);
    const first = new ToolIndex({
      embedder: new HashEmbeddingProvider(),
      backend,
    });
    await first.add([
      { name: "weather", description: "Get weather", inputSchema: {} },
    ]);
    backend.close();
    const reopened = new SqliteVectorBackend(path);
    const second = new ToolIndex({
      embedder: new HashEmbeddingProvider(),
      backend: reopened,
    });
    expect((await second.filter("weather", { k: 1 }))[0]).toMatchObject({
      name: "weather",
    });
    reopened.close();
  } finally {
    try {
      unlinkSync(path);
    } catch {}
  }
});

test("SQLite backend rejects a mismatched embedding dimension", async () => {
  const path = `/tmp/toolscope-${crypto.randomUUID()}.db`;
  try {
    const backend = new SqliteVectorBackend(path);
    const first = new ToolIndex({
      embedder: new HashEmbeddingProvider(32),
      backend,
    });
    await first.add([
      { name: "weather", description: "Get weather", inputSchema: {} },
    ]);
    backend.close();
    const reopened = new SqliteVectorBackend(path);
    const second = new ToolIndex({
      embedder: new HashEmbeddingProvider(64),
      backend: reopened,
    });
    expect(second.filter("weather")).rejects.toThrow(/dimension/i);
    reopened.close();
  } finally {
    try {
      unlinkSync(path);
    } catch {}
  }
});
