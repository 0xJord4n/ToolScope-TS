import { describe, expect, test } from "bun:test";
import {
  filter,
  HashEmbeddingProvider,
  normalizeTool,
  ToolIndex,
} from "../src/index";

const tools = [
  {
    name: "jira_create_issue",
    description: "Create a Jira ticket",
    inputSchema: {},
    tags: ["jira", "write"],
  },
  {
    name: "confluence_search",
    description: "Search documentation pages",
    inputSchema: {},
    tags: ["docs", "read"],
  },
  {
    type: "function",
    function: {
      name: "send_email",
      description: "Send an email message",
      parameters: { type: "object" },
    },
    tags: ["mail", "dangerous"],
  },
];
describe("normalization and filtering", () => {
  test("normalizes MCP and OpenAI tools while preserving originals", () => {
    expect(normalizeTool(tools[0]!).name).toBe("jira_create_issue");
    expect(normalizeTool(tools[2]!).name).toBe("send_email");
    expect(normalizeTool(tools[2]!).original).toBe(tools[2]!);
  });
  test("extracts ToolScope and MCP annotation tags", () => {
    expect(
      normalizeTool({
        name: "delete",
        description: "Delete",
        inputSchema: {},
        toolscope_tags: ["destructive"],
        annotations: { tags: ["admin"] },
      }).tags,
    ).toEqual(["destructive", "admin"]);
  });
  test("retrieves relevant tools and applies allow/deny tags", async () => {
    const index = new ToolIndex({ embedder: new HashEmbeddingProvider(256) });
    await index.add(tools);
    const result = await index.filter("file a jira bug ticket", {
      k: 2,
      denyTags: ["dangerous"],
    });
    expect(result[0]).toBe(tools[0]);
    expect(result).not.toContain(tools[2]);
    expect(await index.filter("search", { k: 5, allowTags: ["docs"] })).toEqual(
      [tools[1]],
    );
  });
  test("stateless filter has parity with indexed filtering", async () => {
    expect(
      await filter("search confluence documentation", tools, {
        embedder: new HashEmbeddingProvider(256),
        k: 1,
      }),
    ).toEqual([tools[1]]);
  });
  test("updates changed tools and removes deleted tools", async () => {
    const index = new ToolIndex({ embedder: new HashEmbeddingProvider(128) });
    await index.add(tools);
    expect(index.size).toBe(3);
    await index.sync([tools[0]!]);
    expect(index.size).toBe(1);
    expect(await index.filter("email", { k: 5 })).toEqual([]);
  });
  test("syncs an empty catalog without calling the embedder", async () => {
    const index = new ToolIndex({
      embedder: {
        async embed(texts) {
          if (texts.length === 0) throw new Error("empty batch unsupported");
          return texts.map(() => [1]);
        },
      },
    });
    await index.add([
      { name: "temporary", description: "Temporary", inputSchema: {} },
    ]);
    await expect(index.sync([])).resolves.toBe(index);
    expect(index.size).toBe(0);
  });
  test("rejects invalid embedding dimensions and vectors", async () => {
    expect(() => new HashEmbeddingProvider(0)).toThrow();
    const index = new ToolIndex({
      embedder: {
        async embed() {
          return [[1, Number.NaN]];
        },
      },
    });
    expect(
      index.add([{ name: "bad", description: "Bad", inputSchema: {} }]),
    ).rejects.toThrow();
  });
});
