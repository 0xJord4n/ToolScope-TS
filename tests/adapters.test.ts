import { describe, expect, test } from "bun:test";
import { createLangGraphToolSelector } from "../src/adapters/langgraph";
import { ToolScopeMcpClient } from "../src/adapters/mcp";
import {
  createVercelPrepareStep,
  selectVercelActiveTools,
  selectVercelTools,
} from "../src/adapters/vercel-ai";
import { HashEmbeddingProvider, MemoryVectorBackend } from "../src/index";

const aiTools = {
  weather: {
    description: "Get weather",
    inputSchema: { type: "object" },
    execute: async () => "sunny",
  },
  stocks: {
    description: "Get stock price",
    inputSchema: { type: "object" },
    execute: async () => 42,
  },
};
describe("framework adapters", () => {
  test("Vercel AI adapter preserves the keyed tool record", async () => {
    const selected = await selectVercelTools("weather in Paris", aiTools, {
      embedder: new HashEmbeddingProvider(),
      k: 1,
    });
    expect(Object.keys(selected)).toEqual(["weather"]);
    expect(selected.weather).toBe(aiTools.weather);
  });
  test("Vercel AI adapter produces activeTools and prepareStep", async () => {
    const active = await selectVercelActiveTools("stock quote", aiTools, {
      embedder: new HashEmbeddingProvider(),
      k: 1,
    });
    expect(active).toEqual(["stocks"]);
    const prepareStep = createVercelPrepareStep(aiTools, {
      embedder: new HashEmbeddingProvider(),
      k: 1,
    });
    expect(
      await prepareStep({ messages: [{ role: "user", content: "weather" }] }),
    ).toEqual({ activeTools: ["weather"] });
  });
  test("Vercel AI adapter preserves policy metadata", async () => {
    const tagged = {
      safe: { description: "Read data", inputSchema: {}, tags: ["read"] },
      dangerous: {
        description: "Delete data",
        inputSchema: {},
        annotations: { tags: ["destructive"] },
      },
    };
    expect(
      Object.keys(
        await selectVercelTools("delete data", tagged, {
          embedder: new HashEmbeddingProvider(),
          k: 2,
          denyTags: ["destructive"],
        }),
      ),
    ).toEqual(["safe"]);
  });
  test("LangGraph selector accepts tool-like objects", async () => {
    const selector = createLangGraphToolSelector(
      [
        {
          name: "weather",
          description: "Get weather",
          schema: { type: "object" },
        },
        {
          name: "stocks",
          description: "Get stocks",
          schema: { type: "object" },
        },
      ],
      { embedder: new HashEmbeddingProvider(), k: 1 },
    );
    expect((await selector("weather in Paris"))[0]?.name).toBe("weather");
  });
  test("MCP wrapper refreshes and passes calls through", async () => {
    let changed: (() => void) | undefined;
    const client = {
      async listTools() {
        return {
          tools: [
            { name: "weather", description: "Get weather", inputSchema: {} },
            { name: "stocks", description: "Get stocks", inputSchema: {} },
          ],
        };
      },
      async callTool(request: unknown) {
        return request;
      },
      onToolsChanged(handler: () => void) {
        changed = handler;
      },
    };
    const wrapped = new ToolScopeMcpClient(client, {
      embedder: new HashEmbeddingProvider(),
      backend: new MemoryVectorBackend(),
      k: 5,
    });
    expect(
      ((await wrapped.listToolsFor("weather")).tools[0] as { name: string })
        ?.name,
    ).toBe("weather");
    changed?.();
    expect(await wrapped.callTool({ name: "weather" })).toEqual({
      name: "weather",
    });
  });
  test("MCP refresh removes revoked tools from a reused backend", async () => {
    let current: unknown[] = [
      { name: "weather", description: "Get weather", inputSchema: {} },
      {
        name: "delete",
        description: "Delete account",
        inputSchema: {},
        tags: ["destructive"],
      },
    ];
    const client = {
      async listTools() {
        return { tools: current };
      },
      async callTool(request: unknown) {
        return request;
      },
    };
    const wrapped = new ToolScopeMcpClient(client, {
      embedder: new HashEmbeddingProvider(),
      backend: new MemoryVectorBackend(),
      k: 5,
    });
    await wrapped.listToolsFor("delete");
    current = [
      { name: "weather", description: "Get weather", inputSchema: {} },
    ];
    wrapped.markDirty();
    expect((await wrapped.listToolsFor("delete")).tools).toEqual([]);
  });
  test("MCP refresh follows pagination and updates changed policy tags", async () => {
    let revoked = false;
    const client = {
      async listTools(request?: unknown) {
        if (request) {
          return {
            tools: [
              { name: "second", description: "Second page", inputSchema: {} },
            ],
          };
        }
        return {
          tools: [
            {
              name: "first",
              description: "First page",
              inputSchema: {},
              tags: [revoked ? "revoked" : "allowed"],
            },
          ],
          nextCursor: "page-2",
        };
      },
      async callTool(request: unknown) {
        return request;
      },
    };
    const wrapped = new ToolScopeMcpClient(client, {
      embedder: new HashEmbeddingProvider(),
      k: 5,
    });
    expect(
      (await wrapped.listToolsFor("page", { minScore: -1 })).tools,
    ).toHaveLength(2);
    revoked = true;
    wrapped.toolsChangedHandler();
    const refreshed = await wrapped.listToolsFor("first", {
      denyTags: ["revoked"],
      minScore: -1,
    });
    expect(
      refreshed.tools.map((tool) => (tool as { name: string }).name),
    ).toEqual(["second"]);
  });
});
