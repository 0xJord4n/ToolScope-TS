import { describe, expect, test } from "bun:test";
import { tool as langchainTool } from "@langchain/core/tools";
import { Annotation, END, START, StateGraph } from "@langchain/langgraph";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { tool as aiTool } from "ai";
import { z } from "zod";
import { HashEmbeddingProvider, normalizeTool } from "../src";
import {
  createLangGraphSelectionNode,
  createLangGraphToolSelector,
} from "../src/adapters/langgraph";
import { ToolScopeMcpClient } from "../src/adapters/mcp";
import { selectVercelActiveTools } from "../src/adapters/vercel-ai";

describe("real framework packages", () => {
  test("selects a real Vercel AI SDK tool", async () => {
    const tools = {
      weather: aiTool({
        description: "Get city weather",
        inputSchema: z.object({ city: z.string() }),
        execute: async ({ city }) => `${city}: sunny`,
      }),
      stocks: aiTool({
        description: "Get stock price",
        inputSchema: z.object({ symbol: z.string() }),
        execute: async ({ symbol }) => `${symbol}: 42`,
      }),
    };
    expect(
      await selectVercelActiveTools("weather for Seoul", tools, {
        embedder: new HashEmbeddingProvider(),
        k: 1,
      }),
    ).toEqual(["weather"]);
    expect(
      normalizeTool({ name: "weather", schema: z.object({ city: z.string() }) })
        .inputSchema,
    ).toMatchObject({
      type: "object",
      properties: { city: { type: "string" } },
    });
  });

  test("selects real LangChain tools and runs as a LangGraph node", async () => {
    const weather = langchainTool(async ({ city }) => `${city}: sunny`, {
      name: "weather",
      description: "Get city weather",
      schema: z.object({ city: z.string() }),
    });
    const stocks = langchainTool(async ({ symbol }) => `${symbol}: 42`, {
      name: "stocks",
      description: "Get stock price",
      schema: z.object({ symbol: z.string() }),
    });
    const options = { embedder: new HashEmbeddingProvider(), k: 1 };
    const selector = createLangGraphToolSelector([weather, stocks], options);
    expect((await selector("weather in Seoul"))[0]).toBe(weather);

    const State = Annotation.Root({
      messages: Annotation<unknown[]>({
        reducer: (_left, right) => right,
        default: () => [],
      }),
      selectedTools: Annotation<unknown[]>({
        reducer: (_left, right) => right,
        default: () => [],
      }),
      toolScopeQuery: Annotation<string>(),
    });
    const graph = new StateGraph(State)
      .addNode(
        "select",
        createLangGraphSelectionNode([weather, stocks], options),
      )
      .addEdge(START, "select")
      .addEdge("select", END)
      .compile();
    const result = await graph.invoke({
      messages: [{ role: "user", content: "weather in Seoul" }],
    });
    expect((result.selectedTools[0] as { name: string }).name).toBe("weather");
  });

  test("accepts the official MCP TypeScript client", () => {
    const client = new Client({ name: "toolscope-test", version: "1.0.0" });
    expect(
      () =>
        new ToolScopeMcpClient(client, {
          embedder: new HashEmbeddingProvider(),
        }),
    ).not.toThrow();
  });
});
