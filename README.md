# ToolScope TS

Semantic, per-prompt tool selection for TypeScript agents. It keeps large tool catalogs out of the model context without introducing a meta-tool or changing how tools are called. Built for **Bun**, with first-class adapters for **LangGraph.js**, the **Vercel AI SDK**, and **MCP**.

This is a TypeScript port and extension of [ToolScope](https://github.com/ilya-kolchinsky/ToolScope). See [NOTICE](NOTICE).

## Why

Models become less reliable and prompts become expensive when every turn includes hundreds or thousands of tool schemas. ToolScope TS embeds the current request, retrieves the best tools, applies policy, and returns the original tool objects unchanged.

## Features

- Canonical normalization for MCP, OpenAI function tools, LangChain/LangGraph tools, and Vercel AI tool records
- Stateless `filter(...)` and reusable `ToolIndex` APIs
- Pluggable embedding providers; HTTP/OpenAI-compatible and deterministic local hashing included
- Hybrid semantic + lexical scoring, score thresholds, custom rerankers, and MMR diversity
- Allow/deny tags, namespaces, and async policy callbacks
- Sticky toolsets for multi-turn sessions
- Rich timing/score traces with callback, console, and redacted JSONL sinks
- In-memory backend and Bun SQLite persistence
- MCP client wrapper with `tools/list_changed` invalidation
- Vercel AI SDK `activeTools` and `prepareStep` integration
- LangGraph dynamic `bindTools` selector and graph selection node
- No hidden model downloads and no required cloud service

## Install

Install the public package with your preferred package manager:

```bash
# npm
npm install toolscope-ts

# pnpm
pnpm add toolscope-ts

# Yarn
yarn add toolscope-ts

# Bun
bun add toolscope-ts
```

Authenticated GitHub installs are also supported:

```bash
# npm
npm install github:0xJord4n/ToolScope-TS

# pnpm
pnpm add github:0xJord4n/ToolScope-TS

# Yarn
yarn add github:0xJord4n/ToolScope-TS

# Bun
bun add github:0xJord4n/ToolScope-TS
```

For local development:

```bash
git clone git@github.com:0xJord4n/ToolScope-TS.git
cd ToolScope-TS
bun install
bun run check
```

## Minimal usage

```ts
import { filter, HashEmbeddingProvider } from "toolscope-ts";

const tools = [
  { name: "jira_create_issue", description: "Create a Jira issue", inputSchema: {} },
  { name: "confluence_search", description: "Search Confluence pages", inputSchema: {} },
];

const selected = await filter("Create a Jira ticket", tools, {
  embedder: new HashEmbeddingProvider(),
  k: 1,
});
```

`HashEmbeddingProvider` is dependency-free and ideal for tests/offline use. For production semantic quality, plug in your embedding service:

```ts
import { HttpEmbeddingProvider, index } from "toolscope-ts";

const toolIndex = await index(tools, {
  embedder: new HttpEmbeddingProvider({
    endpoint: "https://api.openai.com/v1/embeddings",
    model: "text-embedding-3-small",
    apiKey: process.env.OPENAI_API_KEY,
  }),
});

const selected = await toolIndex.filter(messages, {
  k: 8,
  allowTags: ["read"],
  denyTags: ["destructive"],
  sessionId: conversationId,
});
```

## Vercel AI SDK

Use `prepareStep` to retrieve a fresh tool subset on every model step while preserving one stable `tools` record:

```ts
import { generateText } from "ai";
import { createVercelPrepareStep } from "toolscope-ts/adapters/vercel-ai";

const result = await generateText({
  model,
  tools,
  prompt: "Check the weather in Seoul",
  prepareStep: createVercelPrepareStep(tools, { embedder, k: 8 }),
});
```

You can also call `selectVercelTools(...)` for a filtered record or `selectVercelActiveTools(...)` for tool-name keys.

## LangGraph.js

```ts
import { createLangGraphToolSelector, bindSelectedTools } from "toolscope-ts/adapters/langgraph";

const select = createLangGraphToolSelector(allTools, { embedder, k: 8 });

const modelNode = async (state: { messages: unknown[] }) => {
  const scopedModel = await bindSelectedTools(model, select, state.messages);
  return { messages: [await scopedModel.invoke(state.messages)] };
};
```

Keep a `ToolNode(allTools)` for execution; only the model binding is narrowed. `createLangGraphSelectionNode(...)` is available when selected tools should be stored in graph state.

## MCP

```ts
import { ToolScopeMcpClient } from "toolscope-ts/adapters/mcp";
import { ToolListChangedNotificationSchema } from "@modelcontextprotocol/sdk/types.js";

const client = new ToolScopeMcpClient(mcpClient, { embedder, k: 10 });
mcpClient.setNotificationHandler(ToolListChangedNotificationSchema, client.toolsChangedHandler);
const { tools } = await client.listToolsFor(messages);
await client.callTool({ name: tools[0].name, arguments: {} });
```

The wrapper delegates execution, follows paginated tool catalogs, and safely exposes a list-change handler for your application's notification dispatcher. It does not silently replace handlers owned by the application. Custom clients may instead expose `onToolsChanged(handler)`.

## Persistent index

```ts
import { SqliteVectorBackend } from "toolscope-ts/backends/sqlite";
import { ToolIndex } from "toolscope-ts";

const backend = new SqliteVectorBackend("./data/toolscope.db");
const toolIndex = new ToolIndex({ embedder, backend });
await toolIndex.sync(tools);
```

## Selection controls

```ts
const { tools: selected, trace } = await toolIndex.filterWithTrace(messages, {
  k: 12,
  minScore: 0.1,
  semanticWeight: 0.85,
  lexicalWeight: 0.15,
  diversity: 0.25,
  allowTags: ["github"],
  denyTags: ["admin", "destructive"],
  namespace: "engineering",
  policy: async (tool) => authorize(user, tool.name),
  reranker,
  rerankPoolSize: 30,
});
```

Original tool values are returned by identity for in-memory indexes. SQLite serializes JSON-safe tool descriptors; use the persistent backend for catalogs/descriptors, not executable function closures.

## Observability and redaction

```ts
import { JsonlTraceSink, redactTrace, ToolIndex } from "toolscope-ts";

const toolIndex = new ToolIndex({
  embedder,
  traceSinks: [new JsonlTraceSink("./traces.jsonl", redactTrace({ query: true }))],
});
```

Traces contain candidate counts, selected and rejected scores, filter configuration, and embedding/search/rerank timings. Do not persist raw prompts unless your privacy policy allows it.

## Development

```bash
bun test
bun run typecheck
bun run build
```

## Releases and publishing

Conventional commits on `main` are managed by Release Please. Merging its release PR updates the changelog and version, creates a `v*` GitHub release, and triggers the npm publishing workflow for `toolscope-ts`.

Publishing supports npm trusted publishing through GitHub Actions OIDC. The initial npm package must be bootstrapped once by an npm owner, then configured with trusted publisher repository `0xJord4n/ToolScope-TS` and workflow `release.yml`. An `NPM_TOKEN` repository secret can be used for that initial publish.

## License

Apache-2.0. This derivative preserves attribution to the original ToolScope project in [NOTICE](NOTICE).
