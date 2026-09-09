# ToolScope TS

[![npm version](https://img.shields.io/npm/v/toolscope-ts.svg)](https://www.npmjs.com/package/toolscope-ts)
[![npm downloads](https://img.shields.io/npm/dm/toolscope-ts.svg)](https://www.npmjs.com/package/toolscope-ts)
[![CI](https://github.com/0xJord4n/ToolScope-TS/actions/workflows/ci.yml/badge.svg)](https://github.com/0xJord4n/ToolScope-TS/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.9-3178C6.svg)](https://www.typescriptlang.org/)
[![Bun](https://img.shields.io/badge/Bun-1.3+-000000.svg)](https://bun.sh/)
[![GitHub stars](https://img.shields.io/github/stars/0xJord4n/ToolScope-TS.svg)](https://github.com/0xJord4n/ToolScope-TS/stargazers)

**Semantic tool retrieval for TypeScript agents, powered by Bun.**

ToolScope TS selects the most relevant tools for each prompt before the model runs. It keeps large tool catalogs out of the context window while preserving the original executable tool objects and their normal calling behavior.

Built for [LangGraph.js](https://github.com/langchain-ai/langgraphjs), the [Vercel AI SDK](https://ai-sdk.dev/), and the [Model Context Protocol](https://modelcontextprotocol.io/).

[**npm**](https://www.npmjs.com/package/toolscope-ts) · [**latest release**](https://github.com/0xJord4n/ToolScope-TS/releases/latest) · [**issues**](https://github.com/0xJord4n/ToolScope-TS/issues)

## Why ToolScope TS?

Sending every available tool schema on every turn increases token usage and can make tool selection less reliable. ToolScope TS adds a retrieval step in front of the model:

1. Normalize tools from supported frameworks into one internal representation.
2. Embed the current request and retrieve the strongest candidates.
3. Apply tags, namespaces, policies, thresholds, reranking, and diversity controls.
4. Return the original tool objects unchanged for normal execution.

It is a selector—not an agent, proxy, executor, or meta-tool.

## Highlights

- **Framework-native** — accepts MCP, OpenAI-style, LangChain/LangGraph, and Vercel AI tools
- **Original object identity** — returns executable tool objects instead of reconstructed wrappers
- **Flexible retrieval** — semantic + lexical scoring, custom rerankers, thresholds, and MMR diversity
- **Policy-aware** — allow/deny tags, namespaces, async callbacks, and metadata-safe synchronization
- **Multi-turn ready** — bounded sticky toolsets reduce unnecessary selection churn
- **Observable** — structured candidate scores, selection counts, timing traces, and redaction helpers
- **Persistent when needed** — in-memory operation by default, with Bun SQLite support
- **Self-contained** — no required cloud service, hidden model download, or telemetry

## Install

```bash
npm install toolscope-ts
```

<details>
<summary>Other package managers</summary>

```bash
pnpm add toolscope-ts
yarn add toolscope-ts
bun add toolscope-ts
```

</details>

## Quick start

```ts
import { filter, HashEmbeddingProvider } from "toolscope-ts";

const tools = [
  {
    name: "jira_create_issue",
    description: "Create a Jira issue",
    inputSchema: {},
  },
  {
    name: "confluence_search",
    description: "Search Confluence pages",
    inputSchema: {},
  },
];

const selected = await filter("Create a Jira ticket", tools, {
  embedder: new HashEmbeddingProvider(),
  k: 1,
});

console.log(selected[0]); // the original jira_create_issue tool
```

`HashEmbeddingProvider` is deterministic, dependency-free, and useful for tests or offline workflows. For production semantic quality, connect an embedding service.

## Production embeddings

`HttpEmbeddingProvider` supports OpenAI-compatible embedding endpoints:

```ts
import { HttpEmbeddingProvider, index } from "toolscope-ts";

const embedder = new HttpEmbeddingProvider({
  endpoint: "https://api.openai.com/v1/embeddings",
  model: "text-embedding-3-small",
  apiKey: process.env.OPENAI_API_KEY,
});

const toolIndex = await index(tools, { embedder });
const selected = await toolIndex.filter(messages, { k: 8 });
```

You can also implement `EmbeddingProvider` directly or wrap any function with `FunctionEmbeddingProvider`.

## Framework integrations

### Vercel AI SDK

Use `prepareStep` to select a fresh subset on each model step while keeping one stable tool record:

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

The adapter also exposes `selectVercelTools(...)`, `selectVercelActiveTools(...)`, and `createVercelToolSelector(...)`.

### LangGraph.js

Select tools before binding them to the model:

```ts
import { bindSelectedTools, createLangGraphToolSelector } from "toolscope-ts/adapters/langgraph";

const select = createLangGraphToolSelector(allTools, { embedder, k: 8 });

const modelNode = async (state: { messages: unknown[] }) => {
  const scopedModel = await bindSelectedTools(model, select, state.messages);
  return { messages: [await scopedModel.invoke(state.messages)] };
};
```

Keep `ToolNode(allTools)` available for execution; only the model binding is narrowed. Use `createLangGraphSelectionNode(...)` when selected tools should be stored in graph state.

### MCP

Synchronize an MCP catalog and retrieve only the tools needed for the current request:

```ts
import { ToolListChangedNotificationSchema } from "@modelcontextprotocol/sdk/types.js";
import { ToolScopeMcpClient } from "toolscope-ts/adapters/mcp";

const client = new ToolScopeMcpClient(mcpClient, { embedder, k: 10 });

mcpClient.setNotificationHandler(ToolListChangedNotificationSchema, client.toolsChangedHandler);

const { tools } = await client.listToolsFor(messages);
await client.callTool({ name: tools[0].name, arguments: {} });
```

The wrapper follows paginated catalogs, synchronizes additions and removals, and exposes list-change invalidation without silently replacing application-owned handlers.

## Retrieval and policy controls

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
  policy: async (tool) => isEligibleForRetrieval(user, tool.name),
  reranker,
  rerankPoolSize: 30,
});
```

`policy` controls retrieval eligibility, not execution authorization. Authorize the selected tool again before running it.

Available controls include:

- semantic and lexical hybrid scoring
- minimum score thresholds and top-`k` limits
- custom rerankers and MMR diversity
- allow/deny tags and namespace filtering
- synchronous or asynchronous policy callbacks
- bounded sticky toolsets for multi-turn sessions

Policy metadata is refreshed during catalog synchronization so changed authorization tags and executable identities are not retained as stale records.

## Persistence

Use the in-memory backend by default, or persist descriptor catalogs with Bun SQLite:

```ts
import { ToolIndex } from "toolscope-ts";
import { SqliteVectorBackend } from "toolscope-ts/backends/sqlite";

const backend = new SqliteVectorBackend("./data/toolscope.db");
const toolIndex = new ToolIndex({ embedder, backend });

await toolIndex.sync(tools);
```

In-memory indexes return original tool values by identity. SQLite stores JSON-safe descriptors, so executable function closures should be supplied again by the application rather than treated as persistent data.

## Observability and privacy

```ts
import { JsonlTraceSink, redactTrace, ToolIndex } from "toolscope-ts";

const toolIndex = new ToolIndex({
  embedder,
  traceSinks: [new JsonlTraceSink("./traces.jsonl", redactTrace({ query: true }))],
});
```

Selection traces include candidate and filtered counts, per-candidate scores, selected tools, recorded retrieval settings, and embedding/search/rerank timings. Callback and console sinks are also included.

Raw prompts may contain sensitive information. Redact or avoid persisting query text unless your privacy policy permits it.

## Package exports

| Import                            | Purpose                                                               |
| --------------------------------- | --------------------------------------------------------------------- |
| `toolscope-ts`                    | Core index, filters, embeddings, memory backend, policies, and traces |
| `toolscope-ts/adapters/vercel-ai` | Vercel AI SDK selection helpers                                       |
| `toolscope-ts/adapters/langgraph` | LangGraph selectors and dynamic model binding                         |
| `toolscope-ts/adapters/mcp`       | MCP catalog synchronization and selection                             |
| `toolscope-ts/backends/sqlite`    | Bun SQLite persistence                                                |

The package ships ESM JavaScript and TypeScript declarations. Framework integrations are optional peer dependencies; install only the adapters your application uses.

## Examples

- [Minimal selection](examples/minimal.ts)
- [Vercel AI SDK](examples/vercel-ai.ts)
- [LangGraph.js](examples/langgraph.ts)
- [MCP](examples/mcp.ts)

## Development

Requires Bun 1.3 or newer.

```bash
git clone https://github.com/0xJord4n/ToolScope-TS.git
cd ToolScope-TS
bun install
bun run check
bun run package:check
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the complete contribution workflow.

## Security

Do not report vulnerabilities through public issues. See [SECURITY.md](SECURITY.md) for supported versions and private reporting instructions.

## Support

For usage questions, bug reports, and feature requests, [open an issue](https://github.com/0xJord4n/ToolScope-TS/issues). Include a minimal reproduction when reporting unexpected behavior.

## License and attribution

Licensed under [Apache-2.0](LICENSE).

ToolScope TS is a TypeScript port and extension of [ToolScope](https://github.com/ilya-kolchinsky/ToolScope) by Ilya Kolchinsky. See [NOTICE](NOTICE) for attribution and modification details.
