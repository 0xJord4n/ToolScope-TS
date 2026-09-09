# ToolScope TS

[![npm version](https://img.shields.io/npm/v/toolscope-ts.svg)](https://www.npmjs.com/package/toolscope-ts)
[![npm downloads](https://img.shields.io/npm/dm/toolscope-ts.svg)](https://www.npmjs.com/package/toolscope-ts)
[![CI](https://github.com/0xJord4n/ToolScope-TS/actions/workflows/ci.yml/badge.svg)](https://github.com/0xJord4n/ToolScope-TS/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.9-3178C6.svg)](https://www.typescriptlang.org/)
[![Bun](https://img.shields.io/badge/Bun-1.3+-000000.svg)](https://bun.sh/)
[![GitHub stars](https://img.shields.io/github/stars/0xJord4n/ToolScope-TS.svg)](https://github.com/0xJord4n/ToolScope-TS/stargazers)

**Semantic tool retrieval for TypeScript agents, powered by Bun.**

ToolScope TS selects the most relevant tools for each prompt before the model runs. It keeps large tool catalogs out of the context window while preserving original executable tool objects when using the default in-memory backend.

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
- **Original object identity in memory** — the default backend returns executable tool objects instead of reconstructed wrappers
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

In-memory indexes return original tool values by identity. SQLite stores and returns JSON-safe descriptors only; map selected descriptor IDs or names back to trusted runtime implementations before execution.

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

`MultiQueryRetrievalTrace` is separate from the core `SelectionTrace`: it includes the original query and generated subqueries and is not automatically processed by `redactTrace(...)`. Redact it explicitly before persistence.

## Package exports

| Import                            | Purpose                                                               |
| --------------------------------- | --------------------------------------------------------------------- |
| `toolscope-ts`                    | Core index, filters, embeddings, memory backend, policies, and traces |
| `toolscope-ts/adapters/vercel-ai` | Vercel AI SDK selection helpers                                       |
| `toolscope-ts/adapters/langgraph` | LangGraph selectors and dynamic model binding                         |
| `toolscope-ts/adapters/mcp`       | MCP catalog synchronization and selection                             |
| `toolscope-ts/backends/sqlite`    | Bun SQLite persistence                                                |
| `toolscope-ts/paper`              | Opt-in merger, Auto-Correction, BM25/dense multi-query retrieval      |

The package ships ESM JavaScript and TypeScript declarations. Framework adapters are package subpath exports; install the corresponding optional peer dependencies only for the integrations your application uses.

## Paper-inspired pipeline

An opt-in pipeline implements the merger, Auto-Correction, and multi-query retrieval stages described in the ACL 2026 paper [“ToolScope: Enhancing LLM Agent Tool Use through Tool Merging and Context-Aware Filtering”](https://aclanthology.org/2026.acl-long.1573/).

```ts
import { ToolIndex } from "toolscope-ts";
import { ToolScopeMerger, ToolScopeRetriever } from "toolscope-ts/paper";

const merged = await new ToolScopeMerger({
  embedder,
  classifier,
  validator,
  synthesizer,
}).merge(tools);

const mergedIndex = new ToolIndex({ embedder });
await mergedIndex.add(merged.merged);

const retriever = new ToolScopeRetriever({
  index: mergedIndex,
  decomposer,
  reranker,
  k: 8,
});
const { tools: selected } = await retriever.retrieve(messages);

// Resolve a prompt-facing merged descriptor back to trusted originals.
const originals = merged.manifest.resolve(selected[0]!.id);
```

The module provides:

- top-30 dense merger candidates with the paper's default `0.82` threshold
- relationship graph clustering, shortest-name representatives, and bounded Auto-Correction
- capability-preserving schema consolidation and a reversible original-tool manifest
- ordered query decomposition, real BM25 plus dense cosine candidate scoring, and injected cross-encoder reranking
- per-subquery min-max normalization, one reserved result per subquery, then deterministic global top-`k` filling
- provider-neutral prompt builders for classification, correction, synthesis, and decomposition

Resource hardening is enabled by default. Merger model calls run with ordered concurrency `8`; catalogs are capped at `1,000` tools, vectors at `4,096` dimensions, provider-facing descriptors at `16,384` UTF-16 code units, and relationship classification at `30,000` candidate calls. Retrieval uses the same catalog and vector caps and limits both original and generated queries to `16,384` UTF-16 code units. These values are separately exported as `PAPER_2026_MERGER_RESOURCE_DEFAULTS` and `PAPER_2026_RETRIEVER_RESOURCE_DEFAULTS` and may be lowered per deployment. Limits fail closed rather than silently truncating semantic work.

Exact dense merger candidate discovery still performs quadratic pairwise similarity comparisons within the finite catalog cap. Injected providers own timeout, cancellation, retry, and rate-limit policy.

Model-dependent behavior stays behind injected interfaces. Descriptor synthesis accepts only `{ description }`; schemas come exclusively from deterministic consolidation of trusted originals. Generated output is validated and becomes descriptor data—not executable code. Merged descriptors must be resolved through their manifest and authorized against trusted original implementations before execution.

Deliberate safety adaptations reject model-generated schemas, tags, namespaces, and other descriptor fields; preserve trusted tags and namespaces; block cross-namespace merging by default; and never synthesize an executor.

Zod and data-property `toJSON` schema adapters are executable caller-provided objects, not detached JSON. Use them only from trusted application or framework code. Their converted output is strictly validated before fingerprinting; generated or remote schema data should be supplied as plain JSON instead.

This module implements the paper's algorithmic stages and documented defaults; it does not claim reproduction of the paper's benchmark results or model-specific outputs. The paper's authors and institutions are not affiliated with or responsible for ToolScope TS. See the [complete runnable example](examples/paper-pipeline.ts).

## Examples

- [Minimal selection](examples/minimal.ts)
- [Vercel AI SDK](examples/vercel-ai.ts)
- [LangGraph.js](examples/langgraph.ts)
- [MCP](examples/mcp.ts)
- [Paper-inspired merger and multi-query pipeline](examples/paper-pipeline.ts)

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
