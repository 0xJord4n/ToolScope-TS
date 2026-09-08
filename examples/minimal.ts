import { filter, HashEmbeddingProvider } from "../src";

const tools = [
  {
    name: "jira_create_issue",
    description: "Create a Jira issue",
    inputSchema: {},
  },
  {
    name: "confluence_search",
    description: "Search Confluence",
    inputSchema: {},
  },
];
console.log(
  await filter("create a jira ticket", tools, {
    embedder: new HashEmbeddingProvider(),
    k: 1,
  }),
);
