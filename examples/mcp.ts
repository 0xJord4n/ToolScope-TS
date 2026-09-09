import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { ToolListChangedNotificationSchema } from "@modelcontextprotocol/sdk/types.js";
import { HashEmbeddingProvider } from "../src";
import { ToolScopeMcpClient } from "../src/adapters/mcp";

const client = new Client({ name: "my-agent", version: "1.0.0" });
const scopedClient = new ToolScopeMcpClient(client, {
  embedder: new HashEmbeddingProvider(),
  k: 10,
});

// Register through your application's notification dispatcher. ToolScope does
// not silently replace handlers owned by the application.
client.setNotificationHandler(ToolListChangedNotificationSchema, scopedClient.toolsChangedHandler);

// Connect the official client to a transport, then:
// const { tools } = await scopedClient.listToolsFor(messages);
void scopedClient;
