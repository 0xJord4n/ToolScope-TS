import { HashEmbeddingProvider } from "../src";
import {
  bindSelectedTools,
  createLangGraphToolSelector,
} from "../src/adapters/langgraph";

const tools: Array<{
  name: string;
  description: string;
  schema: Record<string, unknown>;
}> = [];
const select = createLangGraphToolSelector(tools, {
  embedder: new HashEmbeddingProvider(),
  k: 8,
});
// In a graph model node: const scopedModel = await bindSelectedTools(model, select, state.messages);
void bindSelectedTools;
void select;
