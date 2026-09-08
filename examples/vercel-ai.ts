import { type PrepareStepFunction, tool } from "ai";
import { z } from "zod";
import { HashEmbeddingProvider } from "../src";
import { createVercelPrepareStep } from "../src/adapters/vercel-ai";

const tools = {
  weather: tool({
    description: "Get weather",
    inputSchema: z.object({ city: z.string() }),
    execute: async ({ city }) => `${city}: sunny`,
  }),
  stocks: tool({
    description: "Get a stock quote",
    inputSchema: z.object({ symbol: z.string() }),
    execute: async ({ symbol }) => `${symbol}: 42`,
  }),
};
const prepareStep: PrepareStepFunction<typeof tools> = createVercelPrepareStep(
  tools,
  { embedder: new HashEmbeddingProvider(), k: 1 },
);
// await generateText({ model, tools, prompt: "Weather in Paris?", prepareStep });
void prepareStep;
