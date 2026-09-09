import { ToolIndex, type EmbeddingProvider, type Reranker } from "../src";
import {
  MultiQueryRetriever,
  ToolMerger,
  type ClusterValidator,
  type DescriptorSynthesizer,
  type MergedToolDescriptor,
  type QueryDecomposer,
  type RelationshipClassifier,
} from "../src/paper";

const terms = ["weather", "forecast", "calendar", "event"];
const embedder: EmbeddingProvider = {
  async embed(texts) {
    return texts.map((text) => {
      const normalized = text.toLowerCase();
      const vector = terms.map((term) => Number(normalized.includes(term)));
      return vector.some(Boolean) ? vector : [0, 0, 0, 1];
    });
  },
};

const classifier: RelationshipClassifier = {
  async classify(left, right) {
    return {
      similar: left.name.startsWith("weather_") && right.name.startsWith("weather_"),
    };
  },
};
const validator: ClusterValidator = {
  async validate() {
    return { merge: true };
  },
};
const synthesizer: DescriptorSynthesizer = {
  async synthesize() {
    return {
      description: "Get current weather or a forecast for a location",
    };
  },
};

const tools = [
  {
    name: "weather_now",
    description: "Get current weather for a city",
    inputSchema: {
      type: "object",
      properties: { city: { type: "string" } },
      required: ["city"],
    },
    execute: async ({ city }: { city: string }) => `Current weather in ${city}`,
  },
  {
    name: "weather_forecast",
    description: "Get a multi-day weather forecast",
    inputSchema: {
      type: "object",
      properties: { city: { type: "string" }, days: { type: "number" } },
      required: ["city", "days"],
    },
    execute: async ({ city, days }: { city: string; days: number }) =>
      `${days}-day forecast for ${city}`,
  },
  {
    name: "calendar_create_event",
    description: "Create a calendar event",
    inputSchema: {
      type: "object",
      properties: { title: { type: "string" } },
      required: ["title"],
    },
    execute: async ({ title }: { title: string }) => `Created ${title}`,
  },
];

const merger = new ToolMerger({
  embedder,
  classifier,
  validator,
  synthesizer,
  autoCorrectionPasses: 1,
  similarityThreshold: 0.5,
});
const merged = await merger.merge(tools);

const index = new ToolIndex({ embedder });
await index.add(merged.merged);

const decomposer: QueryDecomposer = {
  async decompose() {
    return ["weather forecast", "create calendar event"];
  },
};
const reranker: Reranker = {
  async rerank(query, candidates) {
    const words = query.toLowerCase().split(/\W+/);
    return candidates.map((candidate) => {
      const text = `${candidate.name} ${candidate.description}`.toLowerCase();
      return words.filter((word) => text.includes(word)).length;
    });
  },
};
const retriever = new MultiQueryRetriever<MergedToolDescriptor>({
  index,
  decomposer,
  reranker,
  k: 2,
});
const result = await retriever.retrieve("Check the forecast and schedule an event");

for (const selected of result.tools) {
  const originals = merged.manifest.resolve(selected.id);
  console.log(
    selected.name,
    "=>",
    originals.map((tool) => tool.name),
  );
}

// The manifest is a dispatch aid, not authorization. Authorize the chosen
// original tool again immediately before execution.
