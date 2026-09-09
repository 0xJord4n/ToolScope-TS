export * from "./backends/memory.js";
export * from "./embeddings.js";
export * from "./fingerprint.js";
export * from "./normalize.js";
export * from "./sinks.js";
export * from "./text.js";
export * from "./tool-index.js";
export * from "./types.js";

import { ToolIndex } from "./tool-index.js";
import type { StatelessFilterOptions } from "./types.js";
export async function index(tools: readonly unknown[], options: StatelessFilterOptions) {
  const idx = new ToolIndex(options);
  await idx.add(tools);
  return idx;
}
export async function filter(
  messages: unknown,
  tools: readonly unknown[],
  options: StatelessFilterOptions,
) {
  return (await index(tools, options)).filter(messages, options);
}
export async function filterWithTrace(
  messages: unknown,
  tools: readonly unknown[],
  options: StatelessFilterOptions,
) {
  return (await index(tools, options)).filterWithTrace(messages, options);
}
