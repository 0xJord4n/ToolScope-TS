export * from "./backends/memory";
export * from "./embeddings";
export * from "./fingerprint";
export * from "./normalize";
export * from "./sinks";
export * from "./text";
export * from "./tool-index";
export * from "./types";

import { ToolIndex } from "./tool-index";
import type { StatelessFilterOptions } from "./types";
export async function index(
  tools: readonly unknown[],
  options: StatelessFilterOptions,
) {
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
