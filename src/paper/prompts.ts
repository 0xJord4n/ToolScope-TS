import type { CanonicalTool } from "../types.js";

function toolView(tool: CanonicalTool): Record<string, unknown> {
  return {
    id: tool.id,
    name: tool.name,
    description: tool.description,
    inputSchema: tool.inputSchema,
    tags: tool.tags,
    ...(tool.namespace === undefined ? {} : { namespace: tool.namespace }),
  };
}

const safety =
  "Return only concise JSON matching the requested shape. Do not include explanations outside JSON. Do not include or produce implementation code, executable functions, handlers, or commands.";

export function buildRelationshipPrompt(target: CanonicalTool, candidate: CanonicalTool): string {
  return [
    "Classify whether these tool descriptors represent substantially the same capability and should be merged.",
    safety,
    'JSON shape: {"similar": boolean, "reason"?: string}. Keep reason concise.',
    `Target: ${JSON.stringify(toolView(target))}`,
    `Candidate: ${JSON.stringify(toolView(candidate))}`,
  ].join("\n");
}

export function buildCorrectionPrompt(cluster: readonly CanonicalTool[]): string {
  return [
    "Validate this proposed merge cluster. Preserve IDs exactly. If it should split, return non-empty ID groups; omitted IDs will remain singletons.",
    safety,
    'JSON shape: {"merge": true, "reason"?: string} or {"merge": false, "clusters": string[][], "reason"?: string}. Keep reason concise.',
    `Cluster: ${JSON.stringify(cluster.map(toolView))}`,
  ].join("\n");
}

export function buildDescriptorSynthesisPrompt(
  representative: CanonicalTool,
  members: readonly CanonicalTool[],
): string {
  return [
    "Synthesize prompt-facing descriptor data for this merge cluster. The name is fixed by the representative and must not be returned.",
    safety,
    'JSON shape: {"description": string, "inputSchema": object, "tags"?: string[], "namespace"?: string}. Keep all text concise.',
    `Representative: ${JSON.stringify(toolView(representative))}`,
    `Members: ${JSON.stringify(members.map(toolView))}`,
  ].join("\n");
}

export function buildQueryDecompositionPrompt(query: string): string {
  return [
    "Decompose the query into the smallest ordered list of independently retrievable tool needs.",
    safety,
    'JSON shape: {"queries": string[]}. Keep each query concise.',
    `Query: ${JSON.stringify(query)}`,
  ].join("\n");
}
