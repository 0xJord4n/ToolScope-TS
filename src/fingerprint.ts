function canonical(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonical);
  if (value && typeof value === "object")
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>)
        .sort(([a], [b]) => a.localeCompare(b))
        .map(([k, v]) => [k, canonical(v)]),
    );
  return value;
}
export function stableJson(value: unknown): string {
  return JSON.stringify(canonical(value));
}
export function fingerprintTool(
  name: string,
  description: string,
  inputSchema: unknown,
  namespace = "",
): string {
  const input = stableJson({ name, description, inputSchema, namespace });
  return createHash("sha256").update(input).digest("hex");
}

import { createHash } from "node:crypto";
