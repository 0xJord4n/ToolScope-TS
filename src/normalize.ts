import { toJSONSchema } from "zod";
import { fingerprintTool } from "./fingerprint.js";
import type { CanonicalTool, JsonSchema } from "./types.js";

const obj = (v: unknown): v is Record<string, unknown> => !!v && typeof v === "object";
function tagsOf(value: Record<string, unknown>): string[] {
  const meta = obj(value.metadata) ? value.metadata : {};
  const annotations = obj(value.annotations) ? value.annotations : {};
  return [
    ...(Array.isArray(value.toolscope_tags) ? value.toolscope_tags : []),
    ...(Array.isArray(value.tags) ? value.tags : []),
    ...(Array.isArray(meta.tags) ? meta.tags : []),
    ...(Array.isArray(annotations.tags) ? annotations.tags : []),
  ].map(String);
}
function schemaOf(value: Record<string, unknown>): JsonSchema {
  const raw = value.inputSchema ?? value.parameters ?? value.schema ?? value.argsSchema;
  if (obj(raw)) {
    if ("_zod" in raw || "_def" in raw) {
      try {
        return toJSONSchema(raw as never) as JsonSchema;
      } catch {
        return {};
      }
    }
    if (typeof (raw as { toJSON?: unknown }).toJSON === "function")
      return (raw as { toJSON(): JsonSchema }).toJSON();
    return raw;
  }
  return {};
}
export function normalizeTool<T>(input: T, forcedName?: string): CanonicalTool<T> {
  if (!obj(input)) throw new TypeError("Unsupported tool: expected an object");
  const original = input;
  let body: Record<string, unknown> = input;
  const outer: Record<string, unknown> = input;
  if (input.type === "function" && obj(input.function)) body = input.function;
  const name = forcedName ?? String(body.name ?? input.name ?? "");
  if (!name) throw new TypeError("Unsupported tool: missing name");
  const description = String(body.description ?? input.description ?? "");
  const inputSchema = schemaOf(body);
  const tags = [...new Set([...tagsOf(outer), ...tagsOf(body)])];
  const namespace = typeof input.namespace === "string" ? input.namespace : undefined;
  return {
    id: fingerprintTool(name, description, inputSchema, namespace),
    name,
    description,
    inputSchema,
    tags,
    namespace,
    original,
  };
}
export function normalizeTools<T>(tools: readonly T[]): CanonicalTool<T>[] {
  return tools.map((t) => normalizeTool(t));
}
