import { stableJson } from "./fingerprint";
import type { CanonicalTool, ToolTextConfig } from "./types";
export const lowercase = () => (s: string) => s.toLowerCase();
export const collapseWhitespace = () => (s: string) =>
  s.replace(/\s+/g, " ").trim();
export const normalizeUnicode =
  (form: "NFC" | "NFD" | "NFKC" | "NFKD" = "NFKC") =>
  (s: string) =>
    s.normalize(form);
export const stripControlChars = () => (s: string) =>
  s.replace(/[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]/g, "");
export function toolText(
  tool: CanonicalTool,
  config: ToolTextConfig = {},
): string {
  const {
    useName = true,
    useDescription = true,
    useSchema = false,
    useTags = false,
    truncate = 256,
    preprocessors = [],
  } = config;
  const parts: string[] = [];
  if (useName) parts.push(tool.name);
  if (useDescription) parts.push(tool.description);
  if (useTags && tool.tags.length) parts.push(tool.tags.join(" "));
  if (useSchema) parts.push(stableJson(tool.inputSchema));
  let out = parts.filter(Boolean).join("\n");
  for (const fn of preprocessors) out = fn(out);
  return truncate == null ? out : out.slice(0, Math.max(0, truncate));
}
