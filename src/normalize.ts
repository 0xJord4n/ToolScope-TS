import { toJSONSchema, z } from "zod";
import { fingerprintTool } from "./fingerprint.js";
import { cloneStrictJsonData, invalidJsonData } from "./json.js";
import type { CanonicalTool, JsonSchema } from "./types.js";

const obj = (v: unknown): v is Record<string, unknown> => !!v && typeof v === "object";

const trustedZodVersion = Object.getOwnPropertyDescriptor(
  Object.getOwnPropertyDescriptor(z.string(), "_zod")!.value,
  "version",
)!.value;

const markerFieldTypes = {
  def: "object",
  constr: "function",
  traits: "object",
  bag: "object",
  version: "object",
  deferred: "undefined",
  parse: "function",
  processJSONSchema: "function",
  run: "function",
} as const;

function propertyDescriptor(value: object, key: PropertyKey): PropertyDescriptor | undefined {
  let current: object | null = value;
  while (current !== null) {
    const descriptor = Object.getOwnPropertyDescriptor(current, key);
    if (descriptor) return descriptor;
    current = Object.getPrototypeOf(current);
  }
  return undefined;
}

function sameZodVersion(value: unknown): boolean {
  if (!obj(value) || Object.getPrototypeOf(value) !== Object.prototype) return false;
  for (const key of ["major", "minor", "patch"] as const) {
    const actual = Object.getOwnPropertyDescriptor(value, key);
    const trusted = Object.getOwnPropertyDescriptor(trustedZodVersion, key);
    if (
      !actual ||
      !("value" in actual) ||
      !trusted ||
      !("value" in trusted) ||
      actual.value !== trusted.value
    ) {
      return false;
    }
  }
  return true;
}

function zodConstructor(prototype: object): { value: unknown; name: string } | undefined {
  const constructor = Object.getOwnPropertyDescriptor(prototype, "constructor");
  if (!constructor || !("value" in constructor) || typeof constructor.value !== "function") {
    return undefined;
  }
  const name = Object.getOwnPropertyDescriptor(constructor.value, "name");
  const constructorPrototype = Object.getOwnPropertyDescriptor(constructor.value, "prototype");
  if (
    !name ||
    !("value" in name) ||
    typeof name.value !== "string" ||
    !name.value.startsWith("Zod") ||
    !constructorPrototype ||
    !("value" in constructorPrototype) ||
    constructorPrototype.value !== prototype
  ) {
    return undefined;
  }
  return { value: constructor.value, name: name.value };
}

function recognizedObjectMarkerPrototype(value: object): boolean {
  const descriptors = Object.getOwnPropertyDescriptors(value);
  const propValues = descriptors.propValues;
  return (
    Reflect.ownKeys(descriptors).length === 1 &&
    !!propValues &&
    !("value" in propValues) &&
    !propValues.enumerable &&
    propValues.configurable === true &&
    typeof propValues.get === "function" &&
    typeof propValues.set === "function"
  );
}

function recognizedZodSchema(value: object): boolean {
  const marker = Object.getOwnPropertyDescriptor(value, "_zod");
  if (
    !marker ||
    !("value" in marker) ||
    marker.enumerable ||
    marker.configurable ||
    marker.writable ||
    !obj(marker.value)
  ) {
    return false;
  }
  const prototype = Object.getPrototypeOf(value);
  if (!obj(prototype)) return false;
  const constructor = zodConstructor(prototype);
  if (!constructor) return false;
  const markerPrototype = Object.getPrototypeOf(marker.value);
  if (
    !obj(markerPrototype) ||
    markerPrototype === Object.prototype ||
    (constructor.name === "ZodObject" && !recognizedObjectMarkerPrototype(markerPrototype))
  ) {
    return false;
  }

  // JavaScript has no unforgeable library-instance brand: an attacker can copy
  // descriptors and prototypes from a genuine schema. Fail closed on every
  // observable field we can validate without executing user-controlled code.
  for (const key of Reflect.ownKeys(marker.value)) {
    const descriptor = Object.getOwnPropertyDescriptor(marker.value, key);
    if (descriptor && !("value" in descriptor)) return false;
  }
  for (const [key, type] of Object.entries(markerFieldTypes)) {
    const descriptor = Object.getOwnPropertyDescriptor(marker.value, key);
    if (!descriptor || !("value" in descriptor) || typeof descriptor.value !== type) return false;
  }

  const markerConstructor = Object.getOwnPropertyDescriptor(marker.value, "constr");
  if (
    !markerConstructor ||
    !("value" in markerConstructor) ||
    constructor.value !== markerConstructor.value
  ) {
    return false;
  }

  const traits = Object.getOwnPropertyDescriptor(marker.value, "traits");
  const definition = Object.getOwnPropertyDescriptor(marker.value, "def");
  const version = Object.getOwnPropertyDescriptor(marker.value, "version");
  if (
    !traits ||
    !("value" in traits) ||
    !definition ||
    !("value" in definition) ||
    !obj(definition.value)
  ) {
    return false;
  }
  const publicDefinition = Object.getOwnPropertyDescriptor(value, "def");
  const publicType = Object.getOwnPropertyDescriptor(value, "type");
  const definitionType = Object.getOwnPropertyDescriptor(definition.value, "type");
  if (
    !version ||
    !("value" in version) ||
    !sameZodVersion(version.value) ||
    !publicDefinition ||
    !("value" in publicDefinition) ||
    publicDefinition.value !== definition.value ||
    !publicType ||
    !("value" in publicType) ||
    !definitionType ||
    !("value" in definitionType) ||
    publicType.value !== definitionType.value
  ) {
    return false;
  }

  for (const key of Reflect.ownKeys(definition.value)) {
    const descriptor = Object.getOwnPropertyDescriptor(definition.value, key);
    if (!descriptor || "value" in descriptor) continue;
    // ZodObject lazily materializes only `shape` with this descriptor. Its
    // closure identity is per-schema, so no stable getter identity exists to
    // compare; all other nested definition accessors are rejected.
    if (
      constructor.name !== "ZodObject" ||
      key !== "shape" ||
      !descriptor.enumerable ||
      !descriptor.configurable ||
      typeof descriptor.get !== "function" ||
      descriptor.set !== undefined
    ) {
      return false;
    }
  }

  try {
    return (
      Object.getPrototypeOf(traits.value) === Set.prototype &&
      Set.prototype.has.call(traits.value, "ZodType") &&
      Set.prototype.has.call(traits.value, "$ZodType")
    );
  } catch {
    return false;
  }
}

function rejectAdapterAccessors(value: object, recognizedZod: boolean): void {
  const zodPrototype = recognizedZod ? Object.getPrototypeOf(value) : undefined;
  let current: object | null = value;
  while (current !== null) {
    for (const key of ["toJSON", "_zod", "_def"] as const) {
      const descriptor = Object.getOwnPropertyDescriptor(current, key);
      if (
        descriptor &&
        !("value" in descriptor) &&
        !(recognizedZod && key === "_def" && current === zodPrototype)
      ) {
        throw invalidJsonData("tool input schema", `accessor property ${key} is forbidden`);
      }
    }
    current = Object.getPrototypeOf(current);
  }
}

function withoutZodMetadata(value: unknown): unknown {
  if (!obj(value)) return value;
  const descriptors = Object.getOwnPropertyDescriptors(value);
  delete descriptors["~standard"];
  return Object.defineProperties(Object.create(Object.getPrototypeOf(value)), descriptors);
}

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
const schemaAdapterFields = ["inputSchema", "parameters", "schema", "argsSchema"] as const;

function schemaOf(value: Record<string, unknown>): JsonSchema {
  const adapters: Array<{ field: (typeof schemaAdapterFields)[number]; value: unknown }> = [];
  for (const field of schemaAdapterFields) {
    const descriptor = propertyDescriptor(value, field);
    if (!descriptor) continue;
    if (!("value" in descriptor)) {
      throw invalidJsonData("tool schema adapters", `accessor property ${field} is forbidden`);
    }
    if (descriptor.value !== null && descriptor.value !== undefined) {
      adapters.push({ field, value: descriptor.value });
    }
  }
  if (adapters.length > 1) {
    throw new TypeError(
      `Unsupported tool: ambiguous schema adapters (${adapters.map(({ field }) => field).join(", ")})`,
    );
  }
  const raw = adapters[0]?.value;
  if (obj(raw)) {
    const recognizedZod = recognizedZodSchema(raw);
    rejectAdapterAccessors(raw, recognizedZod);
    if (recognizedZod) {
      return withoutZodMetadata(toJSONSchema(raw as never)) as JsonSchema;
    }
    const toJSON = propertyDescriptor(raw, "toJSON");
    if (toJSON && "value" in toJSON && typeof toJSON.value === "function") {
      return toJSON.value.call(raw) as JsonSchema;
    }
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
  const inputSchema = cloneStrictJsonData(schemaOf(body), "tool input schema") as JsonSchema;
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
