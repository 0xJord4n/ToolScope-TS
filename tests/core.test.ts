import { describe, expect, test } from "bun:test";
import { z } from "zod";
import { filter, HashEmbeddingProvider, normalizeTool, ToolIndex } from "../src/index";
import { buildRelationshipPrompt } from "../src/paper/prompts.js";

const tools = [
  {
    name: "jira_create_issue",
    description: "Create a Jira ticket",
    inputSchema: {},
    tags: ["jira", "write"],
  },
  {
    name: "confluence_search",
    description: "Search documentation pages",
    inputSchema: {},
    tags: ["docs", "read"],
  },
  {
    type: "function",
    function: {
      name: "send_email",
      description: "Send an email message",
      parameters: { type: "object" },
    },
    tags: ["mail", "dangerous"],
  },
];
describe("normalization and filtering", () => {
  test("normalizes MCP and OpenAI tools while preserving originals", () => {
    expect(normalizeTool(tools[0]!).name).toBe("jira_create_issue");
    expect(normalizeTool(tools[2]!).name).toBe("send_email");
    expect(normalizeTool(tools[2]!).original).toBe(tools[2]!);
  });
  test("extracts ToolScope and MCP annotation tags", () => {
    expect(
      normalizeTool({
        name: "delete",
        description: "Delete",
        inputSchema: {},
        toolscope_tags: ["destructive"],
        annotations: { tags: ["admin"] },
      }).tags,
    ).toEqual(["destructive", "admin"]);
  });
  test.each(["toJSON", "_zod", "_def"] as const)(
    "rejects an own %s accessor without invoking it",
    (property) => {
      let getterCalls = 0;
      const inputSchema = {};
      Object.defineProperty(inputSchema, property, {
        enumerable: true,
        get() {
          getterCalls += 1;
          throw new Error(`${property} getter must not run`);
        },
      });

      expect(() => normalizeTool({ name: `own-${property}`, inputSchema })).toThrow(
        /strict JSON data/,
      );
      expect(getterCalls).toBe(0);
    },
  );
  test.each(["toJSON", "_zod", "_def"] as const)(
    "rejects an inherited %s accessor without invoking it",
    (property) => {
      let getterCalls = 0;
      const prototype = {};
      Object.defineProperty(prototype, property, {
        enumerable: true,
        get() {
          getterCalls += 1;
          throw new Error(`${property} getter must not run`);
        },
      });
      const inputSchema = Object.create(prototype) as Record<string, unknown>;

      expect(() => normalizeTool({ name: `inherited-${property}`, inputSchema })).toThrow(
        /strict JSON data/,
      );
      expect(getterCalls).toBe(0);
    },
  );
  test.each(["_zod", "_def"] as const)(
    "preserves a JSON data property named %s without treating it as Zod",
    (property) => {
      const inputSchema = { type: "object", [property]: "extension" };

      expect(normalizeTool({ name: `data-${property}`, inputSchema }).inputSchema).toEqual(
        inputSchema,
      );
    },
  );
  test("converts a data-property toJSON method", () => {
    const inputSchema = {
      toJSON() {
        return { type: "object", properties: { value: { type: "string" } } };
      },
    };

    expect(normalizeTool({ name: "to-json", inputSchema }).inputSchema).toEqual({
      type: "object",
      properties: { value: { type: "string" } },
    });
  });
  test("deep-detaches the canonical schema from a direct source schema", () => {
    const inputSchema = {
      type: "object",
      properties: { query: { type: "string", examples: ["original"] } },
    };

    const canonical = normalizeTool({ name: "detached-direct", inputSchema });
    inputSchema.properties.query.examples[0] = "mutated";

    expect(canonical.inputSchema).toEqual({
      type: "object",
      properties: { query: { type: "string", examples: ["original"] } },
    });
    expect(canonical.inputSchema).not.toBe(inputSchema);
    expect(canonical.inputSchema.properties).not.toBe(inputSchema.properties);
  });
  test("deep-detaches a toJSON adapter result while preserving __proto__ data", () => {
    const properties: Record<string, unknown> = {};
    Object.defineProperty(properties, "__proto__", {
      value: { type: "string" },
      enumerable: true,
      configurable: true,
      writable: true,
    });
    const adapted = { type: "object", properties };
    const canonical = normalizeTool({
      name: "detached-adapter",
      inputSchema: { toJSON: () => adapted },
    });

    properties.__proto__ = { type: "number" };

    const canonicalProperties = canonical.inputSchema.properties as Record<string, unknown>;
    expect(Object.hasOwn(canonicalProperties, "__proto__")).toBe(true);
    expect(canonicalProperties.__proto__).toEqual({ type: "string" });
    expect(canonical.inputSchema).not.toBe(adapted);
  });
  test("prompt builders cannot reach accessors added to a source schema after normalization", () => {
    let getterCalls = 0;
    const sourceProperties = { query: { type: "string" } };
    const canonical = normalizeTool({
      name: "post-normalization-accessor",
      inputSchema: { type: "object", properties: sourceProperties },
    });
    Object.defineProperty(sourceProperties, "query", {
      enumerable: true,
      get() {
        getterCalls += 1;
        throw new Error("post-normalization getter must not run");
      },
    });

    expect(() => buildRelationshipPrompt(canonical, canonical)).not.toThrow();
    expect(getterCalls).toBe(0);
  });
  test.each([
    ["own", "_zod"],
    ["own", "_def"],
    ["inherited", "_zod"],
    ["inherited", "_def"],
  ] as const)(
    "rejects an %s %s accessor before using a data-property toJSON method",
    (placement, property) => {
      let getterCalls = 0;
      const prototype = {};
      const inputSchema = Object.create(placement === "inherited" ? prototype : Object.prototype, {
        toJSON: {
          value: () => ({ type: "object" }),
          enumerable: true,
          configurable: true,
          writable: true,
        },
      }) as Record<string, unknown>;
      Object.defineProperty(placement === "own" ? inputSchema : prototype, property, {
        get() {
          getterCalls += 1;
          throw new Error(`${property} getter must not run`);
        },
      });

      expect(() =>
        normalizeTool({ name: `${placement}-${property}-with-to-json`, inputSchema }),
      ).toThrow(/strict JSON data/);
      expect(getterCalls).toBe(0);
    },
  );
  test.each(["own", "inherited"] as const)(
    "rejects an %s toJSON accessor before using a recognized Zod marker",
    (placement) => {
      let getterCalls = 0;
      const inputSchema = z.object({ value: z.string() });
      const target =
        placement === "own"
          ? inputSchema
          : Object.setPrototypeOf(inputSchema, Object.create(Object.getPrototypeOf(inputSchema)));
      Object.defineProperty(
        placement === "own" ? target : Object.getPrototypeOf(target),
        "toJSON",
        {
          get() {
            getterCalls += 1;
            throw new Error("toJSON getter must not run");
          },
        },
      );

      expect(() => normalizeTool({ name: `${placement}-to-json-with-zod`, inputSchema })).toThrow(
        /strict JSON data/,
      );
      expect(getterCalls).toBe(0);
    },
  );
  test.each(["own", "inherited"] as const)(
    "rejects an %s toJSON accessor on a spoofed recognized-looking Zod marker",
    (placement) => {
      let getterCalls = 0;
      const prototype = {};
      const inputSchema = Object.create(placement === "inherited" ? prototype : Object.prototype);
      Object.defineProperty(inputSchema, "_zod", {
        value: {
          constr: z.ZodObject,
          def: {},
          traits: new Set(["ZodType"]),
        },
      });
      Object.defineProperty(placement === "own" ? inputSchema : prototype, "toJSON", {
        get() {
          getterCalls += 1;
          throw new Error("toJSON getter must not run");
        },
      });

      expect(() => normalizeTool({ name: `spoofed-${placement}-to-json`, inputSchema })).toThrow(
        /strict JSON data/,
      );
      expect(getterCalls).toBe(0);
    },
  );
  test("rejects an accessor on a recognized-looking Zod marker without invoking it", () => {
    let getterCalls = 0;
    const marker = {
      constr: z.ZodObject,
      def: { type: "object", shape: {} },
      traits: new Set(["ZodType"]),
    };
    Object.defineProperty(marker, "processJSONSchema", {
      get() {
        getterCalls += 1;
        throw new Error("processJSONSchema getter must not run");
      },
    });
    const inputSchema = Object.create(z.ZodObject.prototype);
    Object.defineProperty(inputSchema, "_zod", { value: marker });

    expect(() => normalizeTool({ name: "zod-marker-accessor-spoof", inputSchema })).toThrow(
      /strict JSON data/,
    );
    expect(getterCalls).toBe(0);
  });
  test("rejects a nested Zod definition accessor spoof without invoking it", () => {
    let getterCalls = 0;
    const definition = {};
    Object.defineProperty(definition, "type", {
      enumerable: true,
      get() {
        getterCalls += 1;
        return "object";
      },
    });
    const inputSchema = Object.create(z.ZodObject.prototype);
    Object.defineProperty(inputSchema, "_zod", {
      value: {
        constr: z.ZodObject,
        traits: new Set(["ZodType"]),
        def: definition,
      },
    });

    expect(() =>
      normalizeTool({ name: "nested-zod-definition-accessor-spoof", inputSchema }),
    ).toThrow(/strict JSON data/);
    expect(getterCalls).toBe(0);
  });
  test("normalizes genuine installed Zod schema kinds", () => {
    const schemas = [
      ["object", z.object({ value: z.string() }), "object"],
      ["string", z.string(), "string"],
      ["array", z.array(z.string()), "array"],
      ["union", z.union([z.string(), z.number()]), ["string", "number"]],
    ] as const;

    for (const [name, inputSchema, type] of schemas) {
      const normalized = normalizeTool({ name: `zod-${name}`, inputSchema }).inputSchema;
      expect(normalized).toHaveProperty("$schema");
      expect(normalized).toHaveProperty("type", type);
    }
  });
  test("does not recognize a spoof using a non-schema Zod export prototype", () => {
    let getterCalls = 0;
    const definition = {};
    Object.defineProperty(definition, "type", {
      get() {
        getterCalls += 1;
        throw new Error("spoofed Zod definition getter must not run");
      },
    });
    const inputSchema = Object.create(z.string.prototype);
    Object.defineProperty(inputSchema, "_zod", {
      value: {
        constr: z.string,
        def: definition,
        traits: new Set(["ZodType"]),
      },
    });

    expect(() => normalizeTool({ name: "non-schema-zod-export-spoof", inputSchema })).toThrow(
      /strict JSON data/,
    );
    expect(getterCalls).toBe(0);
  });
  test("strictly validates the schema returned by a toJSON method", () => {
    let getterCalls = 0;
    const returnedSchema = {};
    Object.defineProperty(returnedSchema, "type", {
      enumerable: true,
      get() {
        getterCalls += 1;
        return "object";
      },
    });

    expect(() =>
      normalizeTool({
        name: "unsafe-to-json-result",
        inputSchema: { toJSON: () => returnedSchema },
      }),
    ).toThrow(/strict JSON data/);
    expect(getterCalls).toBe(0);
  });
  test.each([
    ["inputSchema", "parameters"],
    ["inputSchema", "schema"],
    ["inputSchema", "argsSchema"],
    ["parameters", "schema"],
    ["parameters", "argsSchema"],
    ["schema", "argsSchema"],
  ] as const)("rejects ambiguous non-null schema adapters %s and %s", (first, second) => {
    let adapterCalls = 0;
    const selected = {
      toJSON() {
        adapterCalls += 1;
        return { type: "object" };
      },
    };

    expect(() =>
      normalizeTool({ name: `ambiguous-${first}-${second}`, [first]: selected, [second]: {} }),
    ).toThrow(/ambiguous schema adapters/);
    expect(adapterCalls).toBe(0);
  });
  test.each(["inputSchema", "parameters", "schema", "argsSchema"] as const)(
    "accepts exactly one non-null schema adapter: %s",
    (field) => {
      expect(
        normalizeTool({
          name: `single-${field}`,
          inputSchema: null,
          parameters: undefined,
          schema: null,
          argsSchema: undefined,
          [field]: { type: "object" },
        }).inputSchema,
      ).toEqual({ type: "object" });
    },
  );
  test.each([
    {},
    { inputSchema: null, parameters: undefined, schema: null, argsSchema: undefined },
  ])("uses an empty schema when all schema adapters are absent or nullish", (fields) => {
    expect(normalizeTool({ name: "no-schema", ...fields }).inputSchema).toEqual({});
  });
  test("accepts exactly one inherited non-null schema adapter", () => {
    const tool = Object.assign(Object.create({ parameters: { type: "object" } }), {
      name: "inherited-single-schema",
    });

    expect(normalizeTool(tool).inputSchema).toEqual({ type: "object" });
  });
  test("rejects ambiguity across own and inherited schema adapters before adaptation", () => {
    let adapterCalls = 0;
    const tool = Object.assign(
      Object.create({
        parameters: {
          toJSON() {
            adapterCalls += 1;
            return { type: "object" };
          },
        },
      }),
      { name: "inherited-ambiguous-schema", inputSchema: {} },
    );

    expect(() => normalizeTool(tool)).toThrow(/ambiguous schema adapters/);
    expect(adapterCalls).toBe(0);
  });
  test.each(["own", "inherited"] as const)(
    "rejects an %s schema-adapter accessor without invoking it",
    (placement) => {
      let getterCalls = 0;
      const prototype = {};
      const tool = Object.assign(
        Object.create(placement === "inherited" ? prototype : Object.prototype),
        { name: `${placement}-schema-adapter-accessor` },
      );
      Object.defineProperty(placement === "own" ? tool : prototype, "inputSchema", {
        get() {
          getterCalls += 1;
          throw new Error("schema adapter getter must not run");
        },
      });

      expect(() => normalizeTool(tool)).toThrow(/strict JSON data/);
      expect(getterCalls).toBe(0);
    },
  );
  test("rejects enumerable Array.prototype pollution without reading it", () => {
    let getterCalls = 0;
    const arrayPrototype = Array.prototype as unknown as Record<string, unknown>;
    Object.defineProperty(arrayPrototype, "pollutedSchemaState", {
      configurable: true,
      enumerable: true,
      get() {
        getterCalls += 1;
        throw new Error("polluted array value must not be read");
      },
    });
    try {
      expect(() =>
        normalizeTool({
          name: "array-prototype-pollution",
          inputSchema: { type: "array", examples: ["safe"] },
        }),
      ).toThrow(/inherited enumerable state/);
      expect(getterCalls).toBe(0);
    } finally {
      delete arrayPrototype.pollutedSchemaState;
    }
  });
  test("rejects enumerable custom array prototype state without reading it", () => {
    let getterCalls = 0;
    const prototype = Object.create(Array.prototype);
    Object.defineProperty(prototype, "inheritedSchemaState", {
      enumerable: true,
      get() {
        getterCalls += 1;
        throw new Error("inherited array value must not be read");
      },
    });
    const values = ["safe"];
    Object.setPrototypeOf(values, prototype);

    expect(() =>
      normalizeTool({
        name: "custom-array-prototype",
        inputSchema: { type: "array", examples: values },
      }),
    ).toThrow(/inherited enumerable state/);
    expect(getterCalls).toBe(0);
  });
  test("preserves normal dense arrays", () => {
    expect(
      normalizeTool({
        name: "dense-array",
        inputSchema: { type: "array", examples: ["first", 2, null] },
      }).inputSchema,
    ).toEqual({ type: "array", examples: ["first", 2, null] });
  });
  test("retrieves relevant tools and applies allow/deny tags", async () => {
    const index = new ToolIndex({ embedder: new HashEmbeddingProvider(256) });
    await index.add(tools);
    const result = await index.filter("file a jira bug ticket", {
      k: 2,
      denyTags: ["dangerous"],
    });
    expect(result[0]).toBe(tools[0]);
    expect(result).not.toContain(tools[2]);
    expect(await index.filter("search", { k: 5, allowTags: ["docs"] })).toEqual([tools[1]]);
  });
  test("stateless filter has parity with indexed filtering", async () => {
    expect(
      await filter("search confluence documentation", tools, {
        embedder: new HashEmbeddingProvider(256),
        k: 1,
      }),
    ).toEqual([tools[1]]);
  });
  test("updates changed tools and removes deleted tools", async () => {
    const index = new ToolIndex({ embedder: new HashEmbeddingProvider(128) });
    await index.add(tools);
    expect(index.size).toBe(3);
    await index.sync([tools[0]!]);
    expect(index.size).toBe(1);
    expect(await index.filter("email", { k: 5 })).toEqual([]);
  });
  test("syncs an empty catalog without calling the embedder", async () => {
    const index = new ToolIndex({
      embedder: {
        async embed(texts) {
          if (texts.length === 0) throw new Error("empty batch unsupported");
          return texts.map(() => [1]);
        },
      },
    });
    await index.add([{ name: "temporary", description: "Temporary", inputSchema: {} }]);
    await expect(index.sync([])).resolves.toBe(index);
    expect(index.size).toBe(0);
  });
  test("rejects invalid embedding dimensions and vectors", async () => {
    expect(() => new HashEmbeddingProvider(0)).toThrow();
    const index = new ToolIndex({
      embedder: {
        async embed() {
          return [[1, Number.NaN]];
        },
      },
    });
    expect(index.add([{ name: "bad", description: "Bad", inputSchema: {} }])).rejects.toThrow();
  });
});
