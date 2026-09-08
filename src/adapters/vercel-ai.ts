import { ToolIndex } from "../tool-index";
import type { StatelessFilterOptions } from "../types";
export type VercelToolSet = Record<string, unknown>;
const objectValue = (value: unknown): Record<string, unknown> =>
  value && typeof value === "object" ? (value as Record<string, unknown>) : {};
const entriesOf = <T extends VercelToolSet>(tools: T) =>
  Object.entries(tools).map(([name, value]) => {
    const tool = objectValue(value);
    return {
      ...tool,
      name,
      description:
        typeof tool.description === "string"
          ? tool.description
          : typeof tool.title === "string"
            ? tool.title
            : "",
      inputSchema: tool.inputSchema ?? tool.parameters ?? {},
      originalName: name,
    };
  });
export async function selectVercelTools<T extends VercelToolSet>(
  messages: unknown,
  tools: T,
  options: StatelessFilterOptions,
): Promise<Partial<T>> {
  const entries = entriesOf(tools);
  const idx = new ToolIndex(options);
  await idx.add(entries);
  const selected = (await idx.filter(messages, options)) as Array<{
    originalName: keyof T & string;
  }>;
  return Object.fromEntries(
    selected.map((x) => [x.originalName, tools[x.originalName]]),
  ) as Partial<T>;
}
export function createVercelToolSelector<T extends VercelToolSet>(
  tools: T,
  options: StatelessFilterOptions,
) {
  const entries = entriesOf(tools);
  let ready: Promise<ToolIndex> | undefined;
  const getIndex = () =>
    (ready ??= (async () => {
      const idx = new ToolIndex(options);
      await idx.add(entries);
      return idx;
    })());
  return async (
    messages: unknown,
    overrides: Partial<StatelessFilterOptions> = {},
  ): Promise<Partial<T>> => {
    const selected = (await (
      await getIndex()
    ).filter(messages, { ...options, ...overrides })) as Array<{
      originalName: keyof T & string;
    }>;
    return Object.fromEntries(
      selected.map((entry) => [entry.originalName, tools[entry.originalName]]),
    ) as Partial<T>;
  };
}
export async function selectVercelActiveTools<T extends VercelToolSet>(
  messages: unknown,
  tools: T,
  options: StatelessFilterOptions,
): Promise<Array<keyof T & string>> {
  return Object.keys(
    await selectVercelTools(messages, tools, options),
  ) as Array<keyof T & string>;
}
export function createVercelPrepareStep<T extends VercelToolSet>(
  tools: T,
  options: StatelessFilterOptions,
) {
  const select = createVercelToolSelector(tools, options);
  return async ({
    messages,
  }: {
    messages: unknown;
  }): Promise<{ activeTools: Array<keyof T & string> }> => ({
    activeTools: Object.keys(await select(messages)) as Array<keyof T & string>,
  });
}
