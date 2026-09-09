import { messagesToQueryText, ToolIndex } from "../tool-index.js";
import type { FilterOptions, StatelessFilterOptions } from "../types.js";
export function createLangGraphToolSelector<T>(
  tools: readonly T[],
  options: StatelessFilterOptions,
) {
  let ready: Promise<ToolIndex> | undefined;
  const get = () =>
    (ready ??= (async () => {
      const i = new ToolIndex(options);
      await i.add(tools);
      return i;
    })());
  return async (messages: unknown, overrides: FilterOptions = {}): Promise<T[]> =>
    (await (await get()).filter(messages, { ...options, ...overrides })) as T[];
}
export function createLangGraphSelectionNode<T>(
  tools: readonly T[],
  options: StatelessFilterOptions,
) {
  const select = createLangGraphToolSelector(tools, options);
  return async (state: { messages?: unknown; [key: string]: unknown }) => ({
    selectedTools: await select(state.messages ?? ""),
    toolScopeQuery: messagesToQueryText(state.messages ?? ""),
  });
}
export async function bindSelectedTools<TModel extends { bindTools(tools: unknown[]): unknown }>(
  model: TModel,
  selector: (messages: unknown) => Promise<unknown[]>,
  messages: unknown,
) {
  return model.bindTools(await selector(messages));
}
