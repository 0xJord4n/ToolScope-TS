import { ToolIndex } from "../tool-index";
import type { FilterOptions, StatelessFilterOptions } from "../types";
export interface McpLikeClient {
  listTools(
    ...args: unknown[]
  ): Promise<{ tools: unknown[]; [key: string]: unknown }>;
  callTool(request: unknown, ...args: unknown[]): Promise<unknown>;
  onToolsChanged?(handler: () => void): void;
}
export class ToolScopeMcpClient {
  private index?: ToolIndex;
  private generation = 1;
  private indexedGeneration = 0;
  private refreshPromise?: Promise<void>;
  constructor(
    private client: McpLikeClient,
    private options: StatelessFilterOptions,
  ) {
    client.onToolsChanged?.(() => this.markDirty());
  }
  markDirty() {
    this.generation++;
  }
  /** Register this with the application's MCP notification dispatcher. */
  readonly toolsChangedHandler = () => this.markDirty();
  private async refresh() {
    if (this.index && this.indexedGeneration === this.generation) return;
    if (this.refreshPromise) return this.refreshPromise;
    this.refreshPromise = (async () => {
      while (!this.index || this.indexedGeneration !== this.generation) {
        const targetGeneration = this.generation;
        const tools: unknown[] = [];
        let cursor: string | undefined;
        do {
          const response = cursor
            ? await this.client.listTools({ cursor })
            : await this.client.listTools();
          tools.push(...response.tools);
          cursor =
            typeof response.nextCursor === "string"
              ? response.nextCursor
              : undefined;
        } while (cursor);
        this.index ??= new ToolIndex(this.options);
        await this.index.sync(tools);
        this.indexedGeneration = targetGeneration;
      }
    })().finally(() => {
      this.refreshPromise = undefined;
    });
    return this.refreshPromise;
  }
  async listToolsFor(messages: unknown, options: FilterOptions = {}) {
    await this.refresh();
    return {
      tools: await this.index!.filter(messages, {
        ...this.options,
        ...options,
      }),
    };
  }
  async listTools(...args: unknown[]) {
    return this.client.listTools(...args);
  }
  async callTool(request: unknown, ...args: unknown[]) {
    return this.client.callTool(request, ...args);
  }
}
