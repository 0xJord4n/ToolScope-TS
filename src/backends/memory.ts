import type { IndexedTool, VectorBackend } from "../types";
export class MemoryVectorBackend implements VectorBackend {
  protected records = new Map<string, IndexedTool>();
  get size() {
    return this.records.size;
  }
  upsert(records: IndexedTool[]) {
    for (const record of records) this.records.set(record.tool.id, record);
  }
  remove(ids: string[]) {
    for (const id of ids) this.records.delete(id);
  }
  clear() {
    this.records.clear();
  }
  list() {
    return [...this.records.values()];
  }
}
