import { Database } from "bun:sqlite";
import type { IndexedTool, VectorBackend } from "../types";
export class SqliteVectorBackend implements VectorBackend {
  private db: Database;
  constructor(path: string) {
    this.db = new Database(path, { create: true });
    this.db.exec(
      "PRAGMA journal_mode=WAL; CREATE TABLE IF NOT EXISTS tools (id TEXT PRIMARY KEY, tool TEXT NOT NULL, vector TEXT NOT NULL, text TEXT NOT NULL)",
    );
  }
  get size() {
    return Number(
      (this.db.query("SELECT COUNT(*) AS n FROM tools").get() as { n: number })
        .n,
    );
  }
  upsert(records: IndexedTool[]) {
    const q = this.db.query(
      "INSERT INTO tools(id,tool,vector,text) VALUES($id,$tool,$vector,$text) ON CONFLICT(id) DO UPDATE SET tool=excluded.tool,vector=excluded.vector,text=excluded.text",
    );
    this.db.transaction((rows: IndexedTool[]) => {
      for (const row of rows)
        q.run({
          $id: row.tool.id,
          $tool: JSON.stringify({ ...row.tool, original: row.tool.original }),
          $vector: JSON.stringify(row.vector),
          $text: row.text,
        });
    })(records);
  }
  remove(ids: string[]) {
    const q = this.db.query("DELETE FROM tools WHERE id=?");
    this.db.transaction((values: string[]) => {
      for (const id of values) q.run(id);
    })(ids);
  }
  clear() {
    this.db.exec("DELETE FROM tools");
  }
  list(): IndexedTool[] {
    return (
      this.db.query("SELECT tool,vector,text FROM tools").all() as Array<{
        tool: string;
        vector: string;
        text: string;
      }>
    ).map((row) => ({
      tool: JSON.parse(row.tool),
      vector: JSON.parse(row.vector),
      text: row.text,
    }));
  }
  close() {
    this.db.close();
  }
}
