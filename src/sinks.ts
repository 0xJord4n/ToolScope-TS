import { appendFile } from "node:fs/promises";
import type { SelectionTrace, TraceSink } from "./types.js";
export type TraceTransform = (trace: SelectionTrace) => unknown;
export class JsonlTraceSink implements TraceSink {
  constructor(
    private path: string,
    private transform: TraceTransform = (trace) => trace,
  ) {}
  async emit(trace: SelectionTrace) {
    await appendFile(this.path, `${JSON.stringify(this.transform(trace))}\n`, "utf8");
  }
}
export class ConsoleTraceSink implements TraceSink {
  constructor(private log: (line: string) => void = console.log) {}
  emit(trace: SelectionTrace) {
    this.log(JSON.stringify(trace));
  }
}
export class CallbackTraceSink implements TraceSink {
  constructor(private callback: (trace: SelectionTrace) => void | Promise<void>) {}
  emit(trace: SelectionTrace) {
    return this.callback(trace);
  }
}
export function redactTrace(options: { query?: boolean; toolNames?: boolean } = {}) {
  return (trace: SelectionTrace) => ({
    ...trace,
    query: options.query === false ? trace.query : "[REDACTED]",
    selected: trace.selected.map((score) => ({
      ...score,
      name: options.toolNames ? "[REDACTED]" : score.name,
    })),
    scores: trace.scores.map((score) => ({
      ...score,
      name: options.toolNames ? "[REDACTED]" : score.name,
    })),
  });
}
