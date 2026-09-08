import type { EmbeddingProvider } from "./types";

const tokens = (s: string) =>
  s
    .toLowerCase()
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .split(/[^a-z0-9]+/)
    .filter(Boolean)
    .flatMap((x) => {
      const stem = x.replace(/(ing|ed|es|s)$/, "");
      return stem && stem !== x ? [x, stem] : [x];
    });
function hash(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++)
    h = Math.imul(h ^ s.charCodeAt(i), 16777619);
  return h >>> 0;
}
export class HashEmbeddingProvider implements EmbeddingProvider {
  constructor(public readonly dimensions = 384) {
    if (!Number.isInteger(dimensions) || dimensions <= 0) {
      throw new RangeError("Embedding dimensions must be a positive integer");
    }
  }
  async embed(texts: string[]): Promise<number[][]> {
    return texts.map((text) => {
      const v = Array<number>(this.dimensions).fill(0);
      for (const token of tokens(text)) {
        const h = hash(token);
        v[h % this.dimensions]! += 1;
        v[(h >>> 8) % this.dimensions]! += 0.25;
      }
      const n = Math.hypot(...v) || 1;
      return v.map((x) => x / n);
    });
  }
}
export interface HttpEmbeddingOptions {
  endpoint: string;
  model?: string;
  headers?: Record<string, string>;
  apiKey?: string;
  fetch?: typeof globalThis.fetch;
}
export class HttpEmbeddingProvider implements EmbeddingProvider {
  constructor(private options: HttpEmbeddingOptions) {}
  async embed(texts: string[]): Promise<number[][]> {
    const f = this.options.fetch ?? globalThis.fetch;
    const headers = {
      "content-type": "application/json",
      ...(this.options.apiKey
        ? { authorization: `Bearer ${this.options.apiKey}` }
        : {}),
      ...this.options.headers,
    };
    const response = await f(this.options.endpoint, {
      method: "POST",
      headers,
      body: JSON.stringify({ input: texts, model: this.options.model }),
    });
    if (!response.ok)
      throw new Error(
        `Embedding request failed: ${response.status} ${await response.text()}`,
      );
    const data = (await response.json()) as {
      data?: Array<{ embedding: number[] }>;
      embeddings?: number[][];
    };
    const vectors = data.data?.map((x) => x.embedding) ?? data.embeddings;
    if (!vectors || vectors.length !== texts.length)
      throw new Error("Invalid embedding response");
    return vectors;
  }
}
export class FunctionEmbeddingProvider implements EmbeddingProvider {
  constructor(
    private fn: (texts: string[]) => Promise<number[][]> | number[][],
  ) {}
  async embed(texts: string[]) {
    return await this.fn(texts);
  }
}
