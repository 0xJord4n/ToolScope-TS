export type JsonSchema = Record<string, unknown>;
export interface CanonicalTool<T = unknown> {
  id: string;
  name: string;
  description: string;
  inputSchema: JsonSchema;
  tags: string[];
  namespace?: string;
  original: T;
}
export interface EmbeddingProvider {
  embed(texts: string[]): Promise<number[][]>;
}
export interface Reranker {
  rerank(query: string, tools: CanonicalTool[]): Promise<number[]>;
}
export interface IndexedTool {
  tool: CanonicalTool;
  vector: number[];
  text: string;
}
export interface VectorBackend {
  readonly size: number;
  upsert(records: IndexedTool[]): void | Promise<void>;
  remove(ids: string[]): void | Promise<void>;
  clear(): void | Promise<void>;
  list(): IndexedTool[];
  close?(): void;
}
export interface ToolTextConfig {
  useName?: boolean;
  useDescription?: boolean;
  useSchema?: boolean;
  useTags?: boolean;
  truncate?: number;
  preprocessors?: Array<(text: string) => string>;
}
export interface StickyConfig {
  enabled?: boolean;
  reuseThreshold?: number;
  refreshThreshold?: number;
  keep?: number;
  ttlMs?: number;
  maxSessions?: number;
}
export interface ToolScore {
  id: string;
  name: string;
  score: number;
  semanticScore: number;
  lexicalScore: number;
  rerankScore?: number;
  selected: boolean;
  reason?: string;
}
export interface SelectionTrace {
  id: string;
  timestamp: string;
  sessionId?: string;
  query: string;
  candidateCount: number;
  filteredCount: number;
  selected: ToolScore[];
  scores: ToolScore[];
  timings: {
    embeddingMs: number;
    searchMs: number;
    rerankMs: number;
    totalMs: number;
  };
  config: Record<string, unknown>;
}
export interface TraceSink {
  emit(trace: SelectionTrace): void | Promise<void>;
}
export interface FilterOptions {
  k?: number;
  allowTags?: string[];
  denyTags?: string[];
  namespace?: string;
  minScore?: number;
  sessionId?: string;
  semanticWeight?: number;
  lexicalWeight?: number;
  diversity?: number;
  reranker?: Reranker;
  rerankPoolSize?: number;
  policy?: (tool: CanonicalTool) => boolean | Promise<boolean>;
}
export interface ToolIndexOptions {
  embedder: EmbeddingProvider;
  backend?: VectorBackend;
  text?: ToolTextConfig;
  sticky?: StickyConfig;
  traceSinks?: TraceSink[];
}
export interface StatelessFilterOptions extends FilterOptions, ToolIndexOptions {}
