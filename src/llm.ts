/**
 * llm.ts - LLM abstraction layer for QMD
 *
 * Defines the LLM interface, shared types, and session management.
 * Implementations (e.g. Ollama) live in separate files.
 */

// =============================================================================
// Embedding Formatting Functions
// =============================================================================

/**
 * Detect if a model URI uses a Qwen3-Embedding-like format.
 */
export function isQwen3EmbeddingModel(modelUri: string): boolean {
  return /qwen.*embed/i.test(modelUri) || /embed.*qwen/i.test(modelUri);
}

/**
 * Format a query for embedding.
 */
export function formatQueryForEmbedding(query: string, modelUri?: string): string {
  if (modelUri && isQwen3EmbeddingModel(modelUri)) {
    return `Instruct: Retrieve relevant documents for the given query\nQuery: ${query}`;
  }
  return `task: search result | query: ${query}`;
}

/**
 * Format a document for embedding.
 */
export function formatDocForEmbedding(text: string, title?: string, modelUri?: string): string {
  if (modelUri && isQwen3EmbeddingModel(modelUri)) {
    return title ? `${title}\n${text}` : text;
  }
  return `title: ${title || "none"} | text: ${text}`;
}

// =============================================================================
// Types
// =============================================================================

/**
 * Token with log probability
 */
export type TokenLogProb = {
  token: string;
  logprob: number;
};

/**
 * Embedding result
 */
export type EmbeddingResult = {
  embedding: number[];
  model: string;
};

/**
 * Generation result with optional logprobs
 */
export type GenerateResult = {
  text: string;
  model: string;
  logprobs?: TokenLogProb[];
  done: boolean;
};

/**
 * Rerank result for a single document
 */
export type RerankDocumentResult = {
  file: string;
  score: number;
  index: number;
};

/**
 * Batch rerank result
 */
export type RerankResult = {
  results: RerankDocumentResult[];
  model: string;
};

/**
 * Model info
 */
export type ModelInfo = {
  name: string;
  exists: boolean;
  path?: string;
};

/**
 * Options for embedding
 */
export type EmbedOptions = {
  model?: string;
  isQuery?: boolean;
  title?: string;
};

/**
 * Options for text generation
 */
export type GenerateOptions = {
  model?: string;
  maxTokens?: number;
  temperature?: number;
};

/**
 * Options for reranking
 */
export type RerankOptions = {
  model?: string;
};

/**
 * Options for LLM sessions
 */
export type LLMSessionOptions = {
  /** Max session duration in ms (default: 10 minutes) */
  maxDuration?: number;
  /** External abort signal */
  signal?: AbortSignal;
  /** Debug name for logging */
  name?: string;
};

/**
 * Session interface for scoped LLM access with lifecycle guarantees
 */
export interface ILLMSession {
  embed(text: string, options?: EmbedOptions): Promise<EmbeddingResult | null>;
  embedBatch(texts: string[], options?: EmbedOptions): Promise<(EmbeddingResult | null)[]>;
  expandQuery(query: string, options?: { context?: string; includeLexical?: boolean; intent?: string }): Promise<Queryable[]>;
  rerank(query: string, documents: RerankDocument[], options?: RerankOptions): Promise<RerankResult>;
  /** Whether this session is still valid (not released or aborted) */
  readonly isValid: boolean;
  /** Abort signal for this session (aborts on release or maxDuration) */
  readonly signal: AbortSignal;
}

/**
 * Supported query types for different search backends
 */
export type QueryType = 'lex' | 'vec' | 'hyde';

/**
 * A single query and its target backend type
 */
export type Queryable = {
  type: QueryType;
  text: string;
};

/**
 * Document to rerank
 */
export type RerankDocument = {
  file: string;
  text: string;
  title?: string;
};

// =============================================================================
// LLM Interface
// =============================================================================

/**
 * Abstract LLM interface - implement this for different backends
 */
export interface LLM {
  /**
   * Get embeddings for text
   */
  embed(text: string, options?: EmbedOptions): Promise<EmbeddingResult | null>;

  /**
   * Batch embeddings (may be implemented as sequential calls if backend doesn't support true batching)
   */
  embedBatch(texts: string[], options?: EmbedOptions): Promise<(EmbeddingResult | null)[]>;

  /**
   * Name of the default embedding model
   */
  embedModelName: string;

  /**
   * Generate text completion
   */
  generate(prompt: string, options?: GenerateOptions): Promise<GenerateResult | null>;

  /**
   * Check if a model exists/is available
   */
  modelExists(model: string): Promise<ModelInfo>;

  /**
   * Expand a search query into multiple variations for different backends.
   * Returns a list of Queryable objects.
   */
  expandQuery(query: string, options?: { context?: string, includeLexical?: boolean, intent?: string }): Promise<Queryable[]>;

  /**
   * Rerank documents by relevance to a query
   * Returns list of documents with relevance scores (higher = more relevant)
   */
  rerank(query: string, documents: RerankDocument[], options?: RerankOptions): Promise<RerankResult>;

  /**
   * Dispose of resources
   */
  dispose(): Promise<void>;

  /**
   * Tokenize text (used by chunking)
   */
  tokenize(text: string): Promise<number[]>;

  /**
   * Detokenize token IDs back to text
   */
  detokenize(tokens: readonly number[]): Promise<string>;
}

// =============================================================================
// Session Management
// =============================================================================

/**
 * Error thrown when an operation is attempted on a released or aborted session.
 */
export class SessionReleasedError extends Error {
  constructor(message = "LLM session has been released or aborted") {
    super(message);
    this.name = "SessionReleasedError";
  }
}

/**
 * Simple LLM session wrapper with lifecycle management (abort, timeout, release).
 */
class SimpleLLMSession implements ILLMSession {
  private llm: LLM;
  private released = false;
  private abortController: AbortController;
  private maxDurationTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(llm: LLM, options: LLMSessionOptions = {}) {
    this.llm = llm;
    this.abortController = new AbortController();

    if (options.signal) {
      if (options.signal.aborted) {
        this.abortController.abort(options.signal.reason);
      } else {
        options.signal.addEventListener("abort", () => {
          this.abortController.abort(options.signal!.reason);
        }, { once: true });
      }
    }

    const maxDuration = options.maxDuration ?? 10 * 60 * 1000;
    if (maxDuration > 0) {
      this.maxDurationTimer = setTimeout(() => {
        this.abortController.abort(new Error("Session exceeded max duration"));
      }, maxDuration);
      this.maxDurationTimer.unref();
    }
  }

  get isValid(): boolean {
    return !this.released && !this.abortController.signal.aborted;
  }

  get signal(): AbortSignal {
    return this.abortController.signal;
  }

  release(): void {
    if (this.released) return;
    this.released = true;
    if (this.maxDurationTimer) {
      clearTimeout(this.maxDurationTimer);
      this.maxDurationTimer = null;
    }
    this.abortController.abort(new Error("Session released"));
  }

  async embed(text: string, options?: EmbedOptions): Promise<EmbeddingResult | null> {
    if (!this.isValid) throw new SessionReleasedError();
    return this.llm.embed(text, options);
  }

  async embedBatch(texts: string[], options?: EmbedOptions): Promise<(EmbeddingResult | null)[]> {
    if (!this.isValid) throw new SessionReleasedError();
    return this.llm.embedBatch(texts, options);
  }

  async expandQuery(query: string, options?: { context?: string; includeLexical?: boolean }): Promise<Queryable[]> {
    if (!this.isValid) throw new SessionReleasedError();
    return this.llm.expandQuery(query, options);
  }

  async rerank(query: string, documents: RerankDocument[], options?: RerankOptions): Promise<RerankResult> {
    if (!this.isValid) throw new SessionReleasedError();
    return this.llm.rerank(query, documents, options);
  }
}

/**
 * Execute a function with a scoped LLM session.
 * The session provides lifecycle guarantees - resources won't be disposed mid-operation.
 *
 * @example
 * ```typescript
 * await withLLMSession(llm, async (session) => {
 *   const expanded = await session.expandQuery(query);
 *   const reranked = await session.rerank(query, docs);
 *   return reranked;
 * }, { maxDuration: 10 * 60 * 1000 });
 * ```
 */
export async function withLLMSession<T>(
  llm: LLM,
  fn: (session: ILLMSession) => Promise<T>,
  options?: LLMSessionOptions
): Promise<T> {
  const session = new SimpleLLMSession(llm, options);
  try {
    return await fn(session);
  } finally {
    session.release();
  }
}

/** @deprecated Use withLLMSession instead. */
export async function withLLMSessionForLlm<T>(
  llm: LLM,
  fn: (session: ILLMSession) => Promise<T>,
  options?: LLMSessionOptions
): Promise<T> {
  return withLLMSession(llm, fn, options);
}
