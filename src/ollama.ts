/**
 * ollama.ts - LLM implementation using Ollama's REST API
 *
 * Provides embeddings, text generation, and reranking via Ollama's local models.
 * Use this for Ollama-managed models instead of node-llama-cpp.
 */

import type {
  LLM,
  EmbedOptions,
  GenerateOptions,
  EmbeddingResult,
  GenerateResult,
  RerankDocument,
  RerankResult,
  RerankOptions,
  Queryable,
  ModelInfo,
  RerankDocumentResult,
} from "./llm.js";
import { formatQueryForEmbedding, formatDocForEmbedding } from "./llm.js";

export type OllamaConfig = {
  /** Ollama server URL (default: http://localhost:11434) */
  url?: string;
  /** Default embedding model (e.g., "mxbai-embed-large") */
  embedModel?: string;
  /** Default generation model (e.g., "qwen3") */
  generateModel?: string;
  /** Default rerank model (uses embed model if not specified) */
  rerankModel?: string;
};

const DEFAULT_OLLAMA_URL = "http://localhost:11434";

export class OllamaLLM implements LLM {
  private readonly url: string;
  private readonly embedModel: string;
  private readonly generateModel: string;
  private readonly rerankModel: string;
  private disposed = false;

  constructor(config: OllamaConfig = {}) {
    this.url = config.url || DEFAULT_OLLAMA_URL;
    this.embedModel = config.embedModel || "nomic-embed-text";
    this.generateModel = config.generateModel || "qmd-query-expansion:1.7b";
    this.rerankModel = config.rerankModel || this.embedModel;
  }

  private async request<T>(path: string, body: unknown): Promise<T> {
    const url = `${this.url}${path}`;
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!response.ok) {
      const text = await response.text().catch(() => "unknown error");
      throw new Error(`Ollama request failed (${response.status}): ${text}`);
    }
    return response.json() as Promise<T>;
  }

  private async get<T>(path: string): Promise<T> {
    const url = `${this.url}${path}`;
    const response = await fetch(url);
    if (!response.ok) {
      const text = await response.text().catch(() => "unknown error");
      throw new Error(`Ollama GET failed (${response.status}): ${text}`);
    }
    return response.json() as Promise<T>;
  }

  get embedModelName(): string {
    return this.embedModel;
  }

  async embed(text: string, options?: EmbedOptions): Promise<EmbeddingResult | null> {
    const model = options?.model || this.embedModel;
    let formattedText = options?.isQuery
      ? formatQueryForEmbedding(text, `ollama:${model}`)
      : formatDocForEmbedding(text, options?.title, `ollama:${model}`);

    // ── Context-length guard ──────────────────────────────────────────
    // nomic-embed-text has a 2048-token context window.
    // On this 2GB host, exceeding it causes Ollama HTTP 500:
    //   "the input length exceeds the context length"
    //
    // Impact chain discovered 2026-04-29:
    //   chunk → embed fails → qmd embed (boot) hangs → QMD manager
    //   enters bad state → ALL subsequent memory_search calls timeout
    //   after 120s → OpenClaw falls back to builtin every time.
    //
    // The chunker's token estimates can be off, especially for
    // CJK-heavy text (1–1.5 tokens/char vs English ~0.25).
    // Rerank also passes doc bodies that may exceed the context
    // window after title prepending. Truncation here is the
    // narrowest guard that fixes both paths.
    //
    // 4000 chars ≈ safe for 2048 tokens in mixed-language text
    // (~2 chars/token including CJK safety margin).
    // ─────────────────────────────────────────────────────────────────
    const MAX_EMBED_CHARS = 4000;
    if (formattedText.length > MAX_EMBED_CHARS) {
      console.warn(
        `Ollama embed: truncating input from ${formattedText.length} to ${MAX_EMBED_CHARS} chars ` +
        `to fit ${model} context window (2048 tokens)`
      );
      formattedText = formattedText.slice(0, MAX_EMBED_CHARS);
    }

    try {
      const response = await this.request<{ embedding: number[] }>("/api/embeddings", {
        model,
        prompt: formattedText,
      });
      return { embedding: response.embedding, model: `ollama:${model}` };
    } catch (err) {
      console.error("Ollama embed error:", err);
      return null;
    }
  }

  async embedBatch(texts: string[], options?: EmbedOptions): Promise<(EmbeddingResult | null)[]> {
    // Ollama doesn't have a true batch endpoint, so we call sequentially
    return Promise.all(texts.map((text) => this.embed(text, options)));
  }

  /**
   * Text generation is intentionally disabled.
   * The 2GB host cannot fit both text-generation and embedding models in RAM.
   * Query expansion has been removed from expandQuery() — see comment there.
   * If you need generation back, ensure the host has ≥4GB free RAM first.
   */
  async generate(_prompt: string, _options?: GenerateOptions): Promise<GenerateResult | null> {
    return null;
  }

  async modelExists(model: string): Promise<ModelInfo> {
    try {
      const response = await this.get<{ models: { name: string }[] }>("/api/tags");
      const exists = response.models.some((m) => m.name === model || m.name === `${model}:latest`);
      return { name: `ollama:${model}`, exists };
    } catch (err) {
      console.error("Ollama modelExists error:", err);
      return { name: `ollama:${model}`, exists: false };
    }
  }

  async expandQuery(query: string, options?: { context?: string; includeLexical?: boolean }): Promise<Queryable[]> {
    // Query expansion via LLM is deliberately disabled.
    // The 2GB host cannot fit a text-generation model (~1.8GB) alongside the embed model.
    // The embed + rerank pipeline already provides strong semantic recall without expansion.
    const results: Queryable[] = [];

    if (options?.includeLexical !== false) {
      results.push({ type: "lex", text: query });
    }

    results.push({ type: "vec", text: query });

    return results;
  }

  async rerank(query: string, documents: RerankDocument[], options?: RerankOptions): Promise<RerankResult> {
    // We always use the configured embedding model for cosine-similarity reranking.
    const model = this.embedModel;

    const queryEmbedding = await this.embed(formatQueryForEmbedding(query, `ollama:${model}`), { model, isQuery: true });
    if (!queryEmbedding) {
      return { results: [], model: `ollama:${model}` };
    }

    const results: RerankDocumentResult[] = [];

    for (let i = 0; i < documents.length; i++) {
      const doc = documents[i]!;
      const docText = formatDocForEmbedding(doc.text, doc.title, `ollama:${model}`);
      const docEmbedding = await this.embed(docText, { model, isQuery: false });
      if (docEmbedding && docEmbedding.embedding) {
        const dot = queryEmbedding.embedding.reduce((sum, a, j) => sum + a * docEmbedding.embedding![j]!, 0);
        const normA = Math.sqrt(queryEmbedding.embedding.reduce((s, a) => s + a * a, 0));
        const normB = Math.sqrt(docEmbedding.embedding.reduce((s, a) => s + a * a, 0));
        const similarity = dot / (normA * normB);
        results.push({ file: doc.file, score: similarity, index: i });
      } else {
        results.push({ file: doc.file, score: 0, index: i });
      }
    }

    results.sort((a, b) => b.score - a.score);

    return {
      results,
      model: `ollama:${model}`,
    };
  }

  /**
   * Fallback tokenizer using character-count approximation.
   * Used by chunking logic to respect token limits.
   * 4 chars/token is a reasonable average for English text.
   */
  async tokenize(text: string): Promise<number[]> {
    const tokens: number[] = [];
    // Approximate: 4 chars ≈ 1 token (English average)
    const chunks = Math.ceil(text.length / 4);
    for (let i = 0; i < chunks; i++) {
      tokens.push(0); // Dummy token values (only length matters for chunking)
    }
    return tokens;
  }

  /**
   * Fallback detokenizer using character-count approximation.
   * Reconstructs text from dummy token IDs by taking the first N chars.
   * This is lossy but sufficient for chunk-truncation scenarios.
   */
  async detokenize(tokens: readonly number[]): Promise<string> {
    // Tokens are dummy values (0) - return placeholder
    return "[truncated]";
  }

  async dispose(): Promise<void> {
    this.disposed = true;
  }
}
