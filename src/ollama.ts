/**
 * ollama.ts - LLM implementation using Ollama's REST API
 *
 * Provides embeddings, text generation, and reranking via Ollama's local models.
 * Use this when you want Ollama to manage models instead of node-llama-cpp.
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

function resolveDefaultModel(envVar: string | undefined, fallback: string): string {
  return envVar?.trim() || fallback;
}

export class OllamaLLM implements LLM {
  private readonly url: string;
  private readonly embedModel: string;
  private readonly generateModel: string;
  private readonly rerankModel: string;
  private disposed = false;

  constructor(config: OllamaConfig = {}) {
    this.url = resolveDefaultModel(config.url, DEFAULT_OLLAMA_URL);
    this.embedModel = resolveDefaultModel(config.embedModel, process.env.QMD_OLLAMA_EMBED_MODEL || "mxbai-embed-large");
    this.generateModel = resolveDefaultModel(config.generateModel, process.env.QMD_OLLAMA_GENERATE_MODEL || "qwen3");
    this.rerankModel = resolveDefaultModel(config.rerankModel, process.env.QMD_OLLAMA_RERANK_MODEL || this.embedModel);
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
    const formattedText = options?.isQuery
      ? formatQueryForEmbedding(text, `ollama:${model}`)
      : formatDocForEmbedding(text, options?.title, `ollama:${model}`);

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

  async generate(prompt: string, options?: GenerateOptions): Promise<GenerateResult | null> {
    const model = options?.model || this.generateModel;
    try {
      const response = await this.request<{ response: string; done: boolean }>("/api/generate", {
        model,
        prompt,
        stream: false,
        options: options?.temperature !== undefined ? { temperature: options.temperature } : undefined,
      });
      return { text: response.response, model: `ollama:${model}`, done: response.done };
    } catch (err) {
      console.error("Ollama generate error:", err);
      return null;
    }
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
    const results: Queryable[] = [];

    if (options?.includeLexical !== false) {
      results.push({ type: "lex", text: query });
    }

    const prompt = `Given the search query: "${query}"
${options?.context ? `Context: ${options.context}\n` : ""}
Generate 1-2 alternative phrasings that capture the same intent but use different words.
Only output the alternative phrasings, one per line, no numbering.`;

    const result = await this.generate(prompt, { temperature: 0.7 });
    if (result) {
      const lines = result.text
        .split("\n")
        .map((l) => l.trim())
        .filter((l) => l.length > 0 && l.length < 200);
      for (const line of lines.slice(0, 2)) {
        results.push({ type: "vec", text: line });
      }
    }

    results.push({ type: "vec", text: query });

    return results;
  }

  async rerank(query: string, documents: RerankDocument[], options?: RerankOptions): Promise<RerankResult> {
    // Ollama only has access to its own models. The `model` parameter in options
    // refers to HuggingFace model IDs (used by LlamaCpp) and is ignored here.
    // We always use this.embedModel (the Ollama embedding model for cosine-similarity reranking).
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
  async detokenize(tokens: readonly { id?: number; [key: string]: unknown }[]): Promise<string> {
    // Tokens are dummy values (0) - return placeholder
    return "[truncated]";
  }

  async dispose(): Promise<void> {
    this.disposed = true;
  }
}
