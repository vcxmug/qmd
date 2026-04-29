#!/usr/bin/env bun
/**
 * QMD Reranker Benchmark (Ollama Edition)
 *
 * Measures reranking performance via Ollama API.
 * Reports throughput for different document counts.
 *
 * Usage:
 *   bun src/bench-rerank.ts              # full benchmark
 *   bun src/bench-rerank.ts --quick      # quick smoke test (10 docs, 1 iteration)
 *   bun src/bench-rerank.ts --docs 100   # custom doc count
 */

import { OllamaLLM } from "./ollama.js";
import { type RerankDocument } from "./llm.js";
import { loadConfig } from "./collections.js";

// ============================================================================
// Config
// ============================================================================

const args = process.argv.slice(2);
const quick = args.includes("--quick");
const docsIdx = args.indexOf("--docs");
const DOC_COUNT = docsIdx >= 0 ? parseInt(args[docsIdx + 1]!) : (quick ? 10 : 40);
const ITERATIONS = quick ? 1 : 3;

// Load config to get model names
let embedModel = "nomic-embed-text";
let rerankModel: string | undefined;
let ollamaUrl: string | undefined;
try {
  const config = loadConfig();
  if (config.models) {
    embedModel = config.models.embed || embedModel;
    rerankModel = config.models.rerank;
    ollamaUrl = config.models.ollamaUrl;
  }
} catch { /* use defaults */ }

// ============================================================================
// Test data
// ============================================================================

const QUERY = "How do AI agents work and what are their limitations?";

function generateDocs(n: number): string[] {
  const templates = [
    "Artificial intelligence agents are software systems that perceive their environment and take actions to achieve goals. They use techniques like reinforcement learning, planning, and natural language processing to operate autonomously.",
    "The transformer architecture, introduced in 2017, revolutionized natural language processing. Self-attention mechanisms allow models to weigh the importance of different parts of input sequences when generating outputs.",
    "Machine learning models require careful evaluation to avoid overfitting. Cross-validation, holdout sets, and metrics like precision, recall, and F1 score help assess generalization performance.",
    "Retrieval-augmented generation combines information retrieval with language models. Documents are embedded into vector spaces, retrieved based on query similarity, and used as context for generation.",
    "Neural network training involves forward propagation, loss computation, and backpropagation. Optimizers like Adam and SGD adjust weights to minimize the loss function over training iterations.",
    "Large language models exhibit emergent capabilities at scale, including few-shot learning, chain-of-thought reasoning, and instruction following. These properties were not explicitly trained for.",
    "Embedding models convert text into dense vector representations that capture semantic meaning. Similar texts produce similar vectors, enabling efficient similarity search and clustering.",
    "Autonomous agents face challenges including hallucination, lack of grounding, limited planning horizons, and difficulty with multi-step reasoning. Safety and alignment remain open research problems.",
    "The attention mechanism computes query-key-value interactions to determine which parts of the input are most relevant. Multi-head attention allows the model to attend to different representation subspaces.",
    "Fine-tuning adapts a pre-trained model to specific tasks using domain-specific data. Techniques like LoRA reduce the number of trainable parameters while maintaining performance.",
  ];
  return Array.from({ length: n }, (_, i) => templates[i % templates.length]!);
}

// ============================================================================
// Helpers
// ============================================================================

function median(arr: number[]): number {
  const sorted = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 !== 0 ? sorted[mid]! : (sorted[mid - 1]! + sorted[mid]!) / 2;
}

// ============================================================================
// Main
// ============================================================================

async function main() {
  console.log("═══════════════════════════════════════════════════════════════");
  console.log("  QMD Reranker Benchmark (Ollama)");
  console.log("═══════════════════════════════════════════════════════════════\n");

  const llm = new OllamaLLM({
    url: ollamaUrl,
    embedModel,
    rerankModel,
  });

  console.log("System");
  console.log(`  Backend:    Ollama`);
  console.log(`  URL:        ${ollamaUrl || "http://localhost:11434"}`);
  console.log(`  Embed:      ${embedModel}`);
  console.log(`  Rerank:     ${rerankModel || embedModel + " (using embed model)"}`);

  // Generate test docs
  const docs = generateDocs(DOC_COUNT);
  const rerankDocs: RerankDocument[] = docs.map((text, i) => ({
    file: `doc_${i}.md`,
    text,
    title: `Document ${i}`,
  }));

  console.log(`\nBenchmark`);
  console.log(`  Documents:  ${DOC_COUNT}`);
  console.log(`  Iterations: ${ITERATIONS}`);
  console.log(`  Query:      "${QUERY.slice(0, 50)}..."`);

  // Run benchmark
  const times: number[] = [];
  for (let iter = 0; iter < ITERATIONS; iter++) {
    process.stdout.write(`  Run ${iter + 1}/${ITERATIONS}...`);
    const t0 = performance.now();
    const result = await llm.rerank(QUERY, rerankDocs);
    const elapsed = performance.now() - t0;
    times.push(elapsed);
    process.stdout.write(` ${elapsed.toFixed(0)}ms (${result.results.length} results, top score: ${result.results[0]?.score.toFixed(4) || "N/A"})\n`);
  }

  // Summary
  const med = median(times);
  const docsPerSec = (DOC_COUNT / med) * 1000;
  console.log(`\n  Results:`);
  console.log(`    Median:   ${med.toFixed(0)}ms`);
  console.log(`    Throughput: ${docsPerSec.toFixed(1)} docs/s`);

  await llm.dispose();
  console.log("");
}

main().catch(console.error);
