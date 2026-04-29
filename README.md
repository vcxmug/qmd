# QMD — Ollama Edition

A fork of [tobi/qmd](https://github.com/tobi/qmd) that replaces `node-llama-cpp` with **Ollama** as the sole LLM backend. No GGUF downloads, no native builds — just Ollama's REST API.

> **What is QMD?** An on-device hybrid search engine for markdown files: BM25 keyword search + vector semantic search + LLM re-ranking + AI query expansion. Think of it as your personal search engine for notes, docs, and knowledge bases.

![QMD Architecture](assets/qmd-architecture.png)

## Differences from upstream

| | Upstream | This fork |
|---|---|---|
| LLM backend | `node-llama-cpp` (GGUF) | **Ollama** (REST API) |
| Model management | HuggingFace download | `ollama pull` / `ollama create` |
| GPU | Required for reasonable speed | Ollama handles it |
| Config | `index.yml` + env vars | `index.yml` only |
| Embedding | Local GGUF inference | Ollama `/api/embeddings` |
| Query expansion | Local GGUF model | Ollama `/api/generate` |
| Reranking | Local GGUF model | Ollama cosine-similarity |

---

## Quick Start

### 1. Prerequisites

- **Node.js** >= 22
- **Ollama** installed and running on `http://localhost:11434`

### 2. Pull / Install Models

QMD uses three models. The embedding model is in the Ollama registry; the reranker and query-expansion models need manual setup.

```bash
# ── Embedding (274 MB) ── pull directly
ollama pull nomic-embed-text

# ── Reranker (639 MB) ── download GGUF and create
wget -O /tmp/qwen3-reranker-0.6b-q8_0.gguf \
  "https://huggingface.co/ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/resolve/main/qwen3-reranker-0.6b-q8_0.gguf"
ollama create qwen3-reranker:0.6b -f <(echo 'FROM /tmp/qwen3-reranker-0.6b-q8_0.gguf')

# ── Query Expansion (1.3 GB) ── download GGUF and create
wget -O /tmp/qmd-query-expansion-1.7b-q4_k_m.gguf \
  "https://huggingface.co/tobil/qmd-query-expansion-1.7B-gguf/resolve/main/qmd-query-expansion-1.7B-q4_k_m.gguf"
ollama create qmd-query-expansion:1.7b -f <(echo 'FROM /tmp/qmd-query-expansion-1.7b-q4_k_m.gguf')

# Verify
ollama list
```

> **What's the query-expansion model?** A fine-tuned Qwen3-1.7B that rewrites your search query into keyword variants (`lex:`), semantic variants (`vec:`), and hypothetical-document passages (`hyde:`) — each optimized for a different search backend. Read more at [tobil/qmd-query-expansion-1.7B](https://huggingface.co/tobil/qmd-query-expansion-1.7B).

### 3. Install QMD

```bash
git clone https://github.com/vcxmug/qmd.git
cd qmd
npm install
npm run build
npm link    # optional — makes `qmd` globally available
```

### 4. Configure

Create `~/.config/qmd/index.yml`:

```yaml
models:
  embed: nomic-embed-text
  rerank: qwen3-reranker:0.6b
  generate: qmd-query-expansion:1.7b
  ollamaUrl: http://localhost:11434   # optional, this is the default

collections:
  my-notes:
    path: ~/notes
    pattern: "**/*.md"
```

> The `rerank` and `generate` fields are optional. Without `generate`, QMD skips AI query expansion (keyword + vector search still works). Without `rerank`, QMD uses the embed model for cosine-similarity reranking.

### 5. Index & Search

```bash
qmd index                          # scan collections and index documents
qmd embed                          # generate vector embeddings
qmd query "deployment steps"       # hybrid search (keyword + vector + rerank)
qmd status                         # check index health
```

---

## Usage Reference

```bash
# Search (no embeddings needed)
qmd search "auth config"           # BM25 keyword search

# Vector search (requires embeddings)
qmd vsearch "how to set up auth"   # semantic similarity search

# Hybrid search (best quality)
qmd query "quarterly planning"     # keyword + vector + LLM rerank

# Management
qmd collection add docs ~/Documents  # add a collection
qmd collection list                  # list collections
qmd collection remove docs           # remove a collection
qmd context add docs/ "/" "My work documents"  # add folder context for better search
qmd update                           # re-scan collections for changes
qmd embed --force                    # regenerate all embeddings
qmd cleanup                          # vacuum DB, remove orphans

# Tools
qmd get path/to/file.md             # view indexed document
qmd multi "*.md"                    # batch-dump documents
```

---

## Configuration Reference

Full `~/.config/qmd/index.yml`:

```yaml
# Model configuration (all fields optional — defaults shown)
models:
  embed: nomic-embed-text          # embedding model (Ollama name)
  rerank: qwen3-reranker:0.6b     # reranking model (uses embed model if omitted)
  generate: qmd-query-expansion:1.7b  # query expansion model (skips if omitted)
  ollamaUrl: http://localhost:11434   # Ollama server URL

# Global context injected into every query
global_context: ""

# Collections to index
collections:
  my-notes:
    path: ~/notes                   # absolute or ~-prefixed path
    pattern: "**/*.md"              # glob pattern
    ignore:                         # globs to exclude (optional)
      - "private/**"
      - "*.tmp.md"
    includeByDefault: true          # include in queries by default
    context:                        # folder-level context (optional)
      "/": "Personal knowledge base"
      "/meetings": "Weekly team meeting notes"
```

---

## Development

```bash
npm run qmd -- status               # run CLI from source (no build)
npm run qmd -- search "test query"
npm test                            # run tests
```

---

## Upstream

- Original project: [tobi/qmd](https://github.com/tobi/qmd)
- Original README: [README_original.md](README_original.md)
- License: MIT
