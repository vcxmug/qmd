# QMD — Ollama Edition

A fork of [tobi/qmd](https://github.com/tobi/qmd) that adds **Ollama** as an LLM backend, so you can run QMD on CPU-only servers without downloading GGUF models or dealing with `node-llama-cpp` builds.

> **What is QMD?** An on-device hybrid search engine for markdown files — BM25 keyword search + vector semantic search + LLM re-ranking. Originally powered by `node-llama-cpp` with HuggingFace GGUF models. This fork lets you use Ollama instead.

![QMD Architecture](assets/qmd-architecture.png)

## Differences from upstream

| | Upstream | This fork |
|---|---|---|
| LLM backend | `node-llama-cpp` (GGUF) | **Ollama** (REST API) |
| Model download | Automatic from HuggingFace | Managed by Ollama |
| GPU required | Recommended | **Not needed** (CPU works) |
| Embedding | Local GGUF inference | Ollama `/api/embeddings` |
| Query expansion | Local GGUF model | Ollama `/api/generate` |

## Quick Start

### Prerequisites

- **Node.js** >= 22
- **Ollama** installed and running (default: `http://localhost:11434`)
- An embedding model pulled in Ollama, e.g.:
  ```sh
  ollama pull nomic-embed-text
  ```

### Install

```sh
# Clone this fork
git clone https://github.com/vcxmug/qmd.git
cd qmd

# Install dependencies and build
npm install
npm run build

# Link globally (optional)
npm link
```

### Configuration

QMD looks for its config at `$XDG_CONFIG_HOME/qmd/index.yml` (default: `~/.config/qmd/index.yml`).

Create or edit `~/.config/qmd/index.yml`:

```yaml
# LLM backend — use "ollama" (this fork) or "llama-cpp" (upstream)
models:
  provider: ollama
  ollamaUrl: http://localhost:11434     # optional, this is the default
  embed: nomic-embed-text               # Ollama model for embeddings
  # generate: qwen3                     # optional, for query expansion (need to pull first)
  # rerank: nomic-embed-text            # optional, defaults to embed model

# Your markdown collections
collections:
  my-notes:
    path: ~/notes
    pattern: "**/*.md"
  docs:
    path: ~/Documents
    pattern: "**/*.md"
```

> If you don't configure a `generate` model, QMD skips AI query expansion — keyword + vector search still works fine.

### Index and Search

```sh
# Build the index (scans collections)
qmd index

# Generate embeddings for semantic search
qmd embed

# Keyword search
qmd search "deployment steps"

# Semantic search
qmd vsearch "how to configure the server"

# Hybrid search (keyword + semantic, best quality)
qmd query "quarterly planning"
```

### Optional: Pull a generate model for query expansion

```sh
# Pull a small model (qwen2.5:0.5b is ~400MB and works on CPU)
ollama pull qwen2.5:0.5b
```

Then add to your config:

```yaml
models:
  provider: ollama
  embed: nomic-embed-text
  generate: qwen2.5:0.5b    # enables AI query expansion
```

### Specifying a custom index path

```sh
# Use a different SQLite index
qmd --index /path/to/my-index.sqlite search "query"

# Or set the environment variable
export QMD_INDEX=/path/to/my-index.sqlite
```

## OpenClaw Integration

This fork is tested with OpenClaw's QMD memory backend. When OpenClaw overrides `XDG_CACHE_HOME` and `XDG_CONFIG_HOME`, make sure the config at the overridden path also has `provider: ollama`.

Example OpenClaw QMD config (`~/.openclaw/agents/main/qmd/xdg-config/qmd/index.yml`):

```yaml
models:
  provider: ollama
  ollamaUrl: http://localhost:11434
  embed: nomic-embed-text
collections:
  memory-dir-main:
    path: /path/to/memory
    pattern: "**/*.md"
```

## Development

```sh
# Run CLI from source (no build needed)
npm run qmd -- status
npm run qmd -- search "test"

# Run tests
npm test
```

## Upstream

- Original project: [tobi/qmd](https://github.com/tobi/qmd)
- Original README: [README_original.md](README_original.md)
- License: same as upstream
