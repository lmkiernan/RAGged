# AutoEmbed

AutoEmbed is an ML framework for Retrieval-Augmented Generation (RAG) that automatically selects the Pareto optimal chunking strateg, size, and embedding model for any document corpus to maximize retrieval performance under cost and latency constraints.

---

## 🚀 Features

- **Multi-format Ingestion**: Load and normalize text from PDF, Markdown, and HTML documents.
- **Pluggable Chunking Engine**:  
  - Sentence-Aware (semantic boundaries)  
  - Fixed-Token (uniform N-token chunks)  
  - Sliding-Window (overlapping token windows)  
- **Embedding Interface**:  
  - OpenAIEmbedder (OpenAI API)  
  - HFEmbedder (Hugging Face Transformers)
- **Vector Store Integration**:  
  - Qdrant via Docker Compose  
  - Upsert & search with metadata filters
- **Automated Evaluation Harness**:  
  - LLM-generated or manually authored gold questions  
  - Standard IR metrics: Recall@K, MRR, Precision@K, F1 overlap, Exact Match  
- **Optimization Loop**: Grid/random search over strategies, models, chunk sizes, and overlap → Pareto frontier & recommended config
- **Interactive CLI & Demo**:  
  - `autoembed ingest|chunk|embed|search|eval|optimize` commands  
  - (Optional) Streamlit UI for upload and one-click evaluation

---

## 📁 Repository Structure

```
AutoEmbed/
├── config/
│   └── default.yaml      # Pipeline parameters & objectives
├── data/                 # Example raw files (PDF, MD, HTML)
├── ingested/             # Normalized JSON outputs from ingestion
├── chunks/               # Chunk JSONs per strategy
├── embeddings/           # (Optional) Cached vectors
├── src/
│   ├── config.py         # YAML loader
│   ├── ingestion.py      # File readers
│   ├── chunk_router.py   # Strategy dispatcher
│   ├── run_chunking.py   # CLI to generate chunks
│   ├── embeddings.py     # Embedder interface & factory
│   ├── run_embedding.py  # Embed & upsert to Qdrant
│   ├── vectorstore.py    # Qdrant wrapper (upsert/search)
│   ├── run_search.py     # Interactive query→results
│   ├── eval.py           # Batch eval & metrics
│   └── optimize.py       # ConfigTester & Pareto loop
├── docker-compose.yml    # Qdrant service
├── generate_gold_qa.py   # LLM-based QA generation
├── gold_queries.yaml     # Example gold query file
├── logs/                 # Evaluation & cost logs
└── README.md             # This file
```

---

## ⚙️ Setup Instructions

1. **Clone the repo**
   ```bash
   git clone https://github.com/your-org/AutoEmbed.git
   cd AutoEmbed
   ```

2. **Install dependencies**
   The key libraries are:
   - `openai`, `transformers`, `torch`
   - `qdrant-client`
   - `PyYAML` for config

3. **Configure API keys in a json APIKeys.json at the root**
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

4. **Start Qdrant**
   ```bash
   docker compose up -d
   ```
  
5. **Put your own corpus (files) in data/**

5. **Edit `config/default.yaml`** to set your chunking, embedding, and evaluation objectives.

---

## 🏃 Quick Start

   ```bash
  python3 src/run_ragged.py
   ```
   runs everything

---

## 📘 License & Contribution

MIT License. Contributions welcome via pull requests or issues on GitHub.

