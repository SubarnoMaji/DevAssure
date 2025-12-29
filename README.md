## Setup

### 1. Clone and configure

```bash
cd agent
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 2. Install dependencies

```bash
# Indexer
cd indexer
poetry install

# Agent
cd ../agent
poetry install

# Frontend
cd ../frontend
poetry install
```

## Running

Start each service in a separate terminal:

```bash
# Terminal 1: ChromaDB server (from indexer/)
cd indexer
poetry run chroma run --path ./chroma_db 
# Terminal 2: Index documents (from indexer/)
cd indexer
poetry run python indexer.py

# Terminal 3: File upload API (from frontend/)
cd frontend
poetry run python api.py

# Terminal 4: Agent API (from agent/)
cd agent
poetry run python app.py

# Terminal 5: Streamlit UI (from frontend/)
cd frontend
poetry run streamlit run rag-bot.py
```

## Usage

1. Open http://localhost:8501 in your browser
2. Upload documents via the sidebar
3. Run the indexer to process new documents
4. Chat with your documents using the RAG toggle
