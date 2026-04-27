# 🍳 Pro Chef AI: True Hybrid RAG Recipe Engine

Pro Chef AI is a high-performance **Retrieval-Augmented Generation (RAG)** system. Unlike traditional keyword search, this model uses mathematical vector space to understand the "meaning" behind user queries, ensuring 90%+ accuracy even with complex requests.

## 🧠 System Architecture

This project follows a professional AI pipeline:
1. **Semantic Ingestion**: Recipes are converted into 384-dimensional vectors using the `all-MiniLM-L6-v2` transformer.
2. **Vector Store**: Embeddings are stored in **ChromaDB** using **Cosine Similarity** (`hnsw:space: cosine`).
3. **Hybrid Search**: 
   - **70% Semantic Weight**: Uses vector distance to find conceptually related recipes.
   - **30% Keyword Weight**: Uses exact ingredient intersection to ensure precision.
4. **LLM Generation**: The top-ranked results are passed to **Llama 3.3**, which generates a natural response while strictly preventing hallucinations.



## 🛠️ Installation & Setup

### 1. Backend Setup
```bash
cd backend
# Install AI and Server libraries
pip install fastapi uvicorn chromadb sentence-transformers groq pandas


Dependencies
Key libraries used:

fastapi + uvicorn → API backend

httpx → async HTTP client

pydantic → request/response models

python-dotenv → environment variable management

streamlit → frontend UI

groq → Groq LLM client

langchain-huggingface → vector store + embeddings

chromadb → persistent vector DB

sentence-transformers → MiniLM embeddings

Key Features
Semantic Understanding: Finds "Desserts" when you ask for "something sweet," even if the word sweet isn't in the recipe.

Exclusion Logic: Robustly handles constraints like "no chicken" or "without onions."

Persistent Memory: Embeddings are saved locally to disk, so the server starts instantly after the first setup.
