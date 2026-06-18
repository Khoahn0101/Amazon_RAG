# RAG Chatbot for Product Catalog

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about products by retrieving information from an indexed catalog and generating natural language responses.

## Project Structure

```
rag-chatbot/
├── data/                  # Your Kaggle dataset
├── models/               # Trained models and embeddings
├── src/
│   ├── data_loader.py   # Load and process CSV data
│   ├── chunker.py       # Break products into chunks
│   ├── embedder.py      # Create embeddings with ChromaDB
│   └── rag_pipeline.py  # LangChain RAG logic
├── app.py               # Streamlit UI
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Setup

1. **Create and activate virtual environment**:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

2. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

3. **Add your Kaggle dataset**:
   - Place your CSV file in the `data/` folder

4. **Run the app**:
   ```powershell
   streamlit run app.py
   ```

## Features

- 📚 **Vector Search**: Semantic search using ChromaDB
- 🧠 **RAG Pipeline**: LangChain-based retrieval and generation
- 💬 **Interactive Chat**: Streamlit UI for user interaction
- 🔍 **Hybrid Filtering**: Semantic search + structured filtering (price, specs)

## Technologies

- **LangChain**: RAG orchestration
- **ChromaDB**: Vector storage and retrieval
- **Streamlit**: Web UI
- **OpenAI**: LLM for response generation
- **Pandas**: Data processing

## Learning Goals

This project teaches:
1. Data chunking strategies
2. Vector embeddings and cosine similarity
3. RAG architecture
4. LLM prompt engineering
5. Full-stack ML deployment

## Next Steps

- [ ] Load and explore Kaggle dataset
- [ ] Implement data pipeline
- [ ] Create vector embeddings
- [ ] Build RAG pipeline
- [ ] Create Streamlit UI
- [ ] Deploy to production
