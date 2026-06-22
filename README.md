# RAG Chatbot for Product Catalog

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about products by retrieving information from an indexed catalog and generating natural language responses.

## Project Structure

```
rag-chatbot/
├── data/                  # Kaggle dataset
├── models/               # Trained models and embeddings
├── src/
│   ├── data_loader.py   # Load and process CSV data
│   ├── chunker.py       # Break products into chunks
│   ├── embedder.py      # Create embeddings with ChromaDB
│   └── rag_pipeline.py  # LangChain RAG logic
├── app.py               # Streamlit UI
├── requirements.txt     # Python dependencies
└── README.md           
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

This project:
1. Data chunking strategies
2. Vector embeddings and cosine similarity
3. RAG architecture
4. LLM prompt engineering
5. Full-stack ML deployment
