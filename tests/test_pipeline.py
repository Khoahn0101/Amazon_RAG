import pytest
from rag_pipeline import RAGPipeline

def test_answer_question():
    # Requires .env and Qdrant/Groq keys, and data/chunks.jsonl present
    pipeline = RAGPipeline(data_path='data/chunks.jsonl')
    query = "Show me clothe for my daughter under $100 with at least 4 stars"
    print(f"\nTesting Query: '{query}'")
    answer, docs = pipeline.answer_question(query)
    
    assert answer is not None
    assert len(docs) <= 3
    
    print("\n--- Answer ---")
    print(answer)
    print("\n--- Retrieved Sources ---")
    for doc in docs:
        print(f"- {doc.payload.get('text')} [Price: ${doc.payload.get('price')}, Rating: {doc.payload.get('stars')} stars]")
