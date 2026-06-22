import os
import json
import uuid
import time
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not QDRANT_URL or not QDRANT_API_KEY:
    raise ValueError("Missing QDRANT_URL or QDRANT_API_KEY in the environment variables.")

def index_data():
    # 1. Load the embedding model
    print("Loading embedding model")
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    
    # 2. Connect to Qdrant Cloud
    print(f"Connecting to Qdrant Cloud at: {QDRANT_URL}")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=600)
    
    # 3. Create collection if it doesn't exist
    collection_name = "products"
    print(f"Recreating collection '{collection_name}' in Qdrant (dimension: 384, distance: Cosine)...")
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

    # 4. Load chunks from chunks.jsonl
    chunks_file = 'data/chunks.jsonl'
    if not os.path.exists(chunks_file):
        raise FileNotFoundError(f"Could not find chunks file at {chunks_file}. Run chunker first.")
        
    print(f"Reading chunks from {chunks_file}...")
    chunks = []
    with open(chunks_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
                
    total_chunks = len(chunks)
    print(f"Loaded {total_chunks} chunks to process.")
    
    # 5. Process and upload in batches
    batch_size = 256
    print(f"Starting upload in batches of {batch_size}...")
    
    start_time = time.time()
    
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i : i + batch_size]
        
        # Check if this batch is already indexed
        last_item_id = str(uuid.UUID(hex=batch[-1]['id']))
        try:
            existing = client.retrieve(collection_name=collection_name, ids=[last_item_id])
            if existing:
                processed = min(i + batch_size, total_chunks)
                print(f"Skipped batch {i // batch_size + 1}/{total_chunks // batch_size + 1} (already indexed up to chunk {processed})")
                continue
        except Exception as e:
            print(f"Warning: Could not check if batch was already indexed: {e}. Attempting upload anyway.")
        
        # Extract texts to embed
        texts = [item['text'] for item in batch]
        
        # Generate embeddings locally
        embeddings = model.encode(texts, show_progress_bar=False)
        
        # Prepare points for Qdrant
        points = []
        for idx, item in enumerate(batch):
            # Convert MD5 string ID to a valid UUID format
            point_id = str(uuid.UUID(hex=item['id']))
            vector = embeddings[idx].tolist()
            payload = item['metadata']
            # Store the raw product text in payload so we can retrieve it
            payload['text'] = item['text']
            
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                )
            )

        max_retries = 5
        for attempt in range(max_retries):
            try:
                client.upsert(
                    collection_name=collection_name,
                    points=points
                )
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Error: Failed to upload batch after {max_retries} attempts.")
                    raise e
                wait_time = 2 ** attempt
                print(f"Upload failed: {e}. Retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
        
        elapsed = time.time() - start_time
        processed = min(i + batch_size, total_chunks)
        speed = processed / elapsed if elapsed > 0 else 0
        print(f"Uploaded {processed}/{total_chunks} chunks ({processed/total_chunks*100:.1f}%) - Speed: {speed:.1f} chunks/sec - Time elapsed: {elapsed:.1f}s")

    total_time = time.time() - start_time
    print(f"\nSuccessfully finished! Indexed {total_chunks} chunks in {total_time:.2f} seconds.")

if __name__ == "__main__":
    index_data()
