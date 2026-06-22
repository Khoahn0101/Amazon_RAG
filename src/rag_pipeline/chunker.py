import hashlib
import json

def create_chunks(df):
    """
    Create chunks of text from the DataFrame.
    Args:
        df (pd.DataFrame): The input DataFrame containing product data.
    Returns: List of dicts with 'id', 'text', 'metadata'
    """
    chunks = []
    for row in df.itertuples():
        # Create unique ID for each chunk using a hash of the product name and description
        title = str(row.title or '')
        category = str(row.category_name or '')
        # Product_id to avoid accidental duplicates
        hash_source = f"{title}|{category}"
        chunk_id = hashlib.md5(hash_source.encode('utf-8')).hexdigest()
        stars = str(row.stars or '')
        customer_reviews = str(row.reviews or '')
        current_price = str(row.price or '')
        list_price = str(row.listPrice or '')
        
        chunk_text = f"""\
        Product Name: {title}
        Category: {category}
        Stars: {stars}
        Customer Reviews: {customer_reviews}
        Current Price: {current_price}
        List Price: {list_price}
        """
        metadata = {
            'price': row.price,
            'stars': row.stars,
            'category': row.category_name,
            'imgUrl': str(getattr(row, 'imgUrl', '')),
            'productURL': str(getattr(row, 'productURL', ''))
        }
        chunks.append({
                'id': chunk_id,
                'text': chunk_text,
                'stars': row.stars,
                'metadata': metadata
        })
    return chunks

def save_chunks(chunks, file_path='data/chunks.jsonl'):
    """
    Save chunks to JSONL format (one JSON object per line).
    Args:
        chunks (list): List of chunk dicts with 'id', 'text', 'metadata'
        file_path (str): Path to save the JSONL file
    """
    with open(file_path, 'w') as f:
        for chunk in chunks:
            json.dump(chunk, f)
            f.write('\n')
    print(f"Saved {len(chunks)} chunks to {file_path}")


def load_chunks(file_path='data/chunks.jsonl'):
    """
    Load chunks from JSONL format (one JSON object per line).
    Args:
        file_path (str): Path to the JSONL file     
    Returns:
        list: List of chunk dicts
    """
    chunks = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip(): 
                chunk = json.loads(line)
                chunks.append(chunk)
    print(f"Loaded {len(chunks)} chunks from {file_path}")
    return chunks