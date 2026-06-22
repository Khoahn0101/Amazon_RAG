import streamlit as st
import json
import random
from src.rag_pipeline import RAGPipeline

st.set_page_config(page_title="Amazon Shopping Assistant", page_icon="🛍️", layout="wide")

# Custom CSS for modern, premium look
st.markdown("""
<style>
    /* Global background and font */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
        font-family: 'Inter', sans-serif;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #FFFFFF;
        font-weight: 600;
    }
    
    /* Product Card Styling */
    .product-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 16px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 100%;
    }
    .product-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
        border-color: rgba(255, 255, 255, 0.2);
    }
    .product-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 8px;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    .product-price {
        font-size: 1.25rem;
        font-weight: 700;
        color: #4CAF50;
        margin-bottom: 4px;
    }
    .product-stars {
        color: #FFC107;
        font-size: 0.9rem;
        margin-bottom: 12px;
    }
    .product-btn {
        background-color: #2563EB;
        color: white;
        text-decoration: none;
        padding: 8px 16px;
        border-radius: 6px;
        text-align: center;
        font-weight: 500;
        display: inline-block;
        width: 100%;
        transition: background-color 0.2s;
    }
    .product-btn:hover {
        background-color: #1D4ED8;
        color: white;
    }
    .img-container {
        width: 100%;
        height: 180px;
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 12px;
        border-radius: 8px;
        overflow: hidden;
        background: white;
    }
    .img-container img {
        max-width: 100%;
        max-height: 100%;
        object-fit: contain;
    }
    
    /* Showcase Carousel Styling */
    .showcase-container {
        padding: 20px;
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.1), rgba(147, 51, 234, 0.1));
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.05);
        margin-bottom: 24px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_pipeline():
    return RAGPipeline(data_path="data/chunks.jsonl")

@st.cache_data
def load_showcase_products(data_path="data/chunks.jsonl", num_products=4):
    products = []
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    chunk = json.loads(line)
                    meta = chunk.get('metadata', {})
                    if meta.get('imgUrl') and meta.get('productURL') and 'http' in meta.get('imgUrl'):
                        products.append({
                            'title': chunk.get('text').split('\n')[0].replace('Product Name:', '').strip(),
                            'price': meta.get('price'),
                            'stars': meta.get('stars'),
                            'imgUrl': meta.get('imgUrl'),
                            'productURL': meta.get('productURL')
                        })
    except Exception as e:
        print(f"Error loading showcase: {e}")
    
    if len(products) > num_products:
        return random.sample(products, num_products)
    return products

def render_product_card(product):
    st.markdown(f"""
        <div class="product-card">
            <div class="img-container">
                <img src="{product.get('imgUrl')}" alt="Product Image">
            </div>
            <div>
                <div class="product-title" title="{product.get('title', 'Unknown Product')}">{product.get('title', 'Unknown Product')}</div>
                <div class="product-price">${product.get('price', 'N/A')}</div>
                <div class="product-stars">★ {product.get('stars', 'N/A')}</div>
            </div>
            <a href="{product.get('productURL')}" target="_blank" class="product-btn">View on Amazon</a>
        </div>
    """, unsafe_allow_html=True)

# Main UI
st.title("🛍️ Amazon Smart Assistant")
st.markdown("Ask me anything to find the best products matching your needs!")

pipeline = load_pipeline()

# Random Showcase Section
st.markdown("<div class='showcase-container'><h3>🌟 Featured Products</h3>", unsafe_allow_html=True)
showcase_products = load_showcase_products()
if showcase_products:
    cols = st.columns(len(showcase_products))
    for i, p in enumerate(showcase_products):
        with cols[i]:
            render_product_card(p)
st.markdown("</div>", unsafe_allow_html=True)

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_retrieved_docs" not in st.session_state:
    st.session_state.last_retrieved_docs = None

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "products" in message:
            cols = st.columns(min(3, len(message["products"])))
            for i, p in enumerate(message["products"][:3]):
                with cols[i]:
                    render_product_card(p)

if prompt := st.chat_input("What are you looking for today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching for the best products..."):
            answer, docs = pipeline.answer_question(
                prompt, 
                chat_history=st.session_state.messages[:-1], 
                previous_docs=st.session_state.last_retrieved_docs
            )
            st.session_state.last_retrieved_docs = docs
            st.markdown(answer)
            
            product_list = []
            for doc in docs:
                payload = doc.payload
                product_list.append({
                    'title': payload.get('text', '').split('\n')[0].replace('Product Name:', '').strip(),
                    'price': payload.get('price'),
                    'stars': payload.get('stars'),
                    'imgUrl': payload.get('imgUrl'),
                    'productURL': payload.get('productURL')
                })
            
            if product_list:
                cols = st.columns(min(3, len(product_list)))
                for i, p in enumerate(product_list[:3]):
                    with cols[i]:
                        render_product_card(p)
                        
    st.session_state.messages.append({"role": "assistant", "content": answer, "products": product_list})
