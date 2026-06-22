import os
import json
import re
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

load_dotenv()

class RAGPipeline:
    def __init__(self, data_path: str = 'data/chunks.jsonl'):
        # 1. Initialize embedding model
        print("Loading local embedding model...")
        self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        
        # 2. Connect to Qdrant Cloud
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_key = os.getenv("QDRANT_API_KEY")
        print(f"Connecting to Qdrant at {qdrant_url}...")
        self.qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_key, timeout=60)
        self.collection_name = "products"
        
        # Ensure payload indexes exist for numerical filters (fixes 400 Bad Request range filter error)
        try:
            self.qdrant_client.create_payload_index(
                collection_name=self.collection_name,
                field_name="price",
                field_schema=qmodels.PayloadSchemaType.FLOAT
            )
            self.qdrant_client.create_payload_index(
                collection_name=self.collection_name,
                field_name="stars",
                field_schema=qmodels.PayloadSchemaType.FLOAT
            )
            self.qdrant_client.create_payload_index(
                collection_name=self.collection_name,
                field_name="category",
                field_schema=qmodels.PayloadSchemaType.KEYWORD
            )
            print("Verified payload indexes for 'price', 'stars', and 'category'.")
        except Exception as e:
            print(f"Note: Payload index check completed: {e}")
        
        # 3. Connect to Groq
        groq_key = os.getenv("GROQ_API_KEY")
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            groq_api_key=groq_key,
            temperature=0.2
        )

        # 4. Load valid categories for LLM prompt
        self.valid_categories = self._load_categories(data_path)

    def _load_categories(self, data_path: str) -> list:
        categories = set()
        try:
            import json
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        cat = json.loads(line).get('metadata', {}).get('category')
                        if cat:
                            categories.add(cat)
        except Exception as e:
            print(f"Warning: Could not load categories from chunks.jsonl: {e}")
        return sorted(list(categories))
        
    def preprocess_query(self, user_query: str, chat_history: list = None) -> dict:
        """
        Query Preprocessor
        Uses the LLM to parse the query, resolve conversational history context, and extract filters.
        """
        categories_str = ", ".join([f'"{c}"' for c in self.valid_categories])
        
        history_lines = []
        if chat_history:
            for msg in chat_history[-4:]:  # last 4 messages (2 turns)
                role = "User" if msg["role"] == "user" else "Assistant"
                content = msg["content"]
                if len(content) > 300:
                    content = content[:300] + "..."
                history_lines.append(f"{role}: {content}")
        
        history_context = ""
        if history_lines:
            history_context = "Conversation History:\n" + "\n".join(history_lines) + "\n\n"
            
        system_prompt = f"""You are an expert shopping assistant. Parse the user's shopping search query considering the recent Conversation History (if any).
If the user's query is a follow-up question specifically asking for more details, comparison, or clarification about the *previously recommended/retrieved products* (e.g. "tell me more about them", "compare these 3", "chi tiết hơn về 3 món đồ trên"), set "use_previous_context" to true.
Otherwise, if the user is searching for new products or changing the search criteria significantly, set "use_previous_context" to false.

Output a raw JSON object (WITHOUT any markdown formatting or code blocks) with the following keys:
- "clean_query": The core product terms for vector search, translated to English if necessary. If "use_previous_context" is true, this can be empty or a simple summary of the topic.
- "language": The language of the query (e.g. "Vietnamese", "English").
- "price_max": The maximum price float if specified or implied (null if not). For phrases like "cheap/giá rẻ", use 50.0. If a price filter was previously specified and the user is still discussing those items, inherit or adapt that limit.
- "stars_min": The minimum star rating float if specified or implied (null if not). For phrases like "best/highly rated", use 4.0. If previously specified, inherit or adapt that limit.
- "target_category": The specific product category to filter by if implied (must be from Valid Categories, or null).
- "use_previous_context": A boolean (true or false) indicating whether this query is a follow-up about the *previously retrieved products* without introducing new search terms.

Valid Categories: [{categories_str}]

Example: "luggage under $150 with at least 4 stars"
Output: {{"clean_query": "luggage", "language": "English", "price_max": 150.0, "stars_min": 4.0, "target_category": "Luggage", "use_previous_context": false}}

Example: "vali kéo giá rẻ màu đỏ"
Output: {{"clean_query": "red suitcase", "language": "Vietnamese", "price_max": 50.0, "stars_min": null, "target_category": "Luggage", "use_previous_context": false}}

Example: "cho tôi thêm thông tin về các sản phẩm trên" (when previous turn recommended suitcases)
Output: {{"clean_query": "", "language": "Vietnamese", "price_max": null, "stars_min": null, "target_category": null, "use_previous_context": true}}
"""
        # Ask LLM to parse the query and extract metadata
        user_input_with_history = f"{history_context}Current User Query: {user_query}"
        response = self.llm.invoke([
            ("system", system_prompt),
            ("user", user_input_with_history)
        ])
        
        raw_text = response.content.strip()
        # Clean markdown ticks if any
        if raw_text.startswith("```"):
            raw_text = re.sub(r'^```[a-zA-Z]*\n', '', raw_text)
            raw_text = re.sub(r'\n```$', '', raw_text).strip()
            
        # Find JSON block using regex to avoid parsing issues if the LLM output includes extra conversational text
        json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if json_match:
            raw_text = json_match.group(0)
            
        try:
            parsed = json.loads(raw_text)
            print(f"Preprocessed Query Details: {parsed}")
            return parsed
        except Exception as e:
            print(f"Failed to parse query JSON. Raw text: {raw_text}. Error: {e}")
            # Determine language based on simple unicode heuristic
            is_vietnamese = any(ord(c) > 127 for c in user_query)
            return {
                "clean_query": user_query,
                "language": "Vietnamese" if is_vietnamese else "English",
                "price_max": None,
                "stars_min": None,
                "target_category": None,
                "use_previous_context": False
            }

    def build_qdrant_filter(self, parsed_query: dict) -> qmodels.Filter:
        """
        Component 2: Qdrant Filter Builder
        Converts parsed metadata filters into Qdrant filter conditions.
        """
        conditions = []
        
        # Target Category Filter
        if parsed_query.get("target_category") is not None:
            conditions.append(
                qmodels.FieldCondition(
                    key="category",
                    match=qmodels.MatchValue(value=parsed_query["target_category"])
                )
            )

        # Max Price Filter
        if parsed_query.get("price_max") is not None:
            conditions.append(
                qmodels.FieldCondition(
                    key="price",
                    range=qmodels.Range(lte=float(parsed_query["price_max"]))
                )
            )
            
        # Min Stars Filter
        if parsed_query.get("stars_min") is not None:
            conditions.append(
                qmodels.FieldCondition(
                    key="stars",
                    range=qmodels.Range(gte=float(parsed_query["stars_min"]))
                )
            )
            
        if conditions:
            return qmodels.Filter(must=conditions)
        return None

    def search_products(self, clean_query: str, qdrant_filter: qmodels.Filter = None, top_k: int = 5) -> list:
        """
        Component 3: Hybrid Search
        Vector Search + Metadata filtering
        """
        # Embed query locally
        query_vector = self.embedding_model.encode(clean_query).tolist()
        
        # Execute search in Qdrant with backward compatibility for older clients
        if hasattr(self.qdrant_client, 'query_points'):
            response = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=qdrant_filter,
                limit=top_k
            )
            return response.points
        else:
            return self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=qdrant_filter,
                limit=top_k
            )

    def answer_question(self, user_query: str, chat_history: list = None, previous_docs: list = None) -> tuple:
        """
        Orchestrate the full RAG pipeline
        """
        # 1. Preprocess Query
        parsed_query = self.preprocess_query(user_query, chat_history)
        clean_query = parsed_query["clean_query"]
        use_previous = parsed_query.get("use_previous_context", False)
        
        # 3. Retrieve Context
        if use_previous and previous_docs:
            top_results = previous_docs
            print(f"Reusing previous docs context ({len(top_results)} items).")
        else:
            # 2. Build Filter
            qdrant_filter = self.build_qdrant_filter(parsed_query)
            search_results = self.search_products(clean_query, qdrant_filter, top_k=5)
            
            # Sort results in Python:
            # 1. By stars rating (descending)
            # 2. By vector similarity score (descending)
            sorted_results = sorted(
                search_results,
                key=lambda x: (
                    float(x.payload.get("stars")) if x.payload.get("stars") is not None else 0.0,
                    x.score
                ),
                reverse=True
            )
            
            # Take top 3 best products
            top_results = sorted_results[:3]
        
        # 4. Construct Context Text
        context_items = []
        for i, res in enumerate(top_results):
            payload = res.payload
            item_text = (
                f"Product {i+1}:\n"
                f"Title: {payload.get('text')}\n"
                f"Price: ${payload.get('price')}\n"
                f"Rating: {payload.get('stars')} stars\n"
                f"Category: {payload.get('category')}\n"
            )
            context_items.append(item_text)
            
        context = "\n---\n".join(context_items) if context_items else "No matching products found."

        
        history_lines = []
        if chat_history:
            for msg in chat_history[-4:]:
                role = "User" if msg["role"] == "user" else "Assistant"
                content = msg["content"]
                if len(content) > 300:
                    content = content[:300] + "..."
                history_lines.append(f"{role}: {content}")
        history_context = "\n".join(history_lines) if history_lines else "None"

        # 5. Build Prompt
        prompt_template = PromptTemplate(
            input_variables=["context", "question", "language", "chat_history"],
            template="""You are a professional shopping assistant for an online store.
Your goal is to answer the user's question about products based ONLY on the catalog context provided below.

Respond in the user's language: {language}.

Instructions:
1. Use the provided context to answer questions about prices, ratings, and features.
2. If the user asks for suggestions, recommend the most relevant products from the context.
3. If the context does not contain enough information to answer the question, politely tell the user that you don't have that information in the catalog.
4. Consider the Conversation History below if the user's question is a follow-up.
5. At the very end of your response, write a tag list containing only the indices of the products (e.g. 1, 2, 3) that are relevant to the user's query and match their criteria. Format it exactly as: [RELEVANT_PRODUCTS: 1, 2]
If none of the products match the criteria, format it exactly as: [RELEVANT_PRODUCTS: None]

Conversation History:
{chat_history}

Catalog Context:
{context}

User's Question: {question}

Assistant Answer:"""
        )
        
        # LangChain Chain Execution
        chain = prompt_template | self.llm
        
        # 6. Generate Response
        response = chain.invoke({
            "context": context,
            "question": user_query,
            "language": parsed_query.get("language", "English"),
            "chat_history": history_context
        })
        
        raw_answer = response.content
        
        # Regex to find [RELEVANT_PRODUCTS: 1, 2, 3] or [RELEVANT_PRODUCTS: None]
        tag_match = re.search(r'\[RELEVANT_PRODUCTS:\s*([^\]]+)\]', raw_answer)
        
        relevant_docs = top_results
        cleaned_answer = raw_answer
        
        if tag_match:
            tag_content = tag_match.group(1).strip()
            # Remove the tag from the answer so the user doesn't see it in the UI
            cleaned_answer = re.sub(r'\[RELEVANT_PRODUCTS:\s*[^\]]+\]', '', raw_answer).strip()
            
            if tag_content.lower() != 'none':
                try:
                    # Split by comma, strip, convert to int, and subtract 1 for 0-based indexing
                    indices = [int(idx.strip()) - 1 for idx in tag_content.split(',') if idx.strip().isdigit()]
                    # Filter top_results
                    relevant_docs = [top_results[i] for i in indices if 0 <= i < len(top_results)]
                except Exception as e:
                    print(f"Error parsing relevant product indices: {e}")
            else:
                relevant_docs = []
                
        return cleaned_answer, relevant_docs
