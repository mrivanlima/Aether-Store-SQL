import os
import pyodbc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from loguru import logger
from openai import OpenAI
from src.ingest.vectorizer import VectorizerFactory

app = FastAPI(title="Aether Store AI API", version="1.2.0")

# --- GLOBAL COMPONENTS ---
vectorizer = None
llm_client = None
llm_model_name = "gpt-4o-mini"

@app.on_event("startup")
async def startup_event():
    global vectorizer, llm_client, llm_model_name
    try:
        logger.info("Initializing Vectorizer...")
        vectorizer = VectorizerFactory()
        
        llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
        if llm_provider == "local":
            logger.info("Chat Mode: LOCAL (Ollama)")
            llm_client = OpenAI(
                base_url="http://host.docker.internal:11434/v1",
                api_key="ollama"
            )
            llm_model_name = "llama3.2" 
        else:
            logger.info("Chat Mode: CLOUD (OpenAI)")
            llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            llm_model_name = "gpt-4o-mini"
            
        logger.success(f"System Ready. Vectors: {os.getenv('VECTOR_PROVIDER')} | Chat: {llm_provider.upper()}")
    except Exception as e:
        logger.error(f"Failed to start system: {e}")

def get_db_connection():
    conn_str = (
        f"DRIVER={{ODBC Driver 18 for SQL Server}};"
        f"SERVER={os.getenv('SQL_SERVER')};"
        f"DATABASE={os.getenv('SQL_DATABASE')};"
        f"UID={os.getenv('SQL_USER')};"
        f"PWD={os.getenv('SQL_PASSWORD')};"
        "TrustServerCertificate=yes;"
    )
    return pyodbc.connect(conn_str)

class SearchResult(BaseModel):
    id: str
    title: str
    price: float
    distance: float

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[SearchResult]

def perform_vector_search(query: str, limit: int = 5):
    global vectorizer
    query_vector = vectorizer.generate_embedding(query)
    
    if len(query_vector) == 1536:
        target_col = "ProductVector"
        cast_type = "VECTOR(1536)"
    elif len(query_vector) == 384:
        target_col = "ProductVector_Local"
        cast_type = "VECTOR(384)"
    else:
        raise ValueError(f"Unknown vector dimension: {len(query_vector)}")

    with get_db_connection() as conn:
        cursor = conn.cursor()
        sql = f"""
        SELECT TOP {limit}
            ProductId, Title, Price, Description,
            VECTOR_DISTANCE('cosine', CAST(CAST(? AS NVARCHAR(MAX)) AS {cast_type}), {target_col}) AS Distance
        FROM catalog.Products
        WHERE {target_col} IS NOT NULL
        ORDER BY Distance ASC
        """
        cursor.execute(sql, str(query_vector))
        return [{"id": row.ProductId, "title": row.Title, "price": float(row.Price), "description": row.Description, "distance": float(row.Distance)} for row in cursor.fetchall()]

@app.get("/search", response_model=List[SearchResult])
def search_products(q: str):
    results = perform_vector_search(q)
    return [{"id": r["id"], "title": r["title"], "price": r["price"], "distance": r["distance"]} for r in results]

@app.post("/chat", response_model=ChatResponse)
def chat_with_store(request: ChatRequest):
    global llm_client, llm_model_name
    if not llm_client:
        raise HTTPException(status_code=503, detail="Chat capability unavailable.")

    # 1. Retrieve
    products = perform_vector_search(request.question, limit=5)
    if not products:
        return {"answer": "No products found.", "sources": []}

    # 2. Generate
    logger.info("Generating Answer via LLM...")
    context_text = "\n".join([f"- {p['title']} (${p['price']})" for p in products])
    
    response = llm_client.chat.completions.create(
        model=llm_model_name,
        messages=[
            {"role": "system", "content": "You are a helpful sales assistant."},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {request.question}"}
        ],
        temperature=0.7
    )
    
    return {
        "answer": response.choices[0].message.content,
        "sources": [{"id": p["id"], "title": p["title"], "price": p["price"], "distance": p["distance"]} for p in products]
    }