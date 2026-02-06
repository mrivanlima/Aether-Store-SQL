import os
import pyodbc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from loguru import logger
from src.ingest.vectorizer import ProductVectorizer

# Initialize the App
app = FastAPI(title="Aether Store AI API", version="1.0.0")

# Global Vectorizer (Load once at startup to save time)
vectorizer = None

@app.on_event("startup")
async def startup_event():
    global vectorizer
    try:
        logger.info("Initializing OpenAI Vectorizer...")
        vectorizer = ProductVectorizer()
        logger.success("Vectorizer ready.")
    except Exception as e:
        logger.error(f"Failed to start vectorizer: {e}")

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

# --- API MODELS ---
class SearchResult(BaseModel):
    id: str
    title: str
    price: float
    distance: float

# --- ENDPOINTS ---
@app.get("/")
def health_check():
    return {"status": "online", "engine": "SQL Server 2025"}

@app.get("/search", response_model=List[SearchResult])
def search_products(q: str):
    """
    Semantic Search Endpoint.
    1. Vectorizes the query 'q'
    2. Compares it against database vectors
    3. Returns top 5 matches
    """
    if not q:
        raise HTTPException(status_code=400, detail="Query string 'q' is required")

    global vectorizer
    try:
        # 1. Turn User Query into Math
        query_vector = vectorizer.generate_embedding(q)
        
        # 2. Run the SQL 2025 Vector Search
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Note: We use the Double Cast trick again for safety
            sql = """
            SELECT TOP 5
                ProductId,
                Title,
                Price,
                VECTOR_DISTANCE('cosine', CAST(CAST(? AS NVARCHAR(MAX)) AS VECTOR(1536)), ProductVector) AS Distance
            FROM catalog.Products
            ORDER BY Distance ASC
            """
            
            cursor.execute(sql, str(query_vector))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    "id": row.ProductId,
                    "title": row.Title,
                    "price": float(row.Price),
                    "distance": float(row.Distance)
                })
                
            return results

    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))