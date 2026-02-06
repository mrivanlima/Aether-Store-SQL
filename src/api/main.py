import os
import pyodbc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from loguru import logger
# --- CHANGED: Import the Factory ---
from src.ingest.vectorizer import VectorizerFactory

# Initialize the App
app = FastAPI(title="Aether Store AI API", version="1.0.0")

# Global Vectorizer
vectorizer = None

@app.on_event("startup")
async def startup_event():
    global vectorizer
    try:
        logger.info("Initializing Vectorizer Strategy...")
        # --- CHANGED: Use the Factory ---
        vectorizer = VectorizerFactory()
        logger.success(f"Vectorizer ready. Mode: {os.getenv('VECTOR_PROVIDER', 'openai')}")
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
    Dynamically selects the correct vector column (OpenAI vs Local) based on the vector size.
    """
    if not q:
        raise HTTPException(status_code=400, detail="Query string 'q' is required")

    global vectorizer
    try:
        # 1. Generate Vector (Factory handles logic)
        query_vector = vectorizer.generate_embedding(q)
        
        # 2. Determine which column to search based on vector size
        # This is the "Dual Brain" routing logic
        if len(query_vector) == 1536:
            target_col = "ProductVector"
            cast_type = "VECTOR(1536)"
        elif len(query_vector) == 384:
            target_col = "ProductVector_Local"
            cast_type = "VECTOR(384)"
        else:
            raise ValueError(f"Unknown vector dimension: {len(query_vector)}")

        # 3. Dynamic Search Query
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # We inject target_col safely because we select it from a hardcoded list above
            sql = f"""
            SELECT TOP 5
                ProductId,
                Title,
                Price,
                VECTOR_DISTANCE('cosine', CAST(CAST(? AS NVARCHAR(MAX)) AS {cast_type}), {target_col}) AS Distance
            FROM catalog.Products
            WHERE {target_col} IS NOT NULL
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