"""
Aether-Store-SQL FastAPI Application

This module implements the REST API for semantic product search.

Author: Senior Data Architect
Date: February 4, 2026
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from loguru import logger
import os
from dotenv import load_dotenv

from src.api.search_engine import VectorSearchEngine, SearchResult


# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Aether-Store-SQL API",
    description="Semantic product search API using SQL Server 2025 Vector Search",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize search engine
search_engine = VectorSearchEngine()


# Pydantic models
class ProductResponse(BaseModel):
    """Product search result response model."""
    product_id: int
    sku: str
    product_name: str
    product_description: str
    category: str
    brand: str
    price: float
    stock_quantity: int
    similarity_score: float = Field(ge=0.0, le=1.0)


class SearchResponse(BaseModel):
    """Search API response model."""
    query: str
    result_count: int
    results: List[ProductResponse]


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    version: str
    database: str


# API Endpoints
@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to Aether-Store-SQL API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        database="connected"
    )


@app.get("/search", response_model=SearchResponse)
async def search_products(
    q: str = Query(..., description="Search query", min_length=1),
    top_k: int = Query(10, ge=1, le=100, description="Number of results"),
    min_score: float = Query(0.7, ge=0.0, le=1.0, description="Minimum similarity score")
):
    """
    Semantic product search endpoint.
    
    Args:
        q: Natural language search query
        top_k: Number of results to return
        min_score: Minimum similarity score threshold
        
    Returns:
        SearchResponse with matching products
    """
    try:
        logger.info(f"Search request: query='{q}', top_k={top_k}, min_score={min_score}")
        
        results = search_engine.search(
            query=q,
            top_k=top_k,
            min_score=min_score
        )
        
        products = [
            ProductResponse(
                product_id=r.product_id,
                sku=r.sku,
                product_name=r.product_name,
                product_description=r.product_description,
                category=r.category,
                brand=r.brand,
                price=r.price,
                stock_quantity=r.stock_quantity,
                similarity_score=r.similarity_score
            )
            for r in results
        ]
        
        return SearchResponse(
            query=q,
            result_count=len(products),
            results=products
        )
    
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search/category/{category}", response_model=SearchResponse)
async def search_by_category(
    category: str,
    q: str = Query(..., description="Search query"),
    top_k: int = Query(10, ge=1, le=100)
):
    """
    Search products within a specific category.
    
    Args:
        category: Product category
        q: Search query
        top_k: Number of results
        
    Returns:
        SearchResponse with filtered results
    """
    try:
        results = search_engine.search_by_category(
            query=q,
            category=category,
            top_k=top_k
        )
        
        products = [
            ProductResponse(
                product_id=r.product_id,
                sku=r.sku,
                product_name=r.product_name,
                product_description=r.product_description,
                category=r.category,
                brand=r.brand,
                price=r.price,
                stock_quantity=r.stock_quantity,
                similarity_score=r.similarity_score
            )
            for r in results
        ]
        
        return SearchResponse(
            query=q,
            result_count=len(products),
            results=products
        )
    
    except Exception as e:
        logger.error(f"Category search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """Run the application."""
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    
    uvicorn.run(
        "src.api.app:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()
