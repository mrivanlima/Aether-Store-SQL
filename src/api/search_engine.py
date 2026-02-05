"""
Vector Search Engine

This module implements the semantic search functionality using SQL Server
vector similarity operations.

Author: Senior Data Architect
Date: February 4, 2026
"""

import pyodbc
from typing import List, Dict, Any, Optional
from loguru import logger
from dotenv import load_dotenv
import os
from dataclasses import dataclass

from src.ingest.vectorizer import ProductVectorizer


@dataclass
class SearchResult:
    """Represents a single search result."""
    product_id: int
    sku: str
    product_name: str
    product_description: str
    category: str
    brand: str
    price: float
    stock_quantity: int
    similarity_score: float


class VectorSearchEngine:
    """
    Semantic search engine using SQL Server vector similarity.
    
    Attributes:
        connection_string: SQL Server connection string
        vectorizer: ProductVectorizer for query embedding
    """
    
    def __init__(self, connection_string: Optional[str] = None):
        """
        Initialize the VectorSearchEngine.
        
        Args:
            connection_string: SQL Server connection string
        """
        load_dotenv()
        
        if connection_string:
            self.connection_string = connection_string
        else:
            self.connection_string = self._build_connection_string()
        
        self.vectorizer = ProductVectorizer()
        logger.info("VectorSearchEngine initialized")
    
    def _build_connection_string(self) -> str:
        """Build connection string from environment variables."""
        host = os.getenv("MSSQL_HOST", "localhost")
        port = os.getenv("MSSQL_PORT", "1433")
        user = os.getenv("MSSQL_USER", "sa")
        password = os.getenv("MSSQL_PASSWORD")
        database = os.getenv("MSSQL_DATABASE", "AetherStoreDB")
        
        return (
            f"DRIVER={{ODBC Driver 18 for SQL Server}};"
            f"SERVER={host},{port};"
            f"DATABASE={database};"
            f"UID={user};"
            f"PWD={password};"
            f"TrustServerCertificate=yes;"
        )
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.7
    ) -> List[SearchResult]:
        """
        Perform semantic search for products.
        
        Args:
            query: Natural language search query
            top_k: Number of results to return
            min_score: Minimum similarity score threshold
            
        Returns:
            List of SearchResult objects
        """
        logger.info(f"Searching for: {query} (top_k={top_k}, min_score={min_score})")
        
        # Generate query embedding
        query_embedding = self.vectorizer.generate_embedding(query)
        vector_string = f"[{','.join(map(str, query_embedding))}]"
        
        # Execute search
        conn = pyodbc.connect(self.connection_string)
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                "EXEC dbo.usp_SearchProductsByVector @QueryEmbedding=?, @TopK=?, @MinScore=?",
                vector_string,
                top_k,
                min_score
            )
            
            results = []
            for row in cursor.fetchall():
                result = SearchResult(
                    product_id=row[0],
                    sku=row[1],
                    product_name=row[2],
                    product_description=row[3],
                    category=row[4],
                    brand=row[5],
                    price=float(row[6]),
                    stock_quantity=row[7],
                    similarity_score=float(row[8])
                )
                results.append(result)
            
            logger.info(f"Found {len(results)} results")
            return results
        
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
        
        finally:
            cursor.close()
            conn.close()
    
    def search_by_category(
        self,
        query: str,
        category: str,
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Search within a specific category.
        
        Args:
            query: Search query
            category: Product category filter
            top_k: Number of results
            
        Returns:
            List of filtered search results
        """
        all_results = self.search(query, top_k=top_k * 2, min_score=0.0)
        filtered = [r for r in all_results if r.category == category]
        return filtered[:top_k]
