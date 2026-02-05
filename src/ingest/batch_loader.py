"""
Batch Product Data Loader

This module handles the ingestion of product data from various sources
and loading into SQL Server with vector embeddings.

Author: Senior Data Architect
Date: February 4, 2026
"""

import json
import pandas as pd
import pyodbc
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger
from dotenv import load_dotenv
import os

from src.ingest.vectorizer import ProductVectorizer


class BatchProductLoader:
    """
    Handles batch loading of product data into SQL Server with vector embeddings.
    
    Attributes:
        connection_string: SQL Server connection string
        vectorizer: ProductVectorizer instance for generating embeddings
    """
    
    def __init__(self, connection_string: Optional[str] = None):
        """
        Initialize the BatchProductLoader.
        
        Args:
            connection_string: SQL Server connection string (defaults to env variables)
        """
        load_dotenv()
        
        if connection_string:
            self.connection_string = connection_string
        else:
            self.connection_string = self._build_connection_string()
        
        self.vectorizer = ProductVectorizer()
        logger.info("BatchProductLoader initialized")
    
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
    
    def load_from_json(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Load product data from JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            List of product dictionaries
        """
        logger.info(f"Loading data from: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} products from JSON")
        return data
    
    def load_from_csv(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Load product data from CSV file.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            List of product dictionaries
        """
        logger.info(f"Loading data from: {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} products from CSV")
        return df.to_dict('records')
    
    def insert_products(self, products: List[Dict[str, Any]]) -> int:
        """
        Insert products with embeddings into SQL Server.
        
        Args:
            products: List of product dictionaries
            
        Returns:
            Number of products inserted
        """
        logger.info(f"Starting batch insert for {len(products)} products")
        
        # Prepare texts for vectorization
        texts = [
            self.vectorizer.prepare_product_text(
                name=p.get('name', ''),
                description=p.get('description', ''),
                category=p.get('category', ''),
                brand=p.get('brand', '')
            )
            for p in products
        ]
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = self.vectorizer.generate_batch_embeddings(texts)
        
        # Insert into database
        inserted_count = 0
        conn = pyodbc.connect(self.connection_string)
        cursor = conn.cursor()
        
        insert_query = """
        INSERT INTO dbo.Products (
            SKU, ProductName, ProductDescription, Category, Brand, 
            Price, StockQuantity, ProductEmbedding
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        try:
            for product, embedding in zip(products, embeddings):
                vector_string = f"[{','.join(map(str, embedding))}]"
                
                cursor.execute(
                    insert_query,
                    product.get('sku'),
                    product.get('name'),
                    product.get('description'),
                    product.get('category'),
                    product.get('brand'),
                    product.get('price'),
                    product.get('stock_quantity', 0),
                    vector_string
                )
                inserted_count += 1
            
            conn.commit()
            logger.info(f"Successfully inserted {inserted_count} products")
        
        except Exception as e:
            conn.rollback()
            logger.error(f"Error inserting products: {e}")
            raise
        
        finally:
            cursor.close()
            conn.close()
        
        return inserted_count


def main():
    """Main entry point for batch loading."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch load product data')
    parser.add_argument('--input', required=True, help='Input file path (JSON or CSV)')
    args = parser.parse_args()
    
    loader = BatchProductLoader()
    
    file_path = Path(args.input)
    if file_path.suffix == '.json':
        products = loader.load_from_json(file_path)
    elif file_path.suffix == '.csv':
        products = loader.load_from_csv(file_path)
    else:
        raise ValueError("Unsupported file format. Use JSON or CSV.")
    
    loader.insert_products(products)


if __name__ == "__main__":
    main()
