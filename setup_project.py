"""
Aether-Store-SQL Project Structure Generator

This script creates the complete directory structure and boilerplate files
for the Aether-Store-SQL RAG system with SQL Server 2025 Vector Search.

Author: Senior Data Architect
Date: February 4, 2026
"""

import os
from pathlib import Path


def create_directory_structure(base_path: Path):
    """Create the project directory structure."""
    directories = [
        "src/database",
        "src/ingest",
        "src/api",
        "data",
        "tests"
    ]
    
    for directory in directories:
        dir_path = base_path / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")


def create_gitignore(base_path: Path):
    """Create .gitignore file with Python and project-specific patterns."""
    content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/
.venv

# Environment Variables
.env

# Data Directory
data/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Jupyter Notebook
.ipynb_checkpoints

# pytest
.pytest_cache/

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# OS
.DS_Store
Thumbs.db
"""
    
    file_path = base_path / ".gitignore"
    file_path.write_text(content, encoding='utf-8')
    print(f"✓ Created: .gitignore")


def create_env_example(base_path: Path):
    """Create .env.example file with template environment variables."""
    content = """# SQL Server Configuration
MSSQL_HOST=localhost
MSSQL_PORT=1433
MSSQL_USER=sa
MSSQL_PASSWORD=YourStrong!Passw0rd
MSSQL_DATABASE=AetherStoreDB

# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=text-embedding-3-small

# Application Configuration
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
"""
    
    file_path = base_path / ".env.example"
    file_path.write_text(content, encoding='utf-8')
    print(f"✓ Created: .env.example")


def create_docker_compose(base_path: Path):
    """Create docker-compose.yml with SQL Server 2025 configuration."""
    content = """version: '3.8'

services:
  sqlserver:
    image: mcr.microsoft.com/mssql/server:2025-latest
    container_name: aether-sql-server
    environment:
      - ACCEPT_EULA=Y
      - MSSQL_SA_PASSWORD=YourStrong!Passw0rd
      - MSSQL_PID=Developer
    ports:
      - "1433:1433"
    volumes:
      - sqlserver_data:/var/opt/mssql
    networks:
      - aether-network
    healthcheck:
      test: ["CMD", "/opt/mssql-tools/bin/sqlcmd", "-S", "localhost", "-U", "sa", "-P", "YourStrong!Passw0rd", "-Q", "SELECT 1"]
      interval: 10s
      timeout: 5s
      retries: 5

  api:
    build: .
    container_name: aether-api
    environment:
      - MSSQL_HOST=sqlserver
      - MSSQL_PORT=1433
      - MSSQL_USER=sa
      - MSSQL_PASSWORD=YourStrong!Passw0rd
      - MSSQL_DATABASE=AetherStoreDB
    ports:
      - "8000:8000"
    depends_on:
      sqlserver:
        condition: service_healthy
    networks:
      - aether-network
    volumes:
      - ./data:/app/data

volumes:
  sqlserver_data:

networks:
  aether-network:
    driver: bridge
"""
    
    file_path = base_path / "docker-compose.yml"
    file_path.write_text(content, encoding='utf-8')
    print(f"✓ Created: docker-compose.yml")


def create_readme(base_path: Path):
    """Create professional README.md with project documentation."""
    content = """# Aether-Store-SQL

A state-of-the-art Retrieval-Augmented Generation (RAG) system for e-commerce, leveraging SQL Server 2025's native Vector Search capabilities for semantic product discovery and intelligent recommendations.

## 🏗️ Architecture

Aether-Store-SQL implements a modern RAG architecture with the following components:

- **Vector Database**: SQL Server 2025 with native VECTOR data type and similarity search
- **Embeddings Engine**: OpenAI text-embedding-3-small for product vectorization
- **Batch Ingestion Pipeline**: Efficient product data processing and vector generation
- **Search API**: FastAPI-based REST interface for semantic search queries
- **Docker Orchestration**: Containerized deployment for reproducibility

### Key Features

- 🔍 **Semantic Search**: Natural language product queries using vector similarity
- 🚀 **High Performance**: Native SQL Server vector operations with optimized indexing
- 📊 **Scalable Ingestion**: Batch processing for large product catalogs
- 🐳 **Docker Ready**: Complete containerized environment
- 🔐 **Production Ready**: Environment-based configuration and security best practices

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.10+**: Required for the application runtime
- **Docker Desktop**: For containerized SQL Server and API deployment
- **SQL Server Management Studio** (optional): For database administration
- **OpenAI API Key**: Required for generating product embeddings

## 🚀 Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Aether-Store-SQL.git
cd Aether-Store-SQL
```

### 2. Configure Environment Variables

Copy the example environment file and configure your settings:

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key and customize other settings as needed.

### 3. Install Python Dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
pip install -r requirements.txt
```

### 4. Start SQL Server with Docker

```bash
docker-compose up -d sqlserver
```

### 5. Initialize Database Schema

```bash
python -m src.database.migrations
```

### 6. Ingest Sample Data

```bash
python -m src.ingest.batch_loader --input data/products.json
```

### 7. Start the API Server

```bash
python -m src.api.app
```

The API will be available at `http://localhost:8000`

## 📚 API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 🧪 Running Tests

```bash
pytest tests/ -v
```

## 📁 Project Structure

```
Aether-Store-SQL/
├── src/
│   ├── database/       # SQL schema and migrations
│   ├── ingest/         # Data ingestion and vectorization
│   └── api/            # FastAPI application and search engine
├── data/               # Sample data and ingested files
├── tests/              # Unit and integration tests
├── docker-compose.yml  # Container orchestration
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- SQL Server 2025 Vector Search capabilities
- OpenAI Embeddings API
- FastAPI framework
"""
    
    file_path = base_path / "README.md"
    file_path.write_text(content, encoding='utf-8')
    print(f"✓ Created: README.md")


def create_requirements_txt(base_path: Path):
    """Create requirements.txt with project dependencies."""
    content = """# Database
pyodbc==5.0.1
pymssql==2.2.11

# Vector Embeddings
openai==1.12.0

# API Framework
fastapi==0.109.2
uvicorn[standard]==0.27.1
pydantic==2.6.1
pydantic-settings==2.1.0

# Data Processing
pandas==2.2.0
numpy==1.26.4

# Environment Management
python-dotenv==1.0.1

# Testing
pytest==8.0.0
pytest-asyncio==0.23.4
httpx==0.26.0

# Utilities
tenacity==8.2.3
loguru==0.7.2
"""
    
    file_path = base_path / "requirements.txt"
    file_path.write_text(content, encoding='utf-8')
    print(f"✓ Created: requirements.txt")


def create_schema_sql(base_path: Path):
    """Create schema.sql with vector-enabled product table."""
    content = """-- =============================================
-- Aether-Store-SQL Database Schema
-- SQL Server 2025 with Vector Search Support
-- =============================================

USE master;
GO

-- Create Database
IF NOT EXISTS (SELECT name FROM sys.databases WHERE name = 'AetherStoreDB')
BEGIN
    CREATE DATABASE AetherStoreDB;
END
GO

USE AetherStoreDB;
GO

-- =============================================
-- Products Table with Vector Column
-- =============================================

IF OBJECT_ID('dbo.Products', 'U') IS NOT NULL
    DROP TABLE dbo.Products;
GO

CREATE TABLE dbo.Products
(
    ProductID INT PRIMARY KEY IDENTITY(1,1),
    SKU NVARCHAR(100) NOT NULL UNIQUE,
    ProductName NVARCHAR(500) NOT NULL,
    ProductDescription NVARCHAR(MAX),
    Category NVARCHAR(200),
    Brand NVARCHAR(200),
    Price DECIMAL(18, 2),
    StockQuantity INT DEFAULT 0,
    
    -- Vector embedding for semantic search (1536 dimensions for text-embedding-3-small)
    ProductEmbedding VECTOR(1536) NOT NULL,
    
    -- Metadata
    CreatedAt DATETIME2 DEFAULT GETUTCDATE(),
    UpdatedAt DATETIME2 DEFAULT GETUTCDATE(),
    IsActive BIT DEFAULT 1,
    
    INDEX IX_Products_Category NONCLUSTERED (Category),
    INDEX IX_Products_Brand NONCLUSTERED (Brand),
    INDEX IX_Products_IsActive NONCLUSTERED (IsActive)
);
GO

-- Create Vector Index for similarity search
CREATE INDEX IX_Products_VectorSearch 
ON dbo.Products(ProductEmbedding)
USING VECTOR;
GO

-- =============================================
-- Search History Table
-- =============================================

IF OBJECT_ID('dbo.SearchHistory', 'U') IS NOT NULL
    DROP TABLE dbo.SearchHistory;
GO

CREATE TABLE dbo.SearchHistory
(
    SearchID BIGINT PRIMARY KEY IDENTITY(1,1),
    SearchQuery NVARCHAR(1000) NOT NULL,
    QueryEmbedding VECTOR(1536),
    ResultCount INT,
    ExecutionTimeMs INT,
    SearchedAt DATETIME2 DEFAULT GETUTCDATE(),
    
    INDEX IX_SearchHistory_SearchedAt NONCLUSTERED (SearchedAt DESC)
);
GO

-- =============================================
-- Stored Procedure: Vector Similarity Search
-- =============================================

IF OBJECT_ID('dbo.usp_SearchProductsByVector', 'P') IS NOT NULL
    DROP PROCEDURE dbo.usp_SearchProductsByVector;
GO

CREATE PROCEDURE dbo.usp_SearchProductsByVector
    @QueryEmbedding VECTOR(1536),
    @TopK INT = 10,
    @MinScore FLOAT = 0.7
AS
BEGIN
    SET NOCOUNT ON;
    
    SELECT TOP (@TopK)
        p.ProductID,
        p.SKU,
        p.ProductName,
        p.ProductDescription,
        p.Category,
        p.Brand,
        p.Price,
        p.StockQuantity,
        VECTOR_DISTANCE('cosine', p.ProductEmbedding, @QueryEmbedding) AS SimilarityScore
    FROM dbo.Products p
    WHERE p.IsActive = 1
        AND VECTOR_DISTANCE('cosine', p.ProductEmbedding, @QueryEmbedding) >= @MinScore
    ORDER BY SimilarityScore DESC;
END
GO

PRINT 'Schema created successfully with vector search support!';
GO
"""
    
    file_path = base_path / "src" / "database" / "schema.sql"
    file_path.write_text(content, encoding='utf-8')
    print(f"✓ Created: src/database/schema.sql")


def create_migrations_sql(base_path: Path):
    """Create migrations.sql for database versioning."""
    content = """-- =============================================
-- Aether-Store-SQL Database Migrations
-- Version Control for Schema Changes
-- =============================================

USE AetherStoreDB;
GO

-- Create Migrations Tracking Table
IF OBJECT_ID('dbo.SchemaMigrations', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.SchemaMigrations
    (
        MigrationID INT PRIMARY KEY IDENTITY(1,1),
        VersionNumber NVARCHAR(50) NOT NULL UNIQUE,
        Description NVARCHAR(500),
        AppliedAt DATETIME2 DEFAULT GETUTCDATE(),
        AppliedBy NVARCHAR(100) DEFAULT SYSTEM_USER
    );
    
    INSERT INTO dbo.SchemaMigrations (VersionNumber, Description)
    VALUES ('1.0.0', 'Initial schema with vector search support');
END
GO

PRINT 'Migration tracking initialized';
GO
"""
    
    file_path = base_path / "src" / "database" / "migrations.sql"
    file_path.write_text(content, encoding='utf-8')
    print(f"✓ Created: src/database/migrations.sql")


def create_vectorizer_py(base_path: Path):
    """Create vectorizer.py for embedding generation."""
    content = '''"""
Vector Embedding Generation Module

This module handles the generation of vector embeddings for product data
using OpenAI's text-embedding-3-small model.

Author: Senior Data Architect
Date: February 4, 2026
"""

import os
from typing import List, Optional
from openai import OpenAI
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential


class ProductVectorizer:
    """
    Generates vector embeddings for product data using OpenAI API.
    
    Attributes:
        client: OpenAI client instance
        model: Name of the embedding model to use
        dimension: Dimension of the output vectors (1536 for text-embedding-3-small)
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        dimension: int = 1536
    ):
        """
        Initialize the ProductVectorizer.
        
        Args:
            api_key: OpenAI API key (defaults to env variable)
            model: Embedding model name
            dimension: Vector dimension
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.dimension = dimension
        
        logger.info(f"Initialized ProductVectorizer with model: {self.model}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for a single text input.
        
        Args:
            text: Input text to vectorize
            
        Returns:
            List of floats representing the embedding vector
            
        Raises:
            Exception: If embedding generation fails after retries
        """
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
                dimensions=self.dimension
            )
            embedding = response.data[0].embedding
            logger.debug(f"Generated embedding for text: {text[:50]}...")
            return embedding
        
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
    
    def generate_batch_embeddings(
        self,
        texts: List[str],
        batch_size: int = 100
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches.
        
        Args:
            texts: List of input texts
            batch_size: Number of texts to process per API call
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.info(f"Processing batch {i // batch_size + 1}: {len(batch)} texts")
            
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                    dimensions=self.dimension
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
            except Exception as e:
                logger.error(f"Batch embedding failed: {e}")
                raise
        
        logger.info(f"Generated {len(embeddings)} embeddings successfully")
        return embeddings
    
    def prepare_product_text(
        self,
        name: str,
        description: str,
        category: str = "",
        brand: str = ""
    ) -> str:
        """
        Prepare product text for embedding generation.
        
        Args:
            name: Product name
            description: Product description
            category: Product category
            brand: Product brand
            
        Returns:
            Formatted text for embedding
        """
        components = [
            f"Product: {name}",
            f"Description: {description}"
        ]
        
        if category:
            components.append(f"Category: {category}")
        if brand:
            components.append(f"Brand: {brand}")
        
        return " | ".join(components)
'''
    
    file_path = base_path / "src" / "ingest" / "vectorizer.py"
    file_path.write_text(content, encoding='utf-8')
    print(f"✓ Created: src/ingest/vectorizer.py")


def create_batch_loader_py(base_path: Path):
    """Create batch_loader.py for data ingestion."""
    content = '''"""
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
'''
    
    file_path = base_path / "src" / "ingest" / "batch_loader.py"
    file_path.write_text(content, encoding='utf-8')
    print(f"✓ Created: src/ingest/batch_loader.py")


def create_search_engine_py(base_path: Path):
    """Create search_engine.py for vector search functionality."""
    content = '''"""
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
'''
    
    file_path = base_path / "src" / "api" / "search_engine.py"
    file_path.write_text(content, encoding='utf-8')
    print(f"✓ Created: src/api/search_engine.py")


def create_app_py(base_path: Path):
    """Create app.py for FastAPI application."""
    content = '''"""
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
'''
    
    file_path = base_path / "src" / "api" / "app.py"
    file_path.write_text(content, encoding='utf-8')
    print(f"✓ Created: src/api/app.py")


def create_test_ingest_py(base_path: Path):
    """Create test_ingest.py for unit tests."""
    content = '''"""
Unit Tests for Ingestion Module

Tests for vectorizer and batch loader functionality.

Author: Senior Data Architect
Date: February 4, 2026
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.ingest.vectorizer import ProductVectorizer
from src.ingest.batch_loader import BatchProductLoader


class TestProductVectorizer:
    """Test cases for ProductVectorizer."""
    
    @patch('src.ingest.vectorizer.OpenAI')
    def test_initialization(self, mock_openai):
        """Test vectorizer initialization."""
        vectorizer = ProductVectorizer(api_key="test_key")
        assert vectorizer.model == "text-embedding-3-small"
        assert vectorizer.dimension == 1536
    
    def test_prepare_product_text(self):
        """Test product text preparation."""
        vectorizer = ProductVectorizer(api_key="test_key")
        
        text = vectorizer.prepare_product_text(
            name="Test Product",
            description="Test Description",
            category="Electronics",
            brand="TestBrand"
        )
        
        assert "Test Product" in text
        assert "Test Description" in text
        assert "Electronics" in text
        assert "TestBrand" in text
    
    @patch('src.ingest.vectorizer.OpenAI')
    def test_generate_embedding_success(self, mock_openai):
        """Test successful embedding generation."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 1536)]
        mock_client.embeddings.create.return_value = mock_response
        
        with patch.object(ProductVectorizer, '__init__', lambda x, **kwargs: None):
            vectorizer = ProductVectorizer()
            vectorizer.client = mock_client
            vectorizer.model = "text-embedding-3-small"
            vectorizer.dimension = 1536
            
            embedding = vectorizer.generate_embedding("test text")
            assert len(embedding) == 1536


class TestBatchProductLoader:
    """Test cases for BatchProductLoader."""
    
    @patch('src.ingest.batch_loader.pyodbc')
    @patch('src.ingest.batch_loader.ProductVectorizer')
    def test_initialization(self, mock_vectorizer, mock_pyodbc):
        """Test loader initialization."""
        loader = BatchProductLoader(connection_string="test_connection")
        assert loader.connection_string == "test_connection"
    
    def test_load_from_json(self, tmp_path):
        """Test loading data from JSON file."""
        import json
        
        # Create temporary JSON file
        test_data = [
            {"sku": "TEST001", "name": "Test Product", "price": 99.99}
        ]
        json_file = tmp_path / "test.json"
        json_file.write_text(json.dumps(test_data))
        
        # Test loading
        with patch('src.ingest.batch_loader.ProductVectorizer'):
            loader = BatchProductLoader(connection_string="test")
            products = loader.load_from_json(json_file)
            
            assert len(products) == 1
            assert products[0]["sku"] == "TEST001"


@pytest.fixture
def sample_products():
    """Fixture providing sample product data."""
    return [
        {
            "sku": "ELEC001",
            "name": "Wireless Mouse",
            "description": "Ergonomic wireless mouse with USB receiver",
            "category": "Electronics",
            "brand": "TechBrand",
            "price": 29.99,
            "stock_quantity": 150
        },
        {
            "sku": "ELEC002",
            "name": "Mechanical Keyboard",
            "description": "RGB mechanical gaming keyboard",
            "category": "Electronics",
            "brand": "GamerPro",
            "price": 89.99,
            "stock_quantity": 75
        }
    ]


def test_sample_products_fixture(sample_products):
    """Test the sample products fixture."""
    assert len(sample_products) == 2
    assert all("sku" in p for p in sample_products)
'''
    
    file_path = base_path / "tests" / "test_ingest.py"
    file_path.write_text(content, encoding='utf-8')
    print(f"✓ Created: tests/test_ingest.py")


def create_gitkeep(base_path: Path):
    """Create .gitkeep file in data directory."""
    file_path = base_path / "data" / ".gitkeep"
    file_path.write_text("# This file keeps the data directory in git\n", encoding='utf-8')
    print(f"✓ Created: data/.gitkeep")


def create_init_files(base_path: Path):
    """Create __init__.py files for Python packages."""
    init_content = '"""Package initialization."""\n'
    
    init_paths = [
        base_path / "src" / "__init__.py",
        base_path / "src" / "database" / "__init__.py",
        base_path / "src" / "ingest" / "__init__.py",
        base_path / "src" / "api" / "__init__.py",
        base_path / "tests" / "__init__.py"
    ]
    
    for path in init_paths:
        path.write_text(init_content, encoding='utf-8')
        print(f"✓ Created: {path.relative_to(base_path)}")


def main():
    """Main execution function."""
    print("\n" + "=" * 60)
    print("  Aether-Store-SQL Project Structure Generator")
    print("  SQL Server 2025 Vector Search RAG System")
    print("=" * 60 + "\n")
    
    # Get base path (current directory)
    base_path = Path.cwd()
    
    print(f"Creating project structure in: {base_path}\n")
    
    # Create directory structure
    create_directory_structure(base_path)
    
    # Create root files
    create_gitignore(base_path)
    create_env_example(base_path)
    create_docker_compose(base_path)
    create_readme(base_path)
    create_requirements_txt(base_path)
    
    # Create database files
    create_schema_sql(base_path)
    create_migrations_sql(base_path)
    
    # Create ingestion files
    create_vectorizer_py(base_path)
    create_batch_loader_py(base_path)
    
    # Create API files
    create_search_engine_py(base_path)
    create_app_py(base_path)
    
    # Create test files
    create_test_ingest_py(base_path)
    
    # Create additional files
    create_gitkeep(base_path)
    create_init_files(base_path)
    
    print("\n" + "=" * 60)
    print("  ✓ Project structure created successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Review and customize .env.example, then copy to .env")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Start SQL Server: docker-compose up -d sqlserver")
    print("4. Run database schema: Execute schema.sql")
    print("5. Start the API: python -m src.api.app")
    print("\nHappy coding! 🚀\n")


if __name__ == "__main__":
    main()
