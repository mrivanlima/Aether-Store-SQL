# Aether-Store-SQL

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
git clone https://github.com/mrivanlima/Aether-Store-SQL.git
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
source venv/bin/activate  # On Windows: venv\Scripts\activate
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
