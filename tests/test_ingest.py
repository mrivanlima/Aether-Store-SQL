"""
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
