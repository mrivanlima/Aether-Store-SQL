"""
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
