"""
Vector Embedding Generation Module
"""
import os
from typing import List, Optional
from openai import OpenAI
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

# --- NO IMPORTS FROM src.ingest HERE! ---

class ProductVectorizer:
    """
    Generates vector embeddings for product data using OpenAI API.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        dimension: int = 1536
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("No OpenAI API Key found. Using dummy mode for testing.")
            self.client = None
        else:
            self.client = OpenAI(api_key=self.api_key)
            
        self.model = model
        self.dimension = dimension
        logger.info(f"Initialized ProductVectorizer with model: {self.model}")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate_embedding(self, text: str) -> List[float]:
        # Dummy mode for testing without paying OpenAI
        if not self.client:
            return [0.0] * self.dimension

        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
                dimensions=self.dimension
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise