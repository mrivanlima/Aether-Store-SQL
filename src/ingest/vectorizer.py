"""
Vector Embedding Strategy Module
Supports: OpenAI (Cloud) and SentenceTransformers (Local)
"""
import os
from typing import List, Optional
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

# --- INTERFACE ---
class BaseVectorizer:
    """Interface to ensure all strategies behave the same way."""
    def generate_embedding(self, text: str) -> List[float]:
        raise NotImplementedError

# --- STRATEGY A: OPENAI (What you already have) ---
class OpenAIVectorizer(BaseVectorizer):
    """Paid, Cloud-based, High Accuracy (1536 dim)"""
    def __init__(self):
        from openai import OpenAI
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = "text-embedding-3-small"
        self.dimension = 1536
        
        if not self.api_key:
            logger.warning("No OpenAI API Key found. Using dummy mode.")
            self.client = None
        else:
            self.client = OpenAI(api_key=self.api_key)
            logger.info(f"Initialized OpenAIVectorizer (Model: {self.model})")
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate_embedding(self, text: str) -> List[float]:
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
            logger.error(f"OpenAI Error: {e}")
            raise

# --- STRATEGY B: LOCAL (New Addition) ---
class LocalVectorizer(BaseVectorizer):
    """Free, Local CPU, Private (384 dim)"""
    def __init__(self):
        # Lazy import: Only loads heavy AI libraries if we actually use this mode
        from sentence_transformers import SentenceTransformer
        logger.info("Loading local model 'all-MiniLM-L6-v2' (This runs on CPU)...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = 384 

    def generate_embedding(self, text: str) -> List[float]:
        # Local inference is fast and reliable; no retry logic needed
        embedding = self.model.encode(text)
        return embedding.tolist()

# --- THE FACTORY (The Switch) ---
def VectorizerFactory():
    """Decides which brain to load based on environment variables."""
    # Defaults to 'openai' if not set
    mode = os.getenv("VECTOR_PROVIDER", "openai").lower()
    
    if mode == "local":
        logger.info("Strategy Selected: LOCAL (SentenceTransformers)")
        return LocalVectorizer()
    else:
        logger.info("Strategy Selected: CLOUD (OpenAI)")
        return OpenAIVectorizer()