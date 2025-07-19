import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel, AutoProcessor, AutoModelForImageTextToText
import pypdf
from docx import Document
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from PIL import Image
import requests
from io import BytesIO
import tempfile
import fitz
import io
from config import Config
import torch
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages text embeddings using Qwen3-Embedding-0.6B model."""
    
    def __init__(self):
        self.model_name = Config.EMBEDDING_MODEL_NAME
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model and tokenizer."""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            
            # Use sentence-transformers for easier handling
            self.model = SentenceTransformer(self.model_name)
            self.model.to(self.device)
            
            logger.info(f"✅ Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            # Fallback to a more compatible model
            logger.info("Falling back to sentence-transformers/all-MiniLM-L6-v2")
            self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            self.model.to(self.device)
    
    def get_embeddings(self, texts: List[str], instruction: Optional[str] = None) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            instruction: Optional instruction to improve embedding quality
            
        Returns:
            List of embedding vectors
        """
        try:
            if not texts:
                return []
            
            # For queries, we can use instructions to improve performance
            if instruction:
                # Format texts with instruction for better retrieval
                formatted_texts = [f"Instruct: {instruction}\nQuery: {text}" for text in texts]
            else:
                formatted_texts = texts
            
            # Check if model is available
            if not self.model:
                logger.error("❌ Embedding model not available")
                return []
            
            # Generate embeddings
            embeddings = self.model.encode(
                formatted_texts,
                convert_to_tensor=False,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > 10
            )
            
            # Convert to list format
            import numpy as np
            if isinstance(embeddings, np.ndarray):
                if len(embeddings.shape) == 1:
                    embeddings = [embeddings.tolist()]
                else:
                    embeddings = embeddings.tolist()
            elif hasattr(embeddings, 'tolist'):
                embeddings = embeddings.tolist()
            
            # Ensure we have a list of lists
            if embeddings and not isinstance(embeddings[0], list):
                embeddings = [embeddings]
            
            logger.info(f"✅ Generated embeddings for {len(texts)} texts")
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ Error generating embeddings: {e}")
            return []
    
    def get_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for a search query with instruction."""
        instruction = "Given a web search query, retrieve relevant passages that answer the query"
        embeddings = self.get_embeddings([query], instruction=instruction)
        return embeddings[0] if embeddings else []

