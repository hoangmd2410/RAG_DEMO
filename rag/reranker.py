import torch
from typing import List, Dict, Any, Tuple, Optional
import logging
from config import Config
from transformers import AutoModelForSequenceClassification, AutoTokenizer
logger = logging.getLogger(__name__)


class Reranker:
    """Manages document reranking using Vietnamese reranker model."""
    
    def __init__(self):
        self.model_name = Config.RERANKER_MODEL_NAME
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the reranker model."""
        try:
            logger.info(f"Loading reranker model: {self.model_name}")
            
            # Use transformers directly to avoid protobuf issues
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"✅ Reranker model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"❌ Error loading reranker model: {e}")
            self.model = None
            self.tokenizer = None
    
    def rerank_documents(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank documents based on their relevance to the query.
        
        Args:
            query: The search query string
            documents: List of document dictionaries containing 'content' field
            top_k: Number of top documents to return (if None, returns all)
            
        Returns:
            List of reranked documents with updated scores
        """
        assert self.model is not None, "Reranker model not loaded"
        assert self.tokenizer is not None, "Tokenizer not loaded"
        try:                        # Extract document texts
            pairs = []
            for doc in documents:
                content = doc.get('text', '')
                pairs.append([query, content])
            
            with torch.no_grad():
                inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=8192).to(self.device)
                scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float()
                print(scores)
            
            # Combine documents with their new scores
            for i, doc in enumerate(documents):
                doc['rerank_score'] = float(scores[i])  

            #sort by rerank_score          
            documents.sort(key=lambda x: x['rerank_score'], reverse=True)
            # remove documents with rerank_score < -5
            documents = [doc for doc in documents if doc['rerank_score'] > -5]
            
            logger.info(f"✅ Reranked documents successfully")
            return documents
            
        except Exception as e:
            logger.error(f"❌ Error during reranking: {e}")
            # Return original documents on error
            return documents
    
    
    def is_available(self) -> bool:
        """Check if the reranker model is available."""
        return self.model is not None
