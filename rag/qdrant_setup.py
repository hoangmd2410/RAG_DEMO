import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from qdrant_client.http import models
import uuid
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QdrantManager:
    """Manages Qdrant vector database operations."""
    
    def __init__(self):
        self.client = None
        self.collection_name = Config.QDRANT_COLLECTION_NAME
        self.vector_size = Config.VECTOR_SIZE
        self._connect()
    
    def _connect(self):
        """Connect to Qdrant server."""
        try:
            # Try to connect to Qdrant server
            self.client = QdrantClient(
                host=Config.QDRANT_HOST,
                port=Config.QDRANT_PORT
            )
            
            # Test connection
            self.client.get_collections()
            logger.info(f"✅ Connected to Qdrant at {Config.QDRANT_HOST}:{Config.QDRANT_PORT}")
            
            # Initialize collection if it doesn't exist
            self._initialize_collection()
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Qdrant: {e}")
            logger.info("💡 Please make sure Qdrant is running. You can start it with:")
            logger.info("docker run -p 6333:6333 qdrant/qdrant")
            self.client = None
    
    def _initialize_collection(self):
        """Initialize the collection for storing document vectors."""
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if self.collection_name not in collection_names:
                # Create new collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"✅ Created collection: {self.collection_name}")
            else:
                logger.info(f"✅ Collection already exists: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"❌ Error initializing collection: {e}")
            raise
    
    def add_documents(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]], 
                     document_metadata: Dict[str, Any]) -> bool:
        """
        Add document chunks and their embeddings to Qdrant.
        
        Args:
            chunks: List of text chunks with metadata
            embeddings: List of embedding vectors for each chunk
            document_metadata: Metadata about the source document
            
        Returns:
            Success status
        """
        if not self.client:
            logger.error("❌ Qdrant client not connected")
            return False
        
        try:
            points = []
            document_id = document_metadata.get('document_id', str(uuid.uuid4()))
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # Get chunk text (handle both old and new formats)
                chunk_text = chunk.get('text', chunk) if isinstance(chunk, dict) else str(chunk)
                chunk_length = len(chunk_text)
                
                # Create point with metadata
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        'document_id': document_id,
                        'chunk_id': chunk.get('id', i) if isinstance(chunk, dict) else i,
                        'text': chunk_text,
                        'chunk_length': chunk_length,
                        'chunk_start_pos': chunk.get('start_pos', 0) if isinstance(chunk, dict) else 0,
                        'chunk_end_pos': chunk.get('end_pos', chunk_length) if isinstance(chunk, dict) else chunk_length,
                        'document_name': document_metadata.get('filename', 'unknown'),
                        'document_type': document_metadata.get('file_type', 'unknown'),
                        'upload_timestamp': document_metadata.get('timestamp'),
                        'total_chunks': len(chunks),
                        'chunk_index': i
                    }
                )
                points.append(point)
            
            # Upload points to Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            logger.info(f"✅ Added {len(points)} chunks to Qdrant for document: {document_metadata.get('filename', 'unknown')}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error adding documents to Qdrant: {e}")
            return False
    
    def search_similar(self, query_embedding: List[float], top_k: int = None, 
                      score_threshold: float = None, document_filter: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Search for similar chunks in Qdrant.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            score_threshold: Minimum similarity score
            document_filter: Optional filter for specific documents
            
        Returns:
            List of search results with scores and metadata
        """
        if not self.client:
            logger.error("❌ Qdrant client not connected")
            return []
        
        try:
            top_k = top_k or Config.TOP_K_RESULTS
            score_threshold = score_threshold or Config.SIMILARITY_THRESHOLD
            
            # Prepare search filters
            search_filter = None
            if document_filter:
                conditions = []
                for key, value in document_filter.items():
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value)
                        )
                    )
                if conditions:
                    search_filter = Filter(must=conditions)
            
            # Perform search
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                score_threshold=score_threshold,
                query_filter=search_filter
            )
            
            # Format results
            results = []
            for result in search_results:
                results.append({
                    'id': result.id,
                    'score': result.score,
                    'text': result.payload['text'],
                    'document_id': result.payload['document_id'],
                    'document_name': result.payload['document_name'],
                    'document_type': result.payload['document_type'],
                    'chunk_id': result.payload['chunk_id'],
                    'chunk_index': result.payload['chunk_index'],
                    'total_chunks': result.payload['total_chunks'],
                    'chunk_length': result.payload['chunk_length'],
                    'upload_timestamp': result.payload['upload_timestamp']
                })
            
            logger.info(f"✅ Found {len(results)} similar chunks (score >= {score_threshold})")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error searching in Qdrant: {e}")
            return []
    
    def delete_document(self, document_id: str) -> bool:
        """Delete all chunks of a specific document."""
        if not self.client:
            logger.error("❌ Qdrant client not connected")
            return False
        
        try:
            # Delete points with matching document_id
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=Filter(
                        must=[
                            FieldCondition(
                                key="document_id",
                                match=MatchValue(value=document_id)
                            )
                        ]
                    )
                )
            )
            
            logger.info(f"✅ Deleted document: {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error deleting document: {e}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        if not self.client:
            return {'error': 'Client not connected'}
        
        try:
            collection_info = self.client.get_collection(self.collection_name)
            
            return {
                'name': collection_info.config.params.vectors.size,
                'vector_size': collection_info.config.params.vectors.size,
                'distance': collection_info.config.params.vectors.distance,
                'points_count': collection_info.points_count,
                'segments_count': collection_info.segments_count,
                'status': collection_info.status
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting collection info: {e}")
            return {'error': str(e)}
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents in the collection."""
        if not self.client:
            return []
        
        try:
            # Get a sample of points to extract document information
            search_results = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000,  # Adjust based on your needs
                with_payload=True
            )
            
            # Extract unique documents
            documents = {}
            for point in search_results[0]:
                doc_id = point.payload['document_id']
                if doc_id not in documents:
                    documents[doc_id] = {
                        'document_id': doc_id,
                        'document_name': point.payload['document_name'],
                        'document_type': point.payload['document_type'],
                        'upload_timestamp': point.payload['upload_timestamp'],
                        'total_chunks': point.payload['total_chunks'],
                        'chunk_count': 0
                    }
                documents[doc_id]['chunk_count'] += 1
            
            return list(documents.values())
            
        except Exception as e:
            logger.error(f"❌ Error listing documents: {e}")
            return []
    
    def clear_collection(self) -> bool:
        """Clear all data from the collection."""
        if not self.client:
            logger.error("❌ Qdrant client not connected")
            return False
        
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(self.collection_name)
            self._initialize_collection()
            
            logger.info(f"✅ Cleared collection: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error clearing collection: {e}")
            return False
    
    def is_connected(self) -> bool:
        """Check if connected to Qdrant."""
        return self.client is not None

def setup_qdrant() -> QdrantManager:
    """Initialize and return Qdrant manager."""
    return QdrantManager()

def check_qdrant_connection() -> bool:
    """Check if Qdrant is accessible."""
    try:
        client = QdrantClient(host=Config.QDRANT_HOST, port=Config.QDRANT_PORT)
        client.get_collections()
        return True
    except Exception:
        return False 