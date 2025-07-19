import os
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import uuid
from document_processor import DocumentProcessor
from embedding_manager import EmbeddingManager
from qdrant_setup import QdrantManager
from config import Config
from langchain_text_splitters.markdown import MarkdownTextSplitter
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentIndexer:
    """Handles the complete indexing pipeline for documents."""
    
    def __init__(self):
        self.embedding_manager = EmbeddingManager()
        self.qdrant_manager = QdrantManager()
        self.document_processor = DocumentProcessor()
        self.text_splitter = MarkdownTextSplitter(
            chunk_size=3000,
            chunk_overlap=200,
        )
    
    def index_document(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Complete indexing pipeline for a single document.
        
        Args:
            file_path: Path to the document file
            metadata: Optional metadata about the document
            
        Returns:
            Dictionary with indexing results and statistics
        """
        result = {
            'success': False,
            'document_id': None,
            'filename': None,
            'chunks_count': 0,
            'total_characters': 0,
            'processing_time': 0,
            'error': None,
            'warnings': []
        }
        
        start_time = datetime.now()
        
        try:
            # Step 1: File validation
            logger.info(f"🔍 Starting indexing pipeline for: {file_path}")
            
            validation_result = validate_file(file_path)
            if not validation_result['valid']:
                result['error'] = validation_result['error']
                return result
            
            # Step 2: Extract filename and prepare metadata
            filename = os.path.basename(file_path)
            result['filename'] = filename
            
            document_metadata = {
                'document_id': str(uuid.uuid4()),
                'filename': filename,
                'file_type': validation_result['extension'],
                'file_size_mb': validation_result['size_mb'],
                'timestamp': datetime.now().isoformat(),
                'indexed_by': 'semantic_search_pipeline'
            }
            
            # Add custom metadata if provided
            if metadata:
                document_metadata.update(metadata)
            
            result['document_id'] = document_metadata['document_id']
            
            # Step 3: Text extraction
            logger.info(f"📄 Extracting text from {filename}")
            text = self.document_processor.extract_text(file_path)
            
            if not text:
                result['error'] = "Failed to extract text from document"
                return result
            
            # Clean the extracted text
            text = clean_text(text)
            result['total_characters'] = len(text)
            
            if len(text) < 10:  # Minimum text length check
                result['error'] = "Document contains insufficient text content"
                return result
            
            # Step 4: Text chunking
            logger.info(f"✂️ Chunking text into smaller segments")
            chunk_texts = self.text_splitter.split_text(text)
            
            if not chunk_texts:
                result['error'] = "Failed to create text chunks"
                return result
            
            # Convert to expected chunk format
            chunks = [{'text': chunk_text} for chunk_text in chunk_texts]
            result['chunks_count'] = len(chunks)
            
            # Step 5: Generate embeddings
            logger.info(f"🧠 Generating embeddings for {len(chunks)} chunks")
            embeddings = self.embedding_manager.get_embeddings(chunk_texts)
            
            if not embeddings or len(embeddings) != len(chunks):
                result['error'] = "Failed to generate embeddings for all chunks"
                return result
            
            # Step 6: Store in Qdrant
            logger.info(f"💾 Storing vectors in Qdrant database")
            storage_success = self.qdrant_manager.add_documents(
                chunks=chunks,
                embeddings=embeddings,
                document_metadata=document_metadata
            )
            
            if not storage_success:
                result['error'] = "Failed to store document vectors in database"
                return result
            
            # Calculate processing time
            end_time = datetime.now()
            result['processing_time'] = (end_time - start_time).total_seconds()
            
            result['success'] = True
            logger.info(f"✅ Successfully indexed document: {filename}")
            logger.info(f"   📊 Stats: {len(chunks)} chunks, {len(text)} chars, {result['processing_time']:.2f}s")
            
        except Exception as e:
            result['error'] = f"Indexing failed: {str(e)}"
            logger.error(f"❌ Error indexing document: {e}")
        
        return result
    
    def index_multiple_documents(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Index multiple documents in batch.
        
        Args:
            file_paths: List of file paths to index
            
        Returns:
            List of indexing results for each document
        """
        results = []
        successful_count = 0
        
        logger.info(f"📚 Starting batch indexing for {len(file_paths)} documents")
        
        for i, file_path in enumerate(file_paths, 1):
            logger.info(f"📄 Processing document {i}/{len(file_paths)}")
            
            result = self.index_document(file_path)
            results.append(result)
            
            if result['success']:
                successful_count += 1
            else:
                logger.warning(f"⚠️ Failed to index {result['filename']}: {result['error']}")
        
        logger.info(f"✅ Batch indexing completed: {successful_count}/{len(file_paths)} successful")
        return results
    
    def reindex_document(self, file_path: str, document_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Reindex an existing document (removes old version first).
        
        Args:
            file_path: Path to the document file
            document_id: ID of existing document to replace
            
        Returns:
            Indexing result
        """
        # If document_id provided, delete existing document first
        if document_id and self.qdrant_manager.is_connected():
            logger.info(f"🗑️ Removing existing document: {document_id}")
            self.qdrant_manager.delete_document(document_id)
        
        # Index the new version
        return self.index_document(file_path)
    
    def get_indexing_status(self) -> Dict[str, Any]:
        """Get current status of the indexing system."""
        status = {
            'embedding_model': Config.EMBEDDING_MODEL_NAME,
            'qdrant_connected': self.qdrant_manager.is_connected(),
            'collection_info': {},
            'system_ready': False
        }
        
        if self.qdrant_manager.is_connected():
            status['collection_info'] = self.qdrant_manager.get_collection_info()
            status['system_ready'] = True
        
        return status

def process_document_upload(file_path: str, custom_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convenience function to process a single uploaded document.
    
    Args:
        file_path: Path to the uploaded file
        custom_metadata: Optional custom metadata
        
    Returns:
        Processing result with success status and details
    """
    indexer = DocumentIndexer()
    return indexer.index_document(file_path, custom_metadata)

def batch_index_directory(directory_path: str, file_extensions: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Index all supported documents in a directory.
    
    Args:
        directory_path: Path to directory containing documents
        file_extensions: Optional list of file extensions to process
        
    Returns:
        List of indexing results
    """
    if not os.path.exists(directory_path):
        logger.error(f"❌ Directory not found: {directory_path}")
        return []
    
    # Get supported file extensions
    extensions = file_extensions or Config.SUPPORTED_FORMATS
    
    # Find all files with supported extensions
    file_paths = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext in extensions:
                file_paths.append(os.path.join(root, file))
    
    if not file_paths:
        logger.warning(f"⚠️ No supported documents found in: {directory_path}")
        return []
    
    logger.info(f"📁 Found {len(file_paths)} documents in directory")
    
    # Index all found documents
    indexer = DocumentIndexer()
    return indexer.index_multiple_documents(file_paths)

def verify_indexing_setup() -> Dict[str, Any]:
    """
    Verify that all components for indexing are properly set up.
    
    Returns:
        Dictionary with setup verification results
    """
    verification = {
        'embedding_model_loaded': False,
        'qdrant_connected': False,
        'collection_initialized': False,
        'overall_status': 'not_ready',
        'issues': []
    }
    
    try:
        # Test embedding model
        embedding_manager = EmbeddingManager()
        test_embedding = embedding_manager.get_embeddings(["test text"])
        verification['embedding_model_loaded'] = len(test_embedding) > 0
        
        # Test Qdrant connection
        qdrant_manager = QdrantManager()
        verification['qdrant_connected'] = qdrant_manager.is_connected()
        
        if verification['qdrant_connected']:
            collection_info = qdrant_manager.get_collection_info()
            verification['collection_initialized'] = 'error' not in collection_info
        
        # Determine overall status
        if all([
            verification['embedding_model_loaded'],
            verification['qdrant_connected'],
            verification['collection_initialized']
        ]):
            verification['overall_status'] = 'ready'
        else:
            verification['overall_status'] = 'partial'
            
            if not verification['embedding_model_loaded']:
                verification['issues'].append("Embedding model not loaded properly")
            if not verification['qdrant_connected']:
                verification['issues'].append("Qdrant database not connected")
            if not verification['collection_initialized']:
                verification['issues'].append("Qdrant collection not initialized")
    
    except Exception as e:
        verification['overall_status'] = 'error'
        verification['issues'].append(f"Setup verification failed: {str(e)}")
        logger.error(f"❌ Setup verification error: {e}")
    
    return verification 


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\"\']+', '', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

def validate_file(file_path: str) -> Dict[str, Any]:
    """Validate uploaded file."""
    result = {
        'valid': False,
        'error': None,
        'size_mb': 0,
        'extension': None
    }
    
    try:
        if not os.path.exists(file_path):
            result['error'] = "File does not exist"
            return result
        
        # Check file size
        size_bytes = os.path.getsize(file_path)
        size_mb = size_bytes / (1024 * 1024)
        result['size_mb'] = size_mb
        
        if size_mb > Config.MAX_FILE_SIZE_MB:
            result['error'] = f"File too large: {size_mb:.1f}MB (max: {Config.MAX_FILE_SIZE_MB}MB)"
            return result
        
        # Check file extension
        file_ext = os.path.splitext(file_path)[1].lower()
        result['extension'] = file_ext
        
        if file_ext not in Config.SUPPORTED_FORMATS:
            result['error'] = f"Unsupported format: {file_ext} (supported: {Config.SUPPORTED_FORMATS})"
            return result
        
        result['valid'] = True
        logger.info(f"✅ File validation passed: {file_path}")
        
    except Exception as e:
        result['error'] = f"Validation error: {str(e)}"
        logger.error(f"❌ File validation failed: {e}")
    
    return result 


