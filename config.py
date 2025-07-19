import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Embedding Model Configuration
    EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"  # Using Qwen3-Embedding-0.6B as requested
    
    # Nanonets OCR Configuration
    NANONETS_OCR_MODEL = "nanonets/Nanonets-OCR-s"
    NANONETS_MAX_NEW_TOKENS = 4096
    
    # Qdrant Configuration
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_COLLECTION_NAME = "legal_documents"
    VECTOR_SIZE = 1024  # Dimension for Qwen3-Embedding-0.6B (up to 1024)
    
    # Document Processing Configuration
    CHUNK_SIZE = 500  # Characters per chunk
    CHUNK_OVERLAP = 50  # Overlap between chunks
    MAX_CHUNKS_PER_DOCUMENT = 100
    
    # Retrieval Configuration
    TOP_K_RESULTS = 20  # Number of similar chunks to retrieve
    SIMILARITY_THRESHOLD = 0.2  # Minimum similarity score
    
    # File Processing Configuration
    SUPPORTED_FORMATS = [".pdf", ".docx", ".txt", ".png", ".jpg", ".jpeg", ".html", ".htm", ".tiff", ".tif", ".bmp", ".webp"]
    MAX_FILE_SIZE_MB = 50
    
    # Gradio Configuration
    GRADIO_SHARE = False
    GRADIO_PORT = 7860

# Validate required configurations
def validate_config():
    """Validate that required configuration values are set."""
    if not Config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    print("✅ Configuration validation passed")
    return True 