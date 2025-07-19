#!/usr/bin/env python3
"""
Database Checker for RAG Semantic Search Pipeline

This script provides utilities to inspect and retrieve chunks stored in Qdrant database.
"""

import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, ScrollRequest
import json
from datetime import datetime
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseChecker:
    """Utility class to inspect Qdrant database contents."""
    
    def __init__(self):
        self.client = None
        self.collection_name = Config.QDRANT_COLLECTION_NAME
        self._connect()
    
    def _connect(self):
        """Connect to Qdrant database."""
        try:
            self.client = QdrantClient(
                host=Config.QDRANT_HOST,
                port=Config.QDRANT_PORT
            )
            
            # Test connection
            self.client.get_collections()
            logger.info(f"✅ Connected to Qdrant at {Config.QDRANT_HOST}:{Config.QDRANT_PORT}")
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Qdrant: {e}")
            logger.info("💡 Please make sure Qdrant is running: docker run -p 6333:6333 qdrant/qdrant")
            self.client = None
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get overall database statistics."""
        if not self.client:
            return {"error": "Not connected to database"}
        
        try:
            # Get collection info
            collection_info = self.client.get_collection(self.collection_name)
            
            # Get all points to analyze
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,  # Adjust if you have more documents
                with_payload=True
            )
            
            points = scroll_result[0]
            
            # Analyze the data
            total_chunks = len(points)
            documents = {}
            total_characters = 0
            
            for point in points:
                payload = point.payload
                doc_id = payload.get('document_id', 'unknown')
                doc_name = payload.get('document_name', 'unknown')
                chunk_length = payload.get('chunk_length', 0)
                total_characters += chunk_length
                
                if doc_id not in documents:
                    documents[doc_id] = {
                        'name': doc_name,
                        'chunks': 0,
                        'characters': 0,
                        'type': payload.get('document_type', 'unknown'),
                        'timestamp': payload.get('upload_timestamp', 'unknown')
                    }
                
                documents[doc_id]['chunks'] += 1
                documents[doc_id]['characters'] += chunk_length
            
            stats = {
                'total_documents': len(documents),
                'total_chunks': total_chunks,
                'total_characters': total_characters,
                'average_chunk_size': total_characters / total_chunks if total_chunks > 0 else 0,
                'collection_info': {
                    'vectors_count': collection_info.vectors_count,
                    'indexed_vectors_count': collection_info.indexed_vectors_count,
                    'points_count': collection_info.points_count,
                },
                'documents': documents
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error getting database stats: {e}")
            return {"error": str(e)}
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents in the database."""
        if not self.client:
            return []
        
        try:
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,
                with_payload=True
            )
            
            points = scroll_result[0]
            
            # Group by document
            documents = {}
            for point in points:
                payload = point.payload
                doc_id = payload.get('document_id', 'unknown')
                
                if doc_id not in documents:
                    documents[doc_id] = {
                        'document_id': doc_id,
                        'name': payload.get('document_name', 'unknown'),
                        'type': payload.get('document_type', 'unknown'),
                        'timestamp': payload.get('upload_timestamp', 'unknown'),
                        'total_chunks': payload.get('total_chunks', 0),
                        'chunks_found': 0,
                        'total_characters': 0
                    }
                
                documents[doc_id]['chunks_found'] += 1
                documents[doc_id]['total_characters'] += payload.get('chunk_length', 0)
            
            return list(documents.values())
            
        except Exception as e:
            logger.error(f"❌ Error listing documents: {e}")
            return []
    
    def get_document_chunks(self, document_id: str = None, document_name: str = None) -> List[Dict[str, Any]]:
        """Get all chunks for a specific document."""
        if not self.client:
            return []
        
        try:
            # Build filter
            filter_conditions = []
            
            if document_id:
                filter_conditions.append(
                    FieldCondition(key="document_id", match=MatchValue(value=document_id))
                )
            
            if document_name:
                filter_conditions.append(
                    FieldCondition(key="document_name", match=MatchValue(value=document_name))
                )
            
            if not filter_conditions:
                logger.warning("⚠️ No document filter specified, getting all chunks")
                scroll_result = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=10000,
                    with_payload=True
                )
            else:
                scroll_result = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=Filter(must=filter_conditions),
                    limit=10000,
                    with_payload=True
                )
            
            points = scroll_result[0]
            
            # Format chunks
            chunks = []
            for point in points:
                chunk_data = {
                    'point_id': str(point.id),
                    'document_id': point.payload.get('document_id'),
                    'document_name': point.payload.get('document_name'),
                    'chunk_id': point.payload.get('chunk_id'),
                    'chunk_index': point.payload.get('chunk_index'),
                    'text': point.payload.get('text'),
                    'chunk_length': point.payload.get('chunk_length'),
                    'chunk_start_pos': point.payload.get('chunk_start_pos'),
                    'chunk_end_pos': point.payload.get('chunk_end_pos'),
                    'document_type': point.payload.get('document_type'),
                    'upload_timestamp': point.payload.get('upload_timestamp'),
                    'total_chunks': point.payload.get('total_chunks')
                }
                chunks.append(chunk_data)
            
            # Sort by chunk index
            chunks.sort(key=lambda x: x.get('chunk_index', 0))
            
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Error getting document chunks: {e}")
            return []
    
    def search_chunks(self, query_text: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for chunks containing specific text."""
        if not self.client:
            return []
        
        try:
            # Simple text search in payloads
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,
                with_payload=True
            )
            
            points = scroll_result[0]
            
            # Filter chunks containing the query text
            matching_chunks = []
            for point in points:
                text = point.payload.get('text', '').lower()
                if query_text.lower() in text:
                    chunk_data = {
                        'point_id': str(point.id),
                        'document_id': point.payload.get('document_id'),
                        'document_name': point.payload.get('document_name'),
                        'chunk_id': point.payload.get('chunk_id'),
                        'text': point.payload.get('text'),
                        'chunk_length': point.payload.get('chunk_length'),
                        'match_score': text.count(query_text.lower())  # Simple relevance score
                    }
                    matching_chunks.append(chunk_data)
            
            # Sort by relevance (number of matches)
            matching_chunks.sort(key=lambda x: x['match_score'], reverse=True)
            
            return matching_chunks[:limit]
            
        except Exception as e:
            logger.error(f"❌ Error searching chunks: {e}")
            return []
    
    def delete_document(self, document_id: str) -> bool:
        """Delete all chunks for a specific document."""
        if not self.client:
            return False
        
        try:
            # First, find all points for this document
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
                ),
                limit=10000,
                with_payload=False  # We only need IDs
            )
            
            points = scroll_result[0]
            point_ids = [point.id for point in points]
            
            if not point_ids:
                logger.warning(f"⚠️ No chunks found for document: {document_id}")
                return True
            
            # Delete all points
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=point_ids
            )
            
            logger.info(f"✅ Deleted {len(point_ids)} chunks for document: {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error deleting document: {e}")
            return False
    
    def export_chunks_to_json(self, output_file: str, document_id: str = None) -> bool:
        """Export chunks to JSON file."""
        try:
            if document_id:
                chunks = self.get_document_chunks(document_id=document_id)
            else:
                chunks = self.get_document_chunks()
            
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'total_chunks': len(chunks),
                'chunks': chunks
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Exported {len(chunks)} chunks to: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error exporting chunks: {e}")
            return False

def main():
    """Main function to demonstrate database checking capabilities."""
    checker = DatabaseChecker()
    
    if not checker.client:
        print("❌ Cannot connect to database. Please ensure Qdrant is running.")
        return
    
    print("🔍 RAG Database Checker")
    print("=" * 50)
    
    # Get database statistics
    print("\n📊 Database Statistics:")
    stats = checker.get_database_stats()
    
    if "error" in stats:
        print(f"❌ Error: {stats['error']}")
        return
    
    print(f"  📄 Total Documents: {stats['total_documents']}")
    print(f"  📝 Total Chunks: {stats['total_chunks']}")
    print(f"  📊 Total Characters: {stats['total_characters']:,}")
    print(f"  📏 Average Chunk Size: {stats['average_chunk_size']:.1f} characters")
    print(f"  🗄️ Vector Count: {stats['collection_info']['vectors_count']}")
    
    # List documents
    print("\n📚 Documents in Database:")
    documents = checker.list_documents()
    
    for i, doc in enumerate(documents, 1):
        print(f"  {i}. {doc['name']} ({doc['type']})")
        print(f"     📄 ID: {doc['document_id']}")
        print(f"     📝 Chunks: {doc['chunks_found']}/{doc['total_chunks']}")
        print(f"     📊 Characters: {doc['total_characters']:,}")
        print(f"     🕒 Uploaded: {doc['timestamp']}")
        print()
    
    # Show sample chunks from first document
    if documents:
        for doc in documents:
            print(f"\n📝 Sample Chunks from '{doc['name']}':")
            chunks = checker.get_document_chunks(document_id=doc['document_id'])
        
            for i, chunk in enumerate(chunks):
                print(f"  Chunk {chunk['chunk_index'] + 1}:")
                print(f"    📏 Length: {chunk['chunk_length']} characters")
                print(f"    📄 Text Preview: {chunk['text']}...")
                print()

if __name__ == "__main__":
    main() 