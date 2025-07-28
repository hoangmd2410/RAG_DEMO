import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from openai import AsyncOpenAI
import openai
from embedding_manager import EmbeddingManager
from qdrant_setup import QdrantManager
from config import Config
import asyncio
import json
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from indexing import clean_text


class QueryProcessor:
    """Handles query processing and retrieval from the vector database."""
    
    def __init__(self):
        self.embedding_manager = EmbeddingManager()
        self.qdrant_manager = QdrantManager()
        
        # Initialize OpenAI client
        if Config.OPENAI_API_KEY:
            openai.api_key = Config.OPENAI_API_KEY
        else:
            logger.warning("⚠️ OpenAI API key not found. Some features may not work.")
    
    def search(self, query: str, top_k: Optional[int] = None, score_threshold: Optional[float] = None, 
               document_filter: Optional[Dict[str, Any]] = None, get_entire_text: bool = False) -> Dict[str, Any]:
        """
        Perform semantic search for the given query.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            score_threshold: Minimum similarity score
            document_filter: Optional filter for specific documents
            
        Returns:
            Search results with metadata and statistics
        """
        search_result = {
            'query': query,
            'results': [],
            'total_results': 0,
            'search_time': 0,
            'success': False,
            'error': None
        }
        
        start_time = datetime.now()
        
        try:
            if not query.strip():
                search_result['error'] = "Query cannot be empty"
                return search_result
            
            # Clean and prepare query
            cleaned_query = clean_text(query)
            logger.info(f"🔍 Processing search query: {cleaned_query}")
            
            # Step 1: Generate query embedding
            query_embedding = self.embedding_manager.get_query_embedding(cleaned_query)
            
            if not query_embedding:
                search_result['error'] = "Failed to generate query embedding"
                return search_result
            
            # Step 2: Search in Qdrant
            search_results = self.qdrant_manager.search_similar(
                query_embedding=query_embedding,
                top_k=top_k or Config.TOP_K_RESULTS,
                score_threshold=score_threshold or Config.SIMILARITY_THRESHOLD,
                document_filter=document_filter or {},
                get_entire_text=get_entire_text
            )
            
            # Step 3: Format and enhance results
            formatted_results = []
            for result in search_results:
                formatted_result = {
                    'id': result['id'],
                    'score': round(result['score'], 4),
                    'text': result['text'],
                    'entire_text': result['entire_text'] if get_entire_text else None,
                    'document_name': result['document_name'],
                    'document_type': result['document_type'],
                    'document_id': result['document_id'],
                    'chunk_info': {
                        'chunk_id': result['chunk_id'],
                        'chunk_index': result['chunk_index'],
                        'total_chunks': result['total_chunks'],
                        'chunk_length': result['chunk_length']
                    },
                    'metadata': {
                        'upload_timestamp': result['upload_timestamp']
                    }
                }
                formatted_results.append(formatted_result)
            
            # Calculate search time
            end_time = datetime.now()
            search_result['search_time'] = (end_time - start_time).total_seconds()
            
            search_result['results'] = formatted_results
            search_result['total_results'] = len(formatted_results)
            search_result['success'] = True
            
            logger.info(f"✅ Search completed: {len(formatted_results)} results in {search_result['search_time']:.3f}s")
            
        except Exception as e:
            search_result['error'] = f"Search failed: {str(e)}"
            logger.error(f"❌ Search error: {e}")
        
        return search_result
    
    def advanced_search(self, query: str, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform advanced search with additional filtering and processing.
        
        Args:
            query: Search query string
            filters: Advanced filtering options
            
        Returns:
            Enhanced search results
        """
        # Default filters
        filters = filters or {}
        search_filters = {
            'top_k': filters.get('top_k', Config.TOP_K_RESULTS),
            'score_threshold': filters.get('score_threshold', Config.SIMILARITY_THRESHOLD),
            'document_types': filters.get('document_types', []),
            'date_range': filters.get('date_range', {}),
            'document_names': filters.get('document_names', [])
        }
        
        # Build document filter
        document_filter = {}
        
        if search_filters['document_types']:
            # Note: This would need to be implemented as multiple searches 
            # since Qdrant filters work on individual conditions
            pass
        
        # Perform basic search first
        results = self.search(
            query=query,
            top_k=search_filters['top_k'],
            score_threshold=search_filters['score_threshold'],
            document_filter=document_filter
        )
        
        if not results['success']:
            return results
        
        # Apply post-processing filters
        filtered_results = []
        for result in results['results']:
            # Filter by document type
            if search_filters['document_types']:
                if result['document_type'] not in search_filters['document_types']:
                    continue
            
            # Filter by document name
            if search_filters['document_names']:
                if result['document_name'] not in search_filters['document_names']:
                    continue
            
            filtered_results.append(result)
        
        results['results'] = filtered_results
        results['total_results'] = len(filtered_results)
        
        return results
    
    def get_context_for_query(self, query: str, max_context_length: int = 50000) -> str:
        """
        Get relevant context chunks for a query to use with LLM.
        
        Args:
            query: Search query
            max_context_length: Maximum length of context to return allow for  relevant chunks, other context will be for chat history later
            
        Returns:
            Concatenated context string
        """
        search_results = self.search(query, top_k=Config.TOP_K_RESULTS, get_entire_text=True)
        
        if not search_results['success'] or not search_results['results']:
            return ""
        
        context_parts = []
        current_length = 0
        # Track unique documents by name and their best score
        unique_documents = {}
        
        # First pass: find top 5 unique documents by score
        for result in search_results['results']:
            document_name = result['document_name']
            score = result['score']
            
            if document_name not in unique_documents or score > unique_documents[document_name]['score']:
                unique_documents[document_name] = {
                    'score': score,
                    'entire_text': result['entire_text']
                }
        for doc_name, doc_info in unique_documents.items():
            logger.info(f"DOCUMENT {doc_name}: {doc_info['score']}")
        # Sort by score and take top 5
        top_documents = sorted(unique_documents.items(), key=lambda x: x[1]['score'], reverse=True)[:3]
        
        encoding = tiktoken.encoding_for_model("gpt-4")
        # Add content from top documents
        for i, (doc_name, doc_info) in enumerate(top_documents):
            text = doc_info['entire_text']
            text_length = len(encoding.encode(text))
            logger.info(f"🔍 {doc_name} Text length: {text_length}")
            
            # Check if adding this text would exceed max length
            if current_length + text_length > max_context_length:
                # Try to add partial text
                remaining_length = max_context_length - current_length
                if remaining_length > 100:  # Only add if substantial text can fit
                    text = text[:remaining_length] + "..."
                    context_parts.append(f"DOCUMENT {i+1}: [{doc_name}] \n {text}")
                break
            
            context_parts.append(f"DOCUMENT {i+1}: {doc_name} \n {text}")
            current_length += text_length
        logger.info(f"🔍 Context length: {current_length}")
        
        return "\n--------------------------------------------\n".join(context_parts), top_documents
    
    def answer_question_with_context(self, question: str, use_openai: bool = True) -> Dict[str, Any]:
        """
        Answer a question using retrieved context and LLM.
        
        Args:
            question: Question to answer
            use_openai: Whether to use OpenAI for generating the answer
            
        Returns:
            Answer with context and metadata
        """
        result = {
            'question': question,
            'answer': '',
            'context_used': '',
            'source_documents': [],
            'confidence': 0.0,
            'success': False,
            'error': None
        }
        
        try:
            # Get relevant context  
            context, top_documents = self.get_context_for_query(question)
            if not context:
                result['error'] = "No relevant context found for the question"
                return result
            
            result['context_used'] = context
            # add source documents to result
            result['source_documents'] = [
                {
                    'name': document_name,
                    'score': document_info['score']
                }
                for document_name, document_info in top_documents
            ]
            
            if use_openai and Config.OPENAI_API_KEY:
                # Use OpenAI to generate answer
                prompt = f"""Dựa vào context được cung cấp, trả lời câu hỏi 1 cách chính xác. Nếu context không đủ thông tin để trả lời câu hỏi, hãy nói rằng context không đủ dữ kiện để trả lời câu hỏi.
                Context:
                {context}

                Câu hỏi: {question}
                Ghi nhớ rằng phải trả lời câu hỏi bằng tiếng Việt.
                Trả lời:"""
                
                try:
                    async def get_response(prompt):
                        client = AsyncOpenAI(api_key=Config.OPENAI_API_KEY)
                        # logger.info(f"🔍 Prompt: {prompt}")
                        response = await client.chat.completions.create(
                            model="gpt-4.1-mini",
                            messages=[
                                    {"role": "system", "content": "Bạn là một trợ lý hữu ích giúp trả lời câu hỏi dựa trên context được cung cấp. Hãy trả lời chính xác và ghi nhận nguồn tài liệu khi cần thiết."},
                                    {"role": "user", "content": prompt}
                            ]
                        )
                        content = response.choices[0].message.content
                        return content.strip() if content else "OPENAI API OUTPUT ERROR"
                    
                    response = asyncio.run(get_response(prompt))
                    result['answer'] = response
                    
                except Exception as e:
                    logger.error(f"❌ OpenAI API error: {e}")
                    result['answer'] = "I found relevant context but couldn't generate an answer. Please check the context below."
            else:
                # Provide context without LLM generation
                result['answer'] = "Here is the relevant information I found:"
                 
            result['success'] = True
            
        except Exception as e:
            result['error'] = f"Failed to answer question: {str(e)}"
            logger.error(f"❌ Question answering error: {e}")
        
        return result

    def search_quotes(self, query: str, top_k: Optional[int] = None, score_threshold: Optional[float] = None):
        """
        Search for relevant quotes/sentences from documents using OpenAI.
        
        Args:
            query: Search query string
            top_k: Number of chunks to search through
            score_threshold: Minimum similarity score
            
        Returns:
            Extracted quotes with metadata
        """
        quote_result = {
            'query': query,
            'quotes': None,
            'success': False,
            'error': None
        }
        
        start_time = datetime.now()
        
        try:
            if not query.strip():
                quote_result['error'] = "Query cannot be empty"
                return quote_result
            
            if not Config.OPENAI_API_KEY:
                quote_result['error'] = "OpenAI API key not configured. Quote extraction requires OpenAI."
                return quote_result
            
            # Step 1: Get relevant chunks
            search_results = self.search(
                query=query,
                top_k=top_k or 5,
                score_threshold=score_threshold or 0.3
            )
            
            if not search_results['success'] or not search_results['results']:
                quote_result['error'] = "No relevant documents found"
                return quote_result
            
            # Step 2: Extract quotes from each chunk using OpenAI
            context= ""
            for result in search_results['results']:
                context += f"[{result['document_name']}] {result['text']}\n"
                
                # Create prompt for quote extraction
            prompt = f"""Từ các văn bản sau, hãy trích xuất các câu hoặc đoạn văn cụ thể liên quan trực tiếp đến câu hỏi: "{query}"
            Các văn bản liên quan:
            {context}

            Yêu cầu:
            1. Chỉ trích xuất các câu/đoạn văn có liên quan trực tiếp đến nội dung câu hỏi
            2. Giữ nguyên nội dung, không thay đổi từ ngữ
            3. Mỗi trích dẫn trên một dòng, bắt đầu bằng "- "
            4. Nếu không có thông tin liên quan, trả về "Không có thông tin liên quan"
            5. Mỗi trích dẫn phải đính kèm thêm tên tài liệu
            6. Nếu có 2 trích dẫn giống hệt nhau trong cùng 1 tài liệu thì chỉ lấy 1 trích dẫn
            7. Câu trả lời phải được trả lời bằng tiếng Việt dựa theo JSON format là 1 list các dict như sau:"""+"""
            [
                {
                    "document_name": "Tên tài liệu 1",
                    "quote": "Nội dung trích dẫn"
                },
                {
                    "document_name": "Tên tài liệu 2",
                    "quote": "Nội dung trích dẫn"
                }
            ]
            NHỚ PHẢI TRẢ VỀ KẾT QUẢ DƯỚI DẠNG JSON CỦA 1 LIST CÁC DICT NHƯ TRÊN, NẾU KHÔNG CÓ KẾT QUẢ THÌ TRẢ VỀ LIST TRỐNG [].
            Trích dẫn:"""
                
            # try:
            async def extract_quotes(prompt):
                client = AsyncOpenAI(api_key=Config.OPENAI_API_KEY)
                logger.info(f"🔍 Prompt: {prompt}")
                retry = 3
                while retry > 0:
                    try:
                        response = await client.chat.completions.create(
                            model="gpt-4o-mini",
                            response_format={"type": "json_object"},
                            messages=[
                                {"role": "system", "content": "Bạn là trợ lý chuyên trích xuất thông tin chính xác từ văn bản. Chỉ trích xuất những phần liên quan trực tiếp và giữ nguyên nội dung gốc."},
                                {"role": "user", "content": prompt}
                            ]
                        )
                        content = response.choices[0].message.content
                        if content:
                            parsed_content = json.loads(content)['result']
                            return parsed_content if parsed_content else []
                    except Exception as e:
                        logger.error(f"❌ Error extracting quotes: {e}")
                        retry -= 1
                return []
            
            extracted_quotes = asyncio.run(extract_quotes(prompt))                
            logger.info(f"🔍 Extracted quotes: {extracted_quotes}, type: {type(extracted_quotes)}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to extract quotes from chunk: {e}")
        
        
        # Calculate search time
        end_time = datetime.now()
        quote_result['search_time'] = (end_time - start_time).total_seconds()
        
        quote_result['quotes'] = extracted_quotes
        quote_result['success'] = True
        
        logger.info(f"✅ Quote extraction completed in {quote_result['search_time']:.3f}s")
            
        # except Exception as e:
        #     quote_result['error'] = f"Quote extraction failed: {str(e)}"
        #     logger.error(f"❌ Quote extraction error: {e}")
        
        return quote_result
    
    def get_similar_documents(self, query_text: str = None, document_id: str = None, chunk_ids: List[str] = None) -> List[Dict[str, Any]]:
        """
        Find documents similar to a given query text, document, or specific chunks.
        
        Args:
            query_text: Text query to find similar documents
            document_id: ID of the reference document
            chunk_ids: List of specific chunk IDs to compare against
            
        Returns:
            List of similar documents
        """
        if not query_text and not document_id and not chunk_ids:
            logger.error("❌ Either query_text, document_id, or chunk_ids must be provided")
            return []
        if isinstance(chunk_ids, str):
            chunk_ids = [chunk_ids]

        # try:
        ref_embedding = None
        
        ref_embedding = self.embedding_manager.get_query_embedding(query_text)
        if not ref_embedding:
            logger.error("❌ Failed to generate embedding for query text")
            return []

        if document_id:
            list_chunk = self.qdrant_manager.get_chunks_by_document_id(document_id)
            if not list_chunk:
                logger.error(f"❌ No chunks found for document ID: {document_id}")
                return []
            chunk_embeddings = [chunk['vector'] for chunk in list_chunk if 'vector' in chunk]
            if not chunk_embeddings:
                logger.error(f"❌ No embeddings found for document ID: {document_id}")
                return []
            
        if chunk_ids:
            list_chunk = self.qdrant_manager.get_chunks_by_chunk_ids(chunk_ids)
            if not list_chunk:
                logger.error(f"❌ No chunks found for chunk IDs: {chunk_ids}")
                return []
            chunk_embeddings = [chunk['vector'] for chunk in list_chunk if 'vector' in chunk]
            if not chunk_embeddings:
                logger.error(f"❌ No embeddings found for chunk IDs: {chunk_ids}")
                return []
            ref_embedding = [sum(values) / len(values) for values in zip(*chunk_embeddings)]
            
            if not chunk_embeddings:
                logger.error("❌ No embeddings found for provided chunk IDs")
                return []
            
        #calculate similarity between ref_embedding and chunk_embeddings
        similarity_scores = [cosine_similarity(ref_embedding, chunk['vector'])[0][0] for chunk in chunk_embeddings]
        #sort by similarity score
        list_chunk_with_sims = []
        for chunk, sim in zip(chunk_embeddings, similarity_scores):
            chunk['similarity_score'] = sim
            list_chunk_with_sims.append(chunk)
        #sort by similarity score
        list_chunk_with_sims.sort(key=lambda x: x['similarity_score'], reverse=True)
        return list_chunk_with_sims
        # except Exception as e:
        #     logger.error(f"❌ Error finding similar documents: {e}")
        #     return []



def get_query_suggestions(partial_query: str, max_suggestions: int = 5) -> List[str]:
    """
    Get query suggestions based on indexed content.
    
    Args:
        partial_query: Partial query string
        max_suggestions: Maximum number of suggestions
        
    Returns:
        List of suggested queries
    """
    # This is a simplified implementation
    # In a real system, you might want to implement proper query completion
    suggestions = []
    
    if len(partial_query) < 3:
        return suggestions
    
    # Generate some basic suggestions
    base_suggestions = [
        f"What is {partial_query}?",
        f"How does {partial_query} work?",
        f"Examples of {partial_query}",
        f"Definition of {partial_query}",
        f"{partial_query} process"
    ]
    
    # Filter and return relevant suggestions
    suggestions = [s for s in base_suggestions if len(s) <= 100][:max_suggestions]
    
    return suggestions 