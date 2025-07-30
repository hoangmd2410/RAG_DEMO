import logging
from typing import Dict, Any, Optional
from datetime import datetime
import json
from rag.querying import QueryProcessor
from llm_processor import LLMProcessor, ApiProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryProcessorComparator:
    """Handles comparison of top-k search results from QueryProcessor for similarities and conflicts using LLMProcessor with Qwen-2.5-3B-Instruct."""
    
    def __init__(self, user_prompt_file: str = "comparison_prompt.txt"):
        self.query_processor = QueryProcessor()
        self.llm_processor = ApiProcessor()
        
        # Load system prompt from file
        self.system_prompt = """Bạn là một trợ lý pháp lý chuyên nghiệp, chuyên về so sánh tài liệu và phát hiện xung đột.
                                Nhiệm vụ của bạn là phân tích nhiều điều khoản tài liệu và xác định các mâu thuẫn hoặc điểm tương đồng,
                                trả về kết quả theo định dạng JSON có cấu trúc."""

        
        # Load user prompt from file
        try:
            with open(user_prompt_file, 'r', encoding='utf-8') as f:
                self.user_prompt_template = f.read().strip()
            logger.info(f"Successfully loaded user prompt from {user_prompt_file}")
        except FileNotFoundError:
            logger.error(f"User prompt file {user_prompt_file} not found")
            raise FileNotFoundError(f"User prompt file {user_prompt_file} not found")
        except Exception as e:
            logger.error(f"Failed to read user prompt file {user_prompt_file}: {str(e)}")
            raise IOError(f"Failed to read user prompt file: {str(e)}")
    def compare_pair_chunk(self,chunk_a, chunk_b):

        query = chunk_a['text']
        
        

        doc_name = chunk_b['document_name']

        doc_id = chunk_b['document_id']

        if doc_name not in query:
            # try:
                context = chunk_b['text']

                prompt = self.user_prompt_template.format(query=query, context_text=context)


                response = self.llm_processor.process(
                    user_input=prompt,
                    system_prompt=self.system_prompt
                )
                res = json.loads(response)

                if isinstance(res,list):
                    res = res[0]
                if res != {}:
                    res['document_name'] = doc_name
                    res['document_id'] = doc_id         
                    return res
            # except:
            #     return None
        return None


    def compare_query_results(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.7,
        document_filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Compare top-k documents from QueryProcessor search results to identify similarities and conflicts.
        
        Args:
            query: Search query string
            top_k: Number of results to compare
            score_threshold: Minimum similarity score
            document_filter: Optional filter for specific documents
            
        Returns:
            Structured analysis of similarities and conflicts among search results
        """
        result = {
            'query': query,
            'findings': [],
            'total_results': 0,
            'processing_time': 0,
            'success': False,
            'error': None
        }
        
        

        if not query.strip():
            result['error'] = "Query cannot be empty"
            return result
            
        # Step 1: Perform search using QueryProcessor
        search_results = self.query_processor.search(
            query=query,
            top_k=top_k,
            score_threshold=score_threshold,
            document_filter=document_filter
        )
        
        if not search_results['success']:
            result['error'] = search_results['error'] or "Search failed"
            return result
            
        if not search_results['results']:
            result['error'] = "No search results found"
            return result
            
        # Step 2: Prepare context from search results

        compare_output =   []
        
        for result in search_results['results']:

            doc_name = result['document_name']

            doc_id = result['document_id']

            if doc_name not in query:
                context = result['text']
        
                
            

                # Format the user prompt using the template
                # try:
                prompt = self.user_prompt_template.format(query=query, context_text=context)


                response = self.llm_processor.process(
                    user_input=prompt,
                    system_prompt=self.system_prompt
                )
                
                res = json.loads(response)

                if isinstance(res,list):
                    res = res[0]
                if res != {}:
                    res['document_name'] = doc_name
                    res['document_id'] = doc_id         
                    compare_output.append(res)       

                # except:
                #     result['error'] = f"LLM error"
        return compare_output

