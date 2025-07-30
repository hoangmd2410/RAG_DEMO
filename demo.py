import gradio as gr
import logging
from rag.query_processor_comparator import QueryProcessorComparator
from rag.indexing import DocumentIndexer, clean_text, validate_file
from rag.querying import QueryProcessor
from llm_processor import ApiProcessor, LLMProcessor
from config import Config
import os
from datetime import datetime
from typing import List, Dict
import uuid
import numpy as np
import json
from concurrent.futures import ThreadPoolExecutor
from rag.utils import crop_all_pages, encode_image
from rag.nanonet_ocr import ocr_list_pages
import time
import asyncio
from gradio_tabs import SemanticSearchTab, ChatWithDocumentTab, ExtractInformationTab

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load chunking system prompt
with open(Config.CHUNKING_SYSTEM_PROMPT, 'r', encoding='utf-8') as f:
    chunking_system_prompt = f.read().strip()

with open('./sys_prompts/chat_prompt.md', 'r') as f:
    chat_system_prompt = f.read().strip()

api_processor = ApiProcessor()
llm_processor = LLMProcessor()
indexer = DocumentIndexer(api_processor, chunking_system_prompt)
comparator = QueryProcessorComparator(
    user_prompt_file="./sys_prompts/compare_doc_prompt.md"
)
indexer = DocumentIndexer(api_processor, chunking_system_prompt)
# Override with shared instance to avoid duplicate loading
from rag.embedding_manager import EmbeddingManager
shared_embedding_manager = EmbeddingManager()
indexer.embedding_manager = shared_embedding_manager

comparator = QueryProcessorComparator(
    user_prompt_file="./sys_prompts/compare_doc_prompt.md"
)
# Override with shared instance to avoid duplicate loading
comparator.query_processor.embedding_manager = shared_embedding_manager

list_doc_names = indexer.qdrant_manager.list_document_names()


def index_uploaded_document(file) -> str:
    """
    Index an uploaded Markdown document using DocumentIndexer.
    
    Args:
        file: Uploaded file object from Gradio
        
    Returns:
        Formatted string with indexing results
    """
    try:
        if not file:
            return "❌ No file uploaded"
            
        file_path = file.name
        if not file_path.endswith('.pdf'):
            return "❌ File must be a Markdown (.pdf) file"
            
        result = indexer.index_document(file_path)
        
        if not result['success']:
            return f"❌ Indexing failed: {result['error']}"
            
        return (
            f"✅ Successfully indexed document: {result['filename']}\n"
            f"📊 Stats:\n"
            f"- Document ID: {result['document_id']}\n"
            f"- Chunks: {result['chunks_count']}\n"
            f"- Total Characters: {result['total_characters']}\n"
            f"- Processing Time: {result['processing_time']:.2f} seconds"
        )
    except Exception as e:
        logger.error(f"Indexing error: {str(e)}")
        return f"❌ Indexing error: {str(e)}"

def extract_document(file):
    try:
        if not file:
            return "❌ No file uploaded"
            
        file_path = file.name
        if not file_path.endswith('.pdf'):
            return "❌ File must be a pdf (.pdf) file"
        filename = os.path.basename(file_path)
        validation_result = validate_file(file_path)
        document_metadata = {
            'document_id': str(uuid.uuid4()),
            'filename': filename,
            'file_type': validation_result['extension'],
            'file_size_mb': validation_result['size_mb'],
            'timestamp': datetime.now().isoformat(),
            'indexed_by': 'semantic_search_pipeline'
        }
                
        # Convert PDF to images (base64 encoded)
        encoded_images = crop_all_pages(file_path)
        
        # Use OCR to extract text from images
        async def extract_text_with_ocr():
            ocr_text = await ocr_list_pages(encoded_images)
            return ocr_text
        
        # Get OCR text
        text = asyncio.run(extract_text_with_ocr())
        

        text = clean_text(text)

        chunks = indexer.chunk_text(text)

        chunks_payloads = []

        for i, chunk in enumerate(chunks):
            # Get chunk text (handle both old and new formats)
            chunk_text = chunk.get('text', chunk) if isinstance(chunk, dict) else str(chunk)
            chunk_length = len(chunk_text)
            payload = {
                'document_id': document_metadata['document_id'],
                'chunk_id': chunk.get('id', i) if isinstance(chunk, dict) else i,
                'text': chunk['text'],
                'chunk_length': chunk_length,
                'chunk_start_pos': chunk.get('start_pos', 0) if isinstance(chunk, dict) else 0,
                'chunk_end_pos': chunk.get('end_pos', chunk_length) if isinstance(chunk, dict) else chunk_length,
                'document_name': document_metadata.get('filename', 'unknown'),
                'document_type': document_metadata.get('file_type', 'unknown'),
                'upload_timestamp': document_metadata.get('timestamp'),
                'total_chunks': len(chunks),
                'chunk_index': i
            }
            chunks_payloads.append(payload)
        return chunks_payloads


    except Exception as e:
        logger.error(f"Indexing error: {str(e)}")
        return f"❌ Indexing error: {str(e)}"

def search_documents(query: str, max_results: int = 5, score_threshold: float = 0.7) -> str:
    """
    Perform semantic search using QueryProcessor.
    
    Args:
        query: Search query string
        max_results: Number of documents to return
        score_threshold: Minimum similarity score
    
    Returns:
        Formatted string with search results
    """
    try:
        result = comparator.query_processor.search(
            query=query,
            top_k=max_results,
            score_threshold=score_threshold
        )
        
        if not result['success']:
            return f"❌ Search failed: {result['error']}"
        
        if not result['results']:
            return f"📊 No results found for query: '{query}'"
        
        # Format results
        results_text = f"📊 **Search Results for Query:** {query}\n\n"
        for idx, res in enumerate(result['results'], 1):
            results_text += (
                f"**Result {idx}**\n"
                f"- **Document:** {res['document_name']} (ID: {res['document_id']})\n"
                f"- **Chunk:** {res['chunk_info']['chunk_index']} of {res['chunk_info']['total_chunks']}\n"
                f"- **Score:** {res['score']}\n"
                f"- **Text:** {res['text']}\n"
                f"- **Type:** {res['document_type']}\n"
                f"- **Uploaded:** {res['metadata']['upload_timestamp']}\n\n"
            )
        
        results_text += f"**Total Results:** {result['total_results']}\n"
        results_text += f"**Search Time:** {result['search_time']:.2f} seconds"
        return results_text
        
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return f"❌ Search error: {str(e)}"


def compare_chunk(query: str, max_results: int = 5, score_threshold: float = 0.7) -> str:
    """
    Compare top-k search results for similarities and conflicts using QueryProcessorComparator.
    
    Args:
        query: Search query string
        max_results: Number of documents to compare
        score_threshold: Minimum similarity score
    
    Returns:
        Formatted string with comparison results
    """
    # try:
    result = comparator.compare_query_results(
        query=query,
        top_k=max_results,
        score_threshold=score_threshold
    )
    output = ""
    if 'error' in result:
        return output
    else:
        for res in result:

            formatted_res = json.dumps(res, indent=4, ensure_ascii=False)
            output += f"{formatted_res} \n ==================\n"
    return output
        
    # except Exception as e:
    #     logger.error(f"Comparison error: {str(e)}")
    #     return f"❌ Comparison error: {str(e)}"
def compare_document(file,max_results: int = 5, score_threshold: float = 0.7):
    start = time.time()
    chunks_payloads = extract_document(file)
    end = time.time()
    print(end-start)

    print(len(chunks_payloads))


    output_parts = []

    def _process_chunk_for_comparison(chunk):
        try:
            return compare_chunk(str(chunk), max_results, score_threshold)
        except Exception as e:
            logger.error(f"❌ Error comparing chunk: {e}")
            return None

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(_process_chunk_for_comparison, chunk) for chunk in chunks_payloads]
        
        for future in futures:
            result = future.result()  # Blocks until ready
            if result:
                output_parts.append(result)

    end = time.time()
    print(f"⏱️ Total compare time: {end - start:.2f}s")

    return "\n".join(output_parts).strip()

def _process_pair_for_comparison(pair, comparator_obj):
    """
    Helper function to process a single similar pair.
    Designed to be run by a thread.
    """
    chunk_a, chunk_b = pair
    res = comparator_obj.compare_pair_chunk(chunk_a, chunk_b)

    if res is not None:
        res['source_document_name'] = chunk_a['document_name']
        res['source_document_id'] = chunk_a['document_id']
        res['source_chunk_id'] = chunk_a['chunk_id']
        res['chunk_id'] = chunk_b['chunk_id']
        formatted_res = json.dumps(res, indent=4, ensure_ascii=False)
        # print(formatted_res)
        return f"{formatted_res} \n ==================\n"
    return None

def compare_document_groups(group_a_files: List, group_b_files: List, thresshold: int = 0.8) -> str:
    """
    Compare two groups of documents by extracting chunks, generating embeddings,
    and comparing content.
    
    Args:
        group_a_files: List of files for group A
        group_b_files: List of files for group B
    
    Returns:
        Formatted string containing comparison results
    """
    
    group_a_chunks = []
    for doc_name in group_a_files:
        group_a_chunks.extend(indexer.qdrant_manager.get_document_by_name(doc_name))

    group_b_files = [f for f in group_b_files if f not in group_a_files]

    similar_pairs = []
    for chunk_a in group_a_chunks:
        query_embedding = chunk_a['vector']

        # Search for similar chunks filtered to Group B documents
        for doc_name in group_b_files:
            all_results = indexer.qdrant_manager.search_similar(
                query_embedding=query_embedding,
                top_k=5,  # Large limit to get all possible matches
                score_threshold=thresshold,
                document_filter={'document_name': doc_name}  # Filter to Group B names (OR condition)
            )
            for res in all_results:
                similar_pairs.append((chunk_a, res))

    output_parts = []
    with ThreadPoolExecutor(max_workers=20) as executor: 
        futures = [executor.submit(_process_pair_for_comparison, pair, comparator) for pair in similar_pairs]
        for future in futures:
            result = future.result() # Blocks until the result is ready
            if result is not None:
                output_parts.append(result)

    return "".join(output_parts) # Join all parts into a single string
        
    



def generate(prompt, history):
    # Generate the answer using the QuestionAnswering class with history

    result = comparator.query_processor.search(
            query=prompt,
            top_k=5,
            score_threshold=0.4
    )

    context = f"📊 **Search Results for Query:"
    for idx, res in enumerate(result['results'], 1):
        context += (
            f"**Result {idx}**\n"
            f"- **Document:** {res['document_name']} (ID: {res['document_id']})\n"
            f"- **Chunk:** {res['chunk_info']['chunk_index']} of {res['chunk_info']['total_chunks']}\n"
            f"- **Score:** {res['score']}\n"
            f"- **Text:** {res['text']}\n"
            f"- **Type:** {res['document_type']}\n"
            f"- **Uploaded:** {res['metadata']['upload_timestamp']}\n\n"
        )
    stream = llm_processor.answer_question(
        prompt, history, top_k=5, sys_prompt=chat_system_prompt, context = context
    )  # Returns a generator of tokens

    # Stream the generated text progressively
    stream_buffer = ""
    for token in stream:
        stream_buffer += token
        yield stream_buffer  # Yield the full accumulated text so far
        # time.sleep(0.05)  # Adjust delay for streaming speed

def create_interface():
    """Create a Gradio interface for document upload, search, comparison, and chat."""
    search_tab = SemanticSearchTab(comparator.query_processor)
    chat_tab = ChatWithDocumentTab()
    extract_tab = ExtractInformationTab()

    with gr.Blocks(title="Document Comparison and Chat Demo") as interface:
        gr.Markdown("""
        # 🔍 Document Comparison and Chat Demo
        
        Upload Markdown documents, search for relevant content, compare them for conflicts and similarities, or chat with the AI using **Qwen-2.5-3B-Instruct**.

        """)
        
        # with gr.Tab("📂 Upload Document"):
        #     gr.Markdown("### Upload and Index Markdown Document")
        #     gr.Markdown("*Upload a .md file to index its contents for search and comparison.*")
            
        #     file_input = gr.File(
        #         label="Upload Markdown File",
        #         file_types=[".md",".pdf"]
        #     )
            
        #     index_btn = gr.Button("📂 Index Document", variant="primary")
            
        #     index_output = gr.Textbox(
        #         label="Indexing Results",
        #         lines=10,
        #         interactive=False
        #     )
            
        #     index_btn.click(
        #         fn=index_uploaded_document,
        #         inputs=[file_input],
        #         outputs=index_output
        #     )
        with gr.Tab("📂 Compare Document"):
            gr.Markdown("### Upload PDF Document")
            gr.Markdown("*Upload a .pdf file to index its contents for search and comparison.*")
            
            file_input = gr.File(
                label="Upload pdf File",
                file_types=[".pdf"]
            )
            with gr.Row():
                max_results = gr.Number(
                    label="Max Documents to Return",
                    value=5,
                    minimum=1,
                    maximum=50,
                    precision=0
                )
                score_threshold = gr.Number(
                    label="Score Threshold",
                    value=0.7,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.1
                )
            index_btn = gr.Button("📂 Check Document", variant="primary")
            
            index_output = gr.Textbox(
                label="Indexing Results",
                lines=20,
                interactive=False)
            
     

            
            index_btn.click(
                fn=compare_document,
                inputs=[file_input,max_results,score_threshold],
                outputs=index_output
            )
 
        # with gr.Tab("🔍 Search Documents"):
        #     gr.Markdown("### Search Documents")
        #     gr.Markdown("*Enter a query to search for relevant documents in the indexed database.*")
            
        #     query_input = gr.Textbox(
        #         label="Search Query",
        #         placeholder="Enter a query (e.g., 'contract termination conditions')...",
        #         lines=2
        #     )
            
        #     with gr.Row():
        #         max_results = gr.Number(
        #             label="Max Documents to Return",
        #             value=5,
        #             minimum=1,
        #             maximum=10,
        #             precision=0
        #         )
        #         score_threshold = gr.Number(
        #             label="Score Threshold",
        #             value=0.7,
        #             minimum=0.0,
        #             maximum=1.0,
        #             step=0.1
        #         )
            
        #     search_btn = gr.Button("🔍 Search Documents", variant="primary")
            
        #     search_output = gr.Textbox(
        #         label="Search Results",
        #         lines=15,
        #         interactive=False
        #     )
            
        #     search_btn.click(
        #         fn=search_documents,
        #         inputs=[query_input, max_results, score_threshold],
        #         outputs=search_output
        #     )
        
        with gr.Tab("Compare Group of document"):
            gr.Markdown("### Select group of document to compare")
            with gr.Row():
                group_a_selector = gr.Dropdown(
                    choices=list_doc_names,
                    multiselect=True,
                    max_choices=5,
                    label="Select group of document"
                )
                group_b_selector = gr.Dropdown(
                    choices=list_doc_names,
                    multiselect=True,
                    max_choices=5,
                    label="Select group of document"
                )
            with gr.Row():
                score_threshold = gr.Number(
                    label="Score Threshold",
                    value=0.7,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.1
                )
            
            compare_btn = gr.Button("📊 Compare Documents", variant="primary")
            
            compare_output = gr.Textbox(
                label="Comparison Results",
                lines=15,
                interactive=False
            )
            
            compare_btn.click(
                fn=compare_document_groups,
                inputs=[group_a_selector, group_b_selector, score_threshold],
                outputs=compare_output
            )
        search_tab.get_frontend()
        chat_tab.get_frontend()
        extract_tab.get_frontend()
            
    return interface

def main():
    """Main function to run the Gradio demo."""
    try:
        interface = create_interface()
        interface.launch(
            server_name="0.0.0.0",
            server_port=6007,
            show_error=True,
            share=True
        )
    except Exception as e:
        logger.error(f"❌ Failed to start application: {e}")
        print(f"Error: {e}")
        print("\nPlease ensure:")
        print("1. Qdrant is running: docker run -p 6333:6333 qdrant/qdrant")
        print("2. Dependencies are installed: pip install transformers torch flash-attn gradio")
        print("3. Required files (querying.py, llm_processor.py, config.py, system_prompt.txt, comparison_prompt.txt) are in place")

if __name__ == "__main__":
    main()