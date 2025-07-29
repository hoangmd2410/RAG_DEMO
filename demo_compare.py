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

# Create shared embedding manager
from rag.embedding_manager import EmbeddingManager
shared_embedding_manager = EmbeddingManager()

indexer = DocumentIndexer(api_processor, chunking_system_prompt)
# Override with shared instance to avoid duplicate loading
indexer.embedding_manager = shared_embedding_manager

comparator = QueryProcessorComparator(
    user_prompt_file="./sys_prompts/compare_doc_prompt.md"
)
# Override with shared instance to avoid duplicate loading
comparator.query_processor.embedding_manager = shared_embedding_manager

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
        text = indexer.document_processor.extract_text(file_path)

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
    try:
        result = comparator.compare_query_results(
            query=query,
            top_k=max_results,
            score_threshold=score_threshold
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Comparison error: {str(e)}")
        return f"❌ Comparison error: {str(e)}"
def compare_document(file):

    chunks_payloads = extract_document(file)

    results = ""

    for chunk in chunks_payloads:

        result = compare_chunk(str(chunk))

        if not isinstance(result, dict):

            chunk = str(chunk).replace('```json','')
            result = f"# input query \n ```{chunk}```\n #compared result \n {result} \n ==================\n"

            results += result
    print(results)
    return results

def generate(prompt, history):
    # Generate the answer using the QuestionAnswering class with history
    stream = llm_processor.answer_question(
        prompt, history, top_k=5
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
        
        ## Features:
        - 📂 Upload and index Markdown (.md) documents
        - 🧠 Semantic search with Qwen embeddings
        - 📊 Compare documents for conflicts and similarities
        - 💬 Chat with the AI for general queries
        """)        
        with gr.Tab("📂 Upload Document"):
            gr.Markdown("### Upload and Index Markdown Document")
            gr.Markdown("*Upload a .md file to index its contents for search and comparison.*")
            
            file_input = gr.File(
                label="Upload Markdown File",
                file_types=[".md"]
            )
            
            index_btn = gr.Button("📂 Index Document", variant="primary")
            
            index_output = gr.Textbox(
                label="Indexing Results",
                lines=10,
                interactive=False
            )
            
            index_btn.click(
                fn=index_uploaded_document,
                inputs=[file_input],
                outputs=index_output
            )
        with gr.Tab("📂 Compare Document"):
            gr.Markdown("### Upload PDF Document")
            gr.Markdown("*Upload a .pdf file to index its contents for search and comparison.*")
            
            file_input = gr.File(
                label="Upload pdf File",
                file_types=[".pdf"]
            )
            
            index_btn = gr.Button("📂 Check Document", variant="primary")
            
            index_output = gr.Textbox(
                label="Indexing Results",
                lines=20,
                interactive=False)
            # with gr.Accordion("Indexing Results", open=True):
            #     index_output = gr.Markdown(
            #         value="Upload a PDF and click 'Check Document' to see results.",
            #         show_label=False
            #     )
            
            def run_with_progress(file):
                # yield "⏳ Processing... please wait..."
                # Then return actual results
                return compare_document(file)

            
            index_btn.click(
                fn=run_with_progress,
                inputs=file_input,
                outputs=index_output
            )
            # .then(
            #     fn=compare_document,
            #     inputs=file_input,
            #     outputs=index_output
            # )  
        with gr.Tab("🔍 Search Documents"):
            gr.Markdown("### Search Documents")
            gr.Markdown("*Enter a query to search for relevant documents in the indexed database.*")
            
            query_input = gr.Textbox(
                label="Search Query",
                placeholder="Enter a query (e.g., 'contract termination conditions')...",
                lines=2
            )
            
            with gr.Row():
                max_results = gr.Number(
                    label="Max Documents to Return",
                    value=5,
                    minimum=1,
                    maximum=10,
                    precision=0
                )
                score_threshold = gr.Number(
                    label="Score Threshold",
                    value=0.7,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.1
                )
            
            search_btn = gr.Button("🔍 Search Documents", variant="primary")
            
            search_output = gr.Textbox(
                label="Search Results",
                lines=15,
                interactive=False
            )
            
            search_btn.click(
                fn=search_documents,
                inputs=[query_input, max_results, score_threshold],
                outputs=search_output
            )
        
        with gr.Tab("📊 Compare chunk query"):
            gr.Markdown("### Compare chunk for Conflicts and Similarities")
            gr.Markdown("*Enter a query to search for documents and compare their clauses for conflicts or similarities.*")
            
            query_input = gr.Textbox(
                label="Search Query",
                placeholder="Enter a query (e.g., 'contract termination conditions')...",
                lines=2
            )
            
            with gr.Row():
                max_results = gr.Number(
                    label="Max Documents to Compare",
                    value=5,
                    minimum=2,
                    maximum=10,
                    precision=0
                )
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
                fn=compare_chunk,
                inputs=[query_input, max_results, score_threshold],
                outputs=compare_output
            )
        
        with gr.Tab("💬 Chat"):
            gr.ChatInterface(
                fn=generate,
                chatbot=gr.Chatbot(height=500),  # Set a reasonable height for the chatbot
                textbox=gr.Textbox(placeholder="Type something..."),
                submit_btn=gr.Button("Send"),
                # retry_btn=None,  # Disable retry button if not needed
                # undo_btn=None,   # Disable undo button if not needed
                # clear_btn="Clear"  # Optional: Add a clear button
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
            server_port=7860,
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