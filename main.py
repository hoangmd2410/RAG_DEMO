import gradio as gr
import os
from datetime import datetime
from typing import List, Tuple, Any
import logging

# Import our modules
from config import Config, validate_config
from indexing import DocumentIndexer, verify_indexing_setup
from querying import QueryProcessor
from qdrant_setup import check_qdrant_connection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemanticSearchApp:
    """Main application class for the Gradio interface."""
    
    def __init__(self):
        self.indexer = None
        self.query_processor = None
        self.setup_status = None
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize the application components."""
        try:
            # Validate configuration
            validate_config()
            
            # Verify setup
            self.setup_status = verify_indexing_setup()
            
            if self.setup_status['overall_status'] in ['ready', 'partial']:
                self.indexer = DocumentIndexer()
                self.query_processor = QueryProcessor()
                logger.info("✅ Application components initialized")
            else:
                logger.error("❌ Application setup failed")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize components: {e}")
    
    def upload_and_index_document(self, file) -> str:
        """Handle document upload and indexing."""
        if file is None:
            return "❌ No file uploaded"
        
        if not self.indexer:
            return "❌ System not ready. Please check configuration."
        
        try:
            # The file object now contains the path directly
            file_path = file.name if hasattr(file, 'name') else str(file)
            
            # Index the document directly
            result = self.indexer.index_document(file_path)
            
            if result['success']:
                return (f"✅ **Document indexed successfully!**\n"
                        f"📄 **File:** {result['filename']}\n"
                        f"🆔 **Document ID:** {result['document_id']}\n"
                        f"📊 **Statistics:**\n"
                        f"- Chunks created: {result['chunks_count']}\n"
                        f"- Total characters: {result['total_characters']:,}\n"
                        f"- Processing time: {result['processing_time']:.2f} seconds\n"
                        f"The document is now searchable in the system.")
            else:
                return f"❌ **Indexing failed:** {result['error']}"
                
        except Exception as e:
            return f"❌ **Error processing file:** {str(e)}"
    
    def perform_quote_search(self, query: str, max_results: int = 10, score_threshold: float = 0.3) -> str:
        """Perform quote extraction and return results."""
        if not query.strip():
            return "❌ Please enter a search query"
        
        if not self.query_processor:
            return "❌ System not ready. Please check configuration."
        
        try:
            # Perform quote search
            results = self.query_processor.search_quotes(
                query=query,
                top_k=max_results,
                score_threshold=score_threshold
            )
            
            if not results['success']:
                return f"❌ Quote search failed: {results['error']}"
            
            if not results['quotes']:
                return f"💬 No relevant quotes found for query: '{query}'"
            
            # Format results
            results_text = ""
            for quote in results['quotes']:
                document_name = quote['document_name']
                quote_text = quote['quote']
                results_text += f"[{document_name}]: {quote_text}\n"
            
            return results_text
            
        except Exception as e:
            return f"❌ Quote search error: {str(e)}"       
    
    def answer_question(self, question: str, use_ai: bool = True) -> str:
        """Answer a question using the knowledge base."""
        if not question.strip():
            return "❌ Please enter a question"
        
        if not self.query_processor:
            return "❌ System not ready. Please check configuration."
        
        try:
            result = self.query_processor.answer_question_with_context(question, use_ai)
            
            if not result['success']:
                return f"❌ Failed to answer question: {result['error']}"
            
            response = (f"💡 **Question:** {result['question']}\n"
                        f"🤖 **Answer:** {result['answer']}\n"
                        f"📚 **Source Documents:**")
            
            for doc in result['source_documents']:
                response += f"- {doc['name']} (Score: {doc['score']:.3f})\n"
            
            if result['context_used']:
                response += f"\n📖 **Context Used:**\n{result['context_used']}"
            
            return response
            
        except Exception as e:
            return f"❌ Error answering question: {str(e)}"
    
    def get_system_status(self) -> str:
        """Get current system status."""
        if not self.setup_status:
            return "❌ System status unknown"
        
        status_text = (f"🔧 **System Status**\n\n"
                       f"**Overall Status:** {self.setup_status['overall_status'].upper()}\n\n"
                       f"**Component Status:**\n"
                       f"- ✅ Embedding Model: {'Loaded' if self.setup_status['embedding_model_loaded'] else '❌ Not Loaded'}\n"
                       f"- {'✅' if self.setup_status['qdrant_connected'] else '❌'} Qdrant Database: {'Connected' if self.setup_status['qdrant_connected'] else 'Not Connected'}\n"
                       f"- {'✅' if self.setup_status['collection_initialized'] else '❌'} Collection: {'Initialized' if self.setup_status['collection_initialized'] else 'Not Initialized'}\n\n"
                       f"**Configuration:**\n"
                       f"- Embedding Model: {Config.EMBEDDING_MODEL_NAME}\n"
                       f"- Vector Size: {Config.VECTOR_SIZE}\n"
                       f"- Qdrant Host: {Config.QDRANT_HOST}:{Config.QDRANT_PORT}\n"
                       f"- Collection Name: {Config.QDRANT_COLLECTION_NAME}\n")
        
        if self.setup_status['issues']:
            status_text += "⚠️ **Issues:**\n"
            for issue in self.setup_status['issues']:
                status_text += f"- {issue}\n"
        
        # Add collection info if available
        if self.indexer and self.indexer.qdrant_manager.is_connected():
            collection_info = self.indexer.qdrant_manager.get_collection_info()
            if 'error' not in collection_info:
                status_text += f"\n📊 **Collection Statistics:**\n- Documents: {collection_info.get('points_count', 0)} chunks\n"
        
        return status_text
    
    def list_documents(self) -> str:
        """List all indexed documents."""
        if not self.indexer or not self.indexer.qdrant_manager.is_connected():
            return "❌ System not ready or Qdrant not connected"
        
        try:
            documents = self.indexer.qdrant_manager.list_documents()
            
            if not documents:
                return "📝 No documents indexed yet"
            
            docs_text = f"📚 **Indexed Documents ({len(documents)} total):**\n\n"
            
            for i, doc in enumerate(documents, 1):
                docs_text += (f"**{i}. {doc['document_name']}**\n"
                              f"- Type: {doc['document_type']}\n"
                              f"- Chunks: {doc['chunk_count']}/{doc['total_chunks']}\n")
            
            return docs_text
            
        except Exception as e:
            return f"❌ Error listing documents: {str(e)}"
    
    def extract_information(self, json_format: str, document_text: str) -> str:
        """Extract structured information from document text using LLM."""
        if not json_format.strip() or not document_text.strip():
            return "❌ Please provide both JSON format and document text"
        
        if not self.query_processor:
            return "❌ System not ready. Please check configuration."
        
        try:
            from openai import AsyncOpenAI
            import asyncio
            from config import Config
            
            if not Config.OPENAI_API_KEY:
                return "❌ OpenAI API key not configured. Information extraction requires OpenAI."
            
            # Construct the extraction prompt
            prompt = (f"Given the following document text, extract the specified entities according to the JSON format provided. "
                     f"Respond only with valid JSON format.\n\n"
                     f"**Desired JSON Format:**\n{json_format}\n\n"
                     f"**Document Text:**\n\"\"\"\n{document_text}\n\"\"\"\n\n"
                     f"Extract the information and return it in the exact JSON format specified above. "
                     f"If any field cannot be determined from the text, use null for that field.")
            
            # Use OpenAI to extract information
            async def extract_information_async(prompt):
                client = AsyncOpenAI(api_key=Config.OPENAI_API_KEY)
                response = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that extracts information from documents and returns structured JSON data according to the specified format."},
                        {"role": "user", "content": prompt}
                    ]
                )
                content = response.choices[0].message.content
                return content.strip() if content else "{}"
            
            # Get the response
            extracted_data = asyncio.run(extract_information_async(prompt))
            
            return (f"✅ **Information Extraction Completed**\n\n"
                   f"**Extracted Data:**\n```json\n{extracted_data}\n```\n\n"
                   f"The information has been successfully extracted from the document.")
                
        except Exception as e:
            return f"❌ **Error during extraction:** {str(e)}"

def create_interface():
    """Create the Gradio interface."""
    app = SemanticSearchApp()
    
    # Define the interface with simpler components
    with gr.Blocks(
        title="Semantic Search with Qwen Embeddings"
    ) as interface:
        
        gr.Markdown("""
        # 🔍 Semantic Search Pipeline
        
        Upload documents and search through them using **Qwen3-Embedding-0.6B** and **Qdrant** vector database.
        
        ## Features:
        - 📄 Document upload (PDF, DOCX, TXT)
        - 🧠 Semantic search with AI embeddings
        - 💬 Question answering with OpenAI
        - 📊 System monitoring
        """)
        
        with gr.Tabs():
            # Document Upload Tab
            with gr.Tab("📤 Upload Documents"):
                gr.Markdown("### Upload and Index Documents")
                
                file_input = gr.File(
                    label="Select Document (PDF, DOCX, TXT)",
                    file_count="single"
                )
                upload_btn = gr.Button("📤 Upload & Index", variant="primary")
                
                upload_output = gr.Textbox(
                    label="Upload Status",
                    lines=10,
                    interactive=False
                )
                
                upload_btn.click(
                    fn=app.upload_and_index_document,
                    inputs=file_input,
                    outputs=upload_output
                )
            
            # Quote Search Tab
            with gr.Tab("💬 Search Quote"):
                gr.Markdown("### AI-Powered Quote Extraction")
                gr.Markdown("*Find specific sentences and quotes from your documents that are relevant to your query using OpenAI.*")
                
                search_query = gr.Textbox(
                    label="Search Query",
                    placeholder="Enter your search query to find relevant quotes...",
                    lines=2
                )
                
                with gr.Row():
                    max_results = gr.Number(
                        label="Max Chunks to Search",
                        value=5,
                        minimum=1,
                        maximum=10,
                        precision=0
                    )
                    score_threshold = gr.Number(
                        label="Score Threshold",
                        value=0.3,
                        minimum=0.0,
                        maximum=1.0,
                        step=0.1
                    )
                
                search_btn = gr.Button("💬 Extract Quotes", variant="primary")
                
                with gr.Row():
                    search_results = gr.Textbox(
                        label="Relevant Quotes",
                        lines=15,
                        interactive=False
                    )
                
                search_btn.click(
                    fn=app.perform_quote_search,
                    inputs=[search_query, max_results, score_threshold],
                    outputs=[search_results]
                )
            
            # Q&A Tab
            with gr.Tab("💬 Ask Questions"):
                gr.Markdown("### AI-Powered Question Answering")
                
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask a question about your documents...",
                    lines=3
                )
                
                use_ai_toggle = gr.Checkbox(
                    label="Use OpenAI for Answer Generation",
                    value=True
                )
                
                ask_btn = gr.Button("💬 Ask Question", variant="primary")
                
                answer_output = gr.Textbox(
                    label="Answer",
                    lines=12,
                    interactive=False
                )
                
                ask_btn.click(
                    fn=app.answer_question,
                    inputs=[question_input, use_ai_toggle],
                    outputs=answer_output
                )
            
            # Information Extraction Tab (Pipeline 2)
            with gr.Tab("🔍 Extract Information"):
                gr.Markdown("### Automated Information Extraction (Pipeline 2)")
                gr.Markdown("*Extract structured data from document text using AI. Define your desired JSON format and paste document text.*")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        json_format_input = gr.Textbox(
                            label="Desired JSON Format",
                            placeholder='''Example:
{
  "signer_name": "Full name of document signer",
  "issue_date": "Date in YYYY-MM-DD format", 
  "document_class": "Type of document",
  "key_points": ["List of main points"]
}''',
                            lines=8,
                            max_lines=15
                        )
                    
                    with gr.Column(scale=2):
                        document_text_input = gr.Textbox(
                            label="Document Text",
                            placeholder="Paste your document text here...",
                            lines=8,
                            max_lines=15
                        )
                
                extract_btn = gr.Button("🔍 Extract Information", variant="primary")
                
                extraction_output = gr.Textbox(
                    label="Extraction Results",
                    lines=12,
                    interactive=False
                )
                
                extract_btn.click(
                    fn=app.extract_information,
                    inputs=[json_format_input, document_text_input],
                    outputs=extraction_output
                )
            
            # Management Tab
            with gr.Tab("📊 System Management"):
                gr.Markdown("### System Status and Document Management")
                
                with gr.Row():
                    status_btn = gr.Button("🔧 Check System Status")
                    docs_btn = gr.Button("📚 List Documents")
                
                management_output = gr.Textbox(
                    label="System Information",
                    lines=15,
                    interactive=False
                )
                
                status_btn.click(
                    fn=app.get_system_status,
                    outputs=management_output
                )
                
                docs_btn.click(
                    fn=app.list_documents,
                    outputs=management_output
                )
    
    return interface

def main():
    """Main function to run the application."""
    try:
        # Create and launch the interface
        interface = create_interface()
        
        # Launch with share=True to handle localhost issues
        interface.launch(
            share=True,
            server_name="127.0.0.1",
            server_port=7860,
            show_error=True
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to start application: {e}")
        print(f"Error: {e}")
        print("\nPlease ensure:")
        print("1. Qdrant is running: docker run -p 6333:6333 qdrant/qdrant")
        print("2. Dependencies are installed: poetry install")
        print("3. OpenAI API key is set in .env file")
 

if __name__ == "__main__":
    main() 