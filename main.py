import gradio as gr
import os
from datetime import datetime
from typing import List, Tuple, Any
import logging
import asyncio
from openai import AsyncOpenAI
# Import our modules
from config import Config, validate_config
from indexing import DocumentIndexer, verify_indexing_setup
from querying import QueryProcessor
from qdrant_setup import check_qdrant_connection
from utils import crop_all_pages, encode_image
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
                        f"📚 **Source Documents:\n")
            
            for doc in result['source_documents']:
                response += f"- {doc['name']} (Score: {doc['score']:.3f})\n"
            
            # if result['context_used']:
            #     response += f"\n📖 **Context Used:**\n{result['context_used']}"
            
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
    
    def extract_information(self, file) -> str:
        """Extract structured information from uploaded document using LLM."""
        if file is None:
            return "❌ Please upload a document file"
        
        if not self.query_processor:
            return "❌ System not ready. Please check configuration."
        
        from openai import AsyncOpenAI
        import asyncio
        from config import Config
        import tempfile
        import base64
        
        if not Config.OPENAI_API_KEY:
            return "❌ OpenAI API key not configured. Information extraction requires OpenAI."
        
              
        try:
            # Get file path
            file_path = file.name if hasattr(file, 'name') else str(file)      
            encoded_images = crop_all_pages(file_path)
            
            # Prepare messages with images for OpenAI Vision
            content = []
            
            # Add each page as an image
            for _, encoded_image in enumerate(encoded_images):
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encoded_image}"
                    }
                })
            
            # Use OpenAI Vision API for PDF images
            async def extract_from_images():
                client = AsyncOpenAI(api_key=Config.OPENAI_API_KEY)
                response = await client.chat.completions.create(
                    model="gpt-4o",  # Use gpt-4o for vision capabilities
                    response_format={"type": "json_object"},
                    messages=[
                        {
                            "role": "system", 
                            "content": """Dựa vào nội dung của file pdf được upload, trích xuất các thông tin quan trọng và trả về dưới dạng json:
                            - Thể loại (type): Thể loại của văn bản được upload. Nó có thể là Quyết định, Nghị định, Nghị quyết, Thông tư. Thông tin này được quyết định dựa vào phần đầu nội dung của văn bản sau khi kết thúc phần quốc hiệu tiêu ngữ. Thường được in đậm cùng với tên văn bản
                            - Tên văn bản (Document Name): Tên văn bản được upload, nằm ở phần đầu văn bản ngay sau phần Thể loại. Thường được in đậm cùng với "Thể loại". Phải lấy hết toàn bộ nôi dung nguyên văn của tên văn bản, không được tóm tắt
                            - Cơ quan ban hành (Source): Cơ quan hoặc tổ chức ban hành văn bản, nằm ở phần đầu văn bản
                            - Văn bản căn cứ (Related Documents): Danh sách các văn bản cản cứ mà văn bản hiện tại dựa vào để ban hành. Chúng được nằm ở phần đầu nội dung văn bản sau khi kết thúc phần tên văn bản. Đây là các đoạn văn được bắt đầu bằng motip "Căn cứ". Ví dụ 'Căn cứ Luật giao thông 2025' hoặc 'Căn cứ Thông tư 64 năm 2024"
                            - Ngày tháng năm phát hành (Date of Issue): Ngày tháng năm phát hành của văn bản, nằm ở phần đầu văn bản.
                            - Người kí (Signature Name): Nằm ở phần cuối của văn bản phần chữ kí
                            - Chức vụ người kí (Position): Nằm ở phần cuối văn bản cùng với phần chữ kí
                            Kết quả trả về dưới dạng json theo format như sau:
                            {
                            'result': {
                                "type": "Thể loại của văn bản",
                                "Document Name": "Tên văn bản được upload",
                                "Source": "Cơ quan hoặc tổ chức ban hành văn bản",
                                "Related Documents": List các văn bản cản cứ mà văn bản hiện tại dựa vào để ban hành",
                                "Date of Issue": "Ngày tháng năm phát hành của văn bản. Trả về dưới dạng dd/mm/yyyy. Nếu ngày tháng năm phần nào không có thì trả về --. Ví dụ 22/06/2025 hoặc --/06/2025 nếu không có phần ngày",
                                "Signature Name": "Người kí của văn bản",
                                "Position": "Chức vụ người kí của văn bản. None nếu không có"
                                }
                            }
                            Ví dụ:
                            {
                                "result": {
                                    "type": "Thông tư",
                                    "Document Name": "Hướng dẫn thực hiện bảo đảm cấp nước an toàn khu vực nông thôn",
                                    "Source": "Văn phòng chính phủ",
                                    "Related Documents": ["Nghị định số 105/2022/NĐ-CP", "Nghị định số 117/2007/NĐ-CP", " Nghị định số 124/2011/NĐ-CP"],
                                    "Date of Issue": "22/07/2025",
                                    "Signature Name": "Phạm Minh Chính",
                                    "Position": "Thủ tướng"
                                }
                            }
                            """
                        },
                        {
                            "role": "user",
                            "content": content
                        }
                    ]
                )
                return response.choices[0].message.content
            
            extracted_data = asyncio.run(extract_from_images())
            
            return (f"✅ **Information Extraction Completed**\n\n"
                    f"**Extracted Data:**\n{extracted_data}\n")
            
        except ImportError as e:
            return f"❌ {str(e)}"
        except Exception as e:
            return f"❌ Error processing PDF file: {str(e)}"
    
    def ocr_pdf_document(self, file):
        """OCR PDF document using GPT-4o vision model."""
        if file is None:
            return "❌ Please upload a PDF file", ""
        
        try:           
            if not Config.OPENAI_API_KEY:
                return "❌ OpenAI API key not configured", ""
            
            # Get file path
            file_path = file.name if hasattr(file, 'name') else str(file)
            file_extension = os.path.splitext(file_path)[1].lower()
            filename = os.path.basename(file_path)
            
            if file_extension != '.pdf':
                return "❌ Please upload a PDF file only", ""
            
            
            # Convert PDF to images
            encoded_images = crop_all_pages(file_path)
            
            # Prepare content for OCR
            content = [
            ]
            
            # Add each page as an image
            for encoded_image in encoded_images:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encoded_image}"
                    }
                })
            
            # Use OpenAI Vision API for OCR
            async def perform_ocr():
                client = AsyncOpenAI(api_key=Config.OPENAI_API_KEY)
                response = await client.chat.completions.create(
                    model="gpt-4.1",
                    messages=[
                        {
                            "role": "system", 
                            "content": "convert these images from a pdf file into markdown string. The output is only string with markdown format, nothing more. Remember to keep all contents of all images. You must format output as it shows in these images so the markdown is readable and beautiful."
                        },
                        {
                            "role": "user",
                            "content": content
                        }
                    ]
                )
                return response.choices[0].message.content
            
            # Get OCR text
            ocr_text = asyncio.run(perform_ocr())
            
            status_message = (f"✅ **PDF OCR Completed**\n\n"
                             f"📄 **File:** {filename}\n"
                             f"📊 **Pages Processed:** {len(encoded_images)}\n"
                             f"📝 **Text Length:** {len(ocr_text):,} characters\n\n"
                             f"You can now chat about this document!")
            
            return status_message, ocr_text
            
        except Exception as e:
            return f"❌ **Error during OCR:** {str(e)}", ""
    
    def chat_with_document(self, message, chat_history, ocr_text):
        """Chat about the uploaded document using OCR text as context."""
        if not message.strip():
            return ["Please upload a PDF document first.", ""], ""
        if not ocr_text:
            return ["Please upload a PDF document first.", ""], ""
        
        try:
            if not Config.OPENAI_API_KEY:
                return [message, "❌ OpenAI API key not configured"], ""
            
            # Prepare conversation context
            conversation_history = ""
            for user_msg, bot_msg in chat_history:
                if user_msg and bot_msg:
                    conversation_history += f"User: {user_msg}\nAssistant: {bot_msg}\n\n"
            
            # Create system prompt with document context
            system_prompt = (
                           f"Dựa vào nội dung của văn bản được cung cấp và lịch sử hội thoại, trả lời câu hỏi của người dùng:\n\n"
                           f"Đây là nội dung văn bnả:\n{ocr_text}\n\n"
                           f"Lịch sử hội thoại:\n{conversation_history}\n"
                           f"Trả lời câu hỏi bằng tiếng Việt. "
                           f"Nếu không có thông tin trong văn bản, hãy nói rằng không có thông tin trong văn bản và giải thích "
                           f"Trả lời chính xác và rõ ràng.")
            
            async def get_chat_response():
                client = AsyncOpenAI(api_key=Config.OPENAI_API_KEY)
                response = await client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": message}
                    ]
                )
                return response.choices[0].message.content
            
            # Get response
            bot_response = asyncio.run(get_chat_response())
            
            # Add to chat history
            chat_history.append([message, bot_response])
            
            return chat_history, ""
            
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            chat_history.append([message, error_msg])
            return chat_history, ""
    
    def reset_chat(self):
        """Reset chat history and file upload."""
        return [], None, "", ""


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
        - 🔍 Information extraction from documents
        - 💬 Chat with PDF documents using OCR
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
                gr.Markdown("*Upload a document and provide custom extraction instructions to extract specific information using AI.*")
                
                extraction_file_input = gr.File(
                    label="Select Document (PDF, DOCX, TXT)",
                    file_count="single"
                )
                
                extract_btn = gr.Button("🔍 Extract Information", variant="primary")
                
                extraction_output = gr.Textbox(
                    label="Extraction Results",
                    lines=12,
                    interactive=False
                )
                
                extract_btn.click(
                    fn=app.extract_information,
                    inputs=[extraction_file_input],
                    outputs=extraction_output
                )
            
            # Document Chat Tab
            with gr.Tab("💬 Chat with Document"):
                gr.Markdown("### Chat with Your PDF Document")
                gr.Markdown("*Upload a PDF document, get it OCR'd by GPT-4o, and chat about its content.*")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        chat_file_input = gr.File(
                            label="Upload PDF Document",
                            file_count="single",
                            file_types=[".pdf"]
                        )
                        
                        ocr_btn = gr.Button("📖 Processing Document...", variant="primary")
                        refresh_btn = gr.Button("🔄 Reset Chat", variant="secondary")
                        
                        ocr_status = gr.Textbox(
                            label="Processing Text...",
                            lines=8,
                            interactive=False
                        )
                    
                    with gr.Column(scale=2):
                        chatbot = gr.Chatbot(
                            label="Chat History",
                            height=500
                        )
                        
                        chat_input = gr.Textbox(
                            label="Your Message",
                            placeholder="Ask a question about the uploaded document...",
                            lines=2
                        )
                        
                        send_btn = gr.Button("📤 Send", variant="primary")
                
                # Hidden state to store OCR text
                ocr_text_state = gr.State("")
                
                # OCR button functionality
                ocr_btn.click(
                    fn=app.ocr_pdf_document,
                    inputs=[chat_file_input],
                    outputs=[ocr_status, ocr_text_state]
                )
                
                # Chat functionality
                def submit_message(message, history, ocr_text):
                    new_history, _ = app.chat_with_document(message, history, ocr_text)
                    return new_history, ""
                
                send_btn.click(
                    fn=submit_message,
                    inputs=[chat_input, chatbot, ocr_text_state],
                    outputs=[chatbot, chat_input]
                )
                
                chat_input.submit(
                    fn=submit_message,
                    inputs=[chat_input, chatbot, ocr_text_state],
                    outputs=[chatbot, chat_input]
                )
                
                # Reset functionality
                refresh_btn.click(
                    fn=app.reset_chat,
                    outputs=[chatbot, chat_file_input, ocr_status, ocr_text_state]
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