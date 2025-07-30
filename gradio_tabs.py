import gradio as gr
from config import Config, validate_config
from rag.indexing import DocumentIndexer, verify_indexing_setup
from rag.querying import QueryProcessor
from rag.qdrant_setup import check_qdrant_connection
from rag.utils import crop_all_pages, encode_image
from rag.nanonet_ocr import ocr_list_pages
import asyncio
import os
import logging
from openai import AsyncOpenAI
import json


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


class SemanticSearchTab:
    """Question & Answer Tab"""
    
    def __init__(self, query_processor: QueryProcessor):
        self.query_processor = query_processor
    
    def search_with_individual_downloads(self, query: str):
        """Perform search and prepare individual download files."""
        if not query.strip():
            return "Please enter a search query", None, None, None, None, None
        
        try:
            # Get search results (list of document names)
            document_names = self.query_processor.get_similar_documents(query)
            
            if not document_names:
                return "No relevant documents found for your query.", None, None, None, None, None
            
            # Format results for display
            results_text = f"🔍 **Search Results for:** '{query}'\n\n"
            results_text += f"Found {len(document_names)} relevant documents:\n\n"
            
            # Prepare individual download files (max 5 for UI)
            download_files = [None, None, None, None, None]
            available_count = 0
            
            for i, doc_name in enumerate(document_names[:5]):  # Limit to first 5 results
                results_text += f"**{i+1}. {doc_name}**\n"
                
                # Look for file in ./data/ directory
                data_path = os.path.join("./data", doc_name)
                if os.path.exists(data_path):
                    download_files[i] = data_path
                    available_count += 1
                    results_text += f"   📥 Available for download (File {i+1})\n\n"
                else:
                    results_text += f"   ❌ File not found in data directory\n\n"
            
            if available_count > 0:
                results_text += f"📦 **Total downloadable files:** {available_count}"
            else:
                results_text += "\n⚠️ No files available for download."
            
            return results_text, download_files[0], download_files[1], download_files[2], download_files[3], download_files[4]
                
        except Exception as e:
            return f"❌ Search error: {str(e)}", None, None, None, None, None
    
    def get_frontend(self):
        """Returns the Q&A tab interface"""
        with gr.Tab("Semantic Search"):
            gr.Markdown("### Semantic Search")
            
            question_input = gr.Textbox(
                label="Your Search Query",
                placeholder="Enter your search query to find relevant documents...",
                lines=3
            )
            
            search_btn = gr.Button("💬 Search", variant="primary")
            
            search_output = gr.Textbox(
                label="Search Results",
                lines=12,
                interactive=False
            )
            
            # Individual download sections
            gr.Markdown("### 📥 Individual Downloads")
            
            with gr.Row():
                with gr.Column():
                    download_file_1 = gr.File(label="📄 File 1", visible=True)
                    download_file_2 = gr.File(label="📄 File 2", visible=True)
                
                with gr.Column():
                    download_file_3 = gr.File(label="📄 File 3", visible=True)
                    download_file_4 = gr.File(label="📄 File 4", visible=True)
                
                with gr.Column():
                    download_file_5 = gr.File(label="📄 File 5", visible=True)
            
            # Search functionality with individual downloads
            search_btn.click(
                fn=self.search_with_individual_downloads,
                inputs=[question_input],
                outputs=[
                    search_output, 
                    download_file_1, 
                    download_file_2, 
                    download_file_3, 
                    download_file_4, 
                    download_file_5
                ]
            )


class ExtractInformationTab:
    """Information Extraction Tab"""
    
    def __init__(self,name="Extract Information"):
        self.name = name
    
    def extract_information(self, file)-> str:
        """Extract structured information from uploaded document using LLM."""
        if file is None:
            return "❌ Please upload a document file"
        
        if not Config.OPENAI_API_KEY:
            return "❌ OpenAI API key not configured. Information extraction requires OpenAI."
        
        # try:
        # Get file path
        file_path = file.name if hasattr(file, 'name') else str(file)
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension != '.pdf':
            return "❌ Please upload a PDF file only"


        # Convert PDF to images (base64 encoded)
        encoded_images = crop_all_pages(file_path)
        
        # Use OCR to extract text from images
        async def extract_text_with_ocr():
            ocr_text = await ocr_list_pages(encoded_images)
            return ocr_text
        
        # Get OCR text
        document_text = asyncio.run(extract_text_with_ocr())
        
        if not document_text:
            return "❌ Failed to extract text from document"
        
        # Use OpenAI to extract structured information from the text
        async def extract_information_from_text():
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
                        - Văn bản căn cứ (Related Documents): Danh sách các văn bản cản cứ mà văn bản hiện tại dựa vào để ban hành. Chúng được nằm ở phần đầu nội dung văn bản sau khi kết thúc phần tên văn bản và trước khi bắt đầu nội dung các quy định văn bản . Ví dụ 'Căn cứ Luật giao thông 2025' hoặc 'Căn cứ Thông tư 64 năm 2024" hoặc "Thực hiện nghị định 105/2022/NĐ-CP"
                        - Văn bản đề nghị (Proposed Documents): là 1 văn bản duy nhất đề nghị của văn bản hiện tại. Nó được bắt đầu bằng motip "Theo đề nghị" hoặc "Theo đề xuất". Văn bản đề nghị có thể không có nếu văn bản hiện tại được soạn thảo theo biểu quyết của cử tri
                        - Văn bản sửa đổi/ bổ sung (Amendment Documents): Danh sách các văn bản mà văn bản hiện tại sửa đổi/ bổ sung hoặc bác bỏ. Các văn bản này có thể tìm thấy ở phần Tên văn bản. Có thể rỗng
                        - Văn bản bãi bỏ (Removed Documents): Danh sách các văn bản mà văn bản hiện tại bãi bỏ. Các văn bản này có thể tìm thấy ở phần Tên văn bản. Có thể rỗng
                        - Ngày tháng năm phát hành (Date of Issue): Ngày tháng năm phát hành của văn bản, nằm ở phần đầu văn bản.
                        - Người kí (Signature Name): Nằm ở phần cuối của văn bản phần chữ kí
                        - Chức vụ người kí (Position): Nằm ở phần cuối văn bản cùng với phần chữ kí
                        Kết quả trả về dưới dạng json theo format như sau:
                        {
                        'result': {
                            "type": "Thể loại của văn bản",
                            "Document Name": "Tên văn bản được upload",
                            "Number": "Số kí hiệu của văn bản. Phải đầy đủ số, không được bỏ số 0 ở đầu. Ví dụ: 33/2025/TT-BGDĐT",
                            "Source": "Cơ quan hoặc tổ chức ban hành văn bản",
                            "Related Documents": "List các văn bản cản cứ mà văn bản hiện tại dựa vào để ban hành",
                            "Proposed Documents": "Văn bản đề nghị của văn bản hiện tại. None nếu không có",
                            "Amendment Documents": "List các văn bản mà văn bản hiện tại sửa đổi/ bổ sung. List rỗng [] nếu không có",
                            "Removed Documents": "List các văn bản mà văn bản hiện tại bãi bỏ. List rỗng [] nếu không có",
                            "Date of Issue": "Ngày tháng năm phát hành của văn bản. Trả về dưới dạng dd/mm/yyyy. Nếu ngày tháng năm phần nào không có thì trả về --. Ví dụ 22/06/2025 hoặc --/06/2025 nếu không có phần ngày",
                            "Date of Effective": "Ngày hiệu lực của văn bản. Trả về dưới dạng dd/mm/yyyy. Nếu ngày hiệu lực phần nào không có thì trả về --. Ví dụ 22/06/2025 hoặc --/06/2025 nếu không có phần ngày",
                            "Signature Name": "Người kí của văn bản",
                            "Position": "Chức vụ người kí của văn bản. None nếu không có"
                            }
                        }
                        Ví dụ:
                        {
                            "result": {
                                "type": "Thông tư",
                                "Document Name": "Hướng dẫn thực hiện bảo đảm cấp nước an toàn khu vực nông thôn",
                                "Number": "33/2025/TT-TTCP",
                                "Source": "Văn phòng chính phủ",
                                "Related Documents": ["Nghị định số 105/2022/NĐ-CP", "Nghị định số 117/2007/NĐ-CP", " Nghị định số 124/2011/NĐ-CP"],
                                "Proposed Documents": "281/STC-TCĐT",
                                "Amendment Documents": [],
                                "Removed Documents": [],
                                "Date of Issue": "22/07/2025",
                                "Date of Effective": "25/07/2025",
                                "Signature Name": "Phạm Minh Chính",
                                "Position": "Thủ tướng"
                            }
                        }
                        """
                    },
                    {
                        "role": "user",
                        "content": document_text
                    }
                ]
            )
            json_output = json.loads(response.choices[0].message.content)['result']

            return json_output
        
        extracted_data = asyncio.run(extract_information_from_text())
        
        return (
            f"✅ **Hoàn thành trích xuất**\n\n"
            f"📄 **Độ dài văn bản:** {len(encoded_images)} trang\n"
            f"📝 **Độ dài văn bản sau khi OCR:** {len(document_text):,} ký tự\n\n"
            f"**Dữ liệu trích xuất:**\n"
            f"- Loại văn bản: {extracted_data['type']}\n"
            f"- Tên văn bản: {extracted_data['Document Name']}\n"
            f"- Số kí hiệu: {extracted_data['Number']}\n"
            f"- Cơ quan ban hành: {extracted_data['Source']}\n"
            f"- Văn bản căn cứ: {extracted_data['Related Documents']} \n"
            f"- Văn bản đề nghị: {extracted_data['Proposed Documents']}\n"
            f"- Văn bản sửa đổi/ bổ sung:\n"
            + (
                "".join(f"\t* {i}\n" for i in extracted_data['Amendment Documents'])
                if extracted_data['Amendment Documents'] else ""
            )
            + f"- Văn bản bãi bỏ:\n"
            + (
                "".join(f"\t* {i}\n" for i in extracted_data['Removed Documents'])
                if extracted_data['Removed Documents'] else ""
            )
            + f"- Ngày tháng năm phát hành: {extracted_data['Date of Issue']}\n"
            + f"- Ngày hiệu lực: {extracted_data['Date of Effective']}\n"
            + f"- Người kí: {extracted_data['Signature Name']}\n"
            + f"- Chức vụ người kí: {extracted_data['Position']}\n"
        )

            
        # except ImportError as e:
        #     return f"❌ {str(e)}"
        # except Exception as e:
        #     return f"❌ Error processing PDF file: {str(e)}"
    
    def get_frontend(self):
        """Returns the information extraction tab interface"""
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
                fn=self.extract_information,
                inputs=[extraction_file_input],
                outputs=extraction_output
            )


class ChatWithDocumentTab:
    """Document Chat Tab"""
    
    def __init__(self, name="Chat with Document"):
        self.name = name
    
    def ocr_multiple_documents(self, files):
        """OCR multiple PDF documents using GPT-4o vision model."""
        if not files or len(files) == 0:
            return "❌ Please upload at least one PDF file", {}
        
        try:           
            if not Config.OPENAI_API_KEY:
                return "❌ OpenAI API key not configured", {}
            
            processed_documents = {}
            status_messages = []
            total_pages = 0
            
            # Prepare all tasks for parallel processing
            tasks_data = []
            
            for file in files:
                # Get file path
                file_path = file.name if hasattr(file, 'name') else str(file)
                file_extension = os.path.splitext(file_path)[1].lower()
                filename = os.path.basename(file_path)
                
                if file_extension != '.pdf':
                    status_messages.append(f"⚠️ Skipped {filename} (not a PDF file)")
                    continue
                
                # Convert PDF to images
                encoded_images = crop_all_pages(file_path)
                total_pages += len(encoded_images)

                tasks_data.append({
                    'filename': filename,
                    'pages': encoded_images,
                    'pages_count': len(encoded_images)
                })
            
            # Run OCR tasks in parallel
            async def process_all_files():
                tasks = []
                for task_data in tasks_data:
                    # ocr_list_pages expects list of base64 images and returns text
                    task = ocr_list_pages(task_data['pages'])
                    tasks.append(task)
                
                if tasks:
                    return await asyncio.gather(*tasks, return_exceptions=True)
                return []
            
            # Execute all OCR tasks
            if tasks_data:
                results = asyncio.run(process_all_files())
                
                # Process results
                for i, result in enumerate(results):
                    filename = tasks_data[i]['filename']
                    pages_count = tasks_data[i]['pages_count']
                    
                    if isinstance(result, Exception):
                        status_messages.append(f"❌ Failed to process {filename}: {str(result)}")
                    else:
                        # result is just the OCR text string
                        ocr_text = result
                        
                        processed_documents[filename] = {
                            'text': ocr_text,
                            'pages': pages_count,
                            'characters': len(ocr_text)
                        }
                        status_messages.append(f"✅ {filename}: {pages_count} pages, {len(ocr_text):,} characters")
            
            if not processed_documents:
                return "❌ No documents were successfully processed", {}
            
            # Create combined status message
            status_message = (f"✅ **Multiple PDF OCR Completed**\n\n"
                             f"📄 **Documents Processed:** {len(processed_documents)}\n"
                             f"📊 **Total Pages:** {total_pages}\n\n"
                             f"**Processing Results:**\n")
            
            for msg in status_messages:
                status_message += f"- {msg}\n"
            
            status_message += f"\nYou can now chat about these documents!"
            
            return status_message, processed_documents
            
        except Exception as e:
            return f"❌ **Error during OCR:** {str(e)}", {}
    
    def chat_with_documents(self, message, chat_history, documents_data):
        """Chat about the uploaded documents using OCR text as context."""
        if not message.strip():
            return chat_history + [[message, "Please enter a message."]], ""
        
        if not documents_data:
            return chat_history + [[message, "Please upload PDF documents first."]], ""
        
        try:
            if not Config.OPENAI_API_KEY:
                return chat_history + [[message, "❌ OpenAI API key not configured"]], ""
            
            # Prepare conversation context
            conversation_history = ""
            for user_msg, bot_msg in chat_history:
                if user_msg and bot_msg:
                    conversation_history += f"User: {user_msg}\nAssistant: {bot_msg}\n\n"
            
            # Combine all document texts
            all_documents_text = ""
            document_list = ""
            for filename, doc_data in documents_data.items():
                document_list += f"- {filename} ({doc_data['pages']} pages)\n"
                all_documents_text += f"\n--- Document: {filename} ---\n"
                all_documents_text += doc_data['text'] + "\n"
            
            # Create system prompt with document context
            system_prompt = (
                           f"Bạn là một trợ lý AI thông minh. Dựa vào nội dung của các văn bản được cung cấp và lịch sử hội thoại, hãy trả lời câu hỏi của người dùng.\n\n"
                           f"Danh sách các văn bản đã được tải lên:\n{document_list}\n"
                           f"Nội dung các văn bản:\n{all_documents_text}\n\n"
                           f"Lịch sử hội thoại:\n{conversation_history}\n"
                           f"Hướng dẫn:\n"
                           f"- Trả lời câu hỏi bằng tiếng Việt\n"
                           f"- Khi trích dẫn thông tin, hãy ghi rõ tên văn bản nguồn\n"
                           f"- Nếu thông tin không có trong các văn bản, hãy nói rõ điều đó\n"
                           f"- Trả lời chính xác, rõ ràng và chi tiết\n"
                           f"- Có thể so sánh thông tin giữa các văn bản nếu cần thiết")
            
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
            new_history = chat_history + [[message, bot_response]]
            
            return new_history, ""
            
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            new_history = chat_history + [[message, error_msg]]
            return new_history, ""
    
    def reset_chat(self):
        """Reset chat history and file upload."""
        return [], None, "", {}

    def get_frontend(self):
        """Returns the document chat tab interface"""
        with gr.Tab("💬 Chat with Documents"):
            gr.Markdown("### Chat with Your PDF Documents")
            gr.Markdown("*Upload multiple PDF documents, and chat about their content.*")
            
            with gr.Row():
                with gr.Column(scale=1):
                    chat_file_input = gr.File(
                        label="Upload PDF Documents",
                        file_count="multiple",
                        file_types=[".pdf"]
                    )
                    
                    ocr_btn = gr.Button("📖 Process Documents", variant="primary")
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
                        placeholder="Ask questions about the uploaded documents...",
                        lines=2
                    )
                    
                    send_btn = gr.Button("📤 Send", variant="primary")
            
            # Hidden state to store OCR data for all documents
            documents_data_state = gr.State({})
            
            # OCR button functionality
            ocr_btn.click(
                fn=self.ocr_multiple_documents,
                inputs=[chat_file_input],
                outputs=[ocr_status, documents_data_state]
            )
            
            # Chat functionality
            def submit_message(message, history, documents_data):
                new_history, _ = self.chat_with_documents(message, history, documents_data)
                return new_history, ""
            
            send_btn.click(
                fn=submit_message,
                inputs=[chat_input, chatbot, documents_data_state],
                outputs=[chatbot, chat_input]
            )
            
            chat_input.submit(
                fn=submit_message,
                inputs=[chat_input, chatbot, documents_data_state],
                outputs=[chatbot, chat_input]
            )
            
            # Reset functionality
            refresh_btn.click(
                fn=self.reset_chat,
                outputs=[chatbot, chat_file_input, ocr_status, documents_data_state]
            )


if __name__ == "__main__":
    try:
        # search_tab = SemanticSearchTab(QueryProcessor())
        extract_tab = ExtractInformationTab()
        # chat_tab = ChatWithDocumentTab()
        with gr.Blocks(title="AI Document Understanding") as interface:
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
                # search_tab.get_frontend()
                extract_tab.get_frontend()
                # chat_tab.get_frontend()
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