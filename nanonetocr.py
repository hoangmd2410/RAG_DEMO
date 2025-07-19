import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer, AutoProcessor
from PIL import Image
import fitz
import io
import re
from config import Config
import logging

logger = logging.getLogger(__name__)
# # Download required NLTK data
# try:
#     nltk.data.find('tokenizers/punkt')
# except LookupError:
#     nltk.download('punkt')
class NanoNetsOCRProcessor:
    """Handles document preprocessing using Nanonets-OCR-s model."""
    
    def __init__(self):
        self.model_name = Config.NANONETS_OCR_MODEL
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model = None
        self.processor = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the Nanonets OCR model and processor."""
        try:
            logger.info(f"Loading Nanonets OCR model: {self.model_name}")
            
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True,
            )
            self.model.eval()
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.processor = AutoProcessor.from_pretrained(self.model_name, use_fast=True)
            
            logger.info(f"✅ Nanonets OCR model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"❌ Error loading Nanonets OCR model: {e}")
            logger.warning("⚠️ Falling back to traditional text extraction methods")
            self.model = None
            self.processor = None
            self.tokenizer = None
    
    def convert_to_markdown(self, image_path: str) -> str:
        """
        Convert document image to markdown using Nanonets-OCR-s.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Markdown text extracted from the image
        """
        if not self.model or not self.processor:
            logger.warning("⚠️ Nanonets OCR model not available, using fallback")
            return ""
        
        try:
            # Prepare the prompt for optimal OCR performance
            prompt = """Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes."""
            
            # Load and process the image
            image = Image.open(image_path)
            
            # Prepare messages for the model
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": prompt},
                ]},
            ]
            
            # Apply chat template
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Process inputs
            inputs = self.processor(
                text=[text], 
                images=[image], 
                padding=True, 
                return_tensors="pt"
            )
            inputs = inputs.to(self.model.device)
            
            # Generate output
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=Config.NANONETS_MAX_NEW_TOKENS, 
                    do_sample=False
                )
            
            # Decode the generated text
            generated_ids = [
                output_ids[len(input_ids):] 
                for input_ids, output_ids in zip(inputs.input_ids, output_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=True
            )
            
            markdown_text = output_text[0] if output_text else ""
            
            logger.info(f"✅ Successfully converted image to markdown: {len(markdown_text)} characters")
            return markdown_text
            
        except Exception as e:
            logger.error(f"❌ Error converting image to markdown: {e}")
            return ""
    
    
    def extract_pages_content(self, pdf_path: str):
        """
        Extract text and images from each page of a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            max_pages: Maximum number of pages to process (None for all pages)
            save_images: Whether to save images to files (default: False, keeps in memory)
        
        Returns:
            List of lists, where each inner list contains page elements (text blocks and images)
            Each element is a dict with keys: 'type', 'content', 'bbox', 'page_num'
        """
        # try:
            # Open PDF
        pdf_document = fitz.open(pdf_path)
        
        logger.info(f"📄 Processing PDF: {pdf_path}")        
        all_pages_content = []
        
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            page_content = []
            
            logger.info(f"📄 Processing Page {page_num + 1}...")
            
            # Extract text blocks
            text_dict = page.get_text("dict")
            text_elements = []
            
            for block in text_dict["blocks"]:
                if "lines" in block:  # Text block
                    block_text = ""
                    for line in block["lines"]:
                        for span in line["spans"]:
                            block_text += span["text"]
                        block_text += "\n"
                    
                    if block_text.strip():  # Only add non-empty text
                        text_elements.append({
                            'type': 'text',
                            'content': block_text.strip(),
                            'bbox': block["bbox"],  # (x0, y0, x1, y1)
                            'page_num': page_num + 1
                        })
            
            # Extract images
            image_list = page.get_images()
            image_elements = []
            
            for _, img in enumerate(image_list):
                # try:
                    # Get image data
                xref = img[0]
                pix = fitz.Pixmap(pdf_document, xref)
                
                # Skip very small images (likely decorative)
                if pix.width < 50 or pix.height < 50:
                    pix = None
                    continue
                
                # Get image rectangle (position on page)
                image_rects = page.get_image_rects(xref)
                bbox = image_rects[0] if image_rects else (0, 0, pix.width, pix.height)
                

                # Convert to RGB if needed
                if pix.n - pix.alpha >= 4:  # CMYK
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                
                # Crop image to the bbox
                x0, y0, x1, y1 = bbox
                crop_rect = fitz.Rect(x0, y0, x1, y1)
                cropped_pix = page.get_pixmap(clip=crop_rect)
                
                # Convert to PIL Image
                img_bytes = cropped_pix.tobytes("png")

                # Load into PIL directly from bytes
                pilimage = Image.open(io.BytesIO(img_bytes))
                image_elements.append({
                    'type': 'image',
                    'content': pilimage,
                    'bbox': bbox,
                    'page_num': page_num + 1,
                    'size': (pix.width, pix.height)
                })
                pix = None

            all_elements = text_elements + image_elements
            all_elements.sort(key=lambda x: (x['bbox'][1], x['bbox'][0]))  # Sort by y, then x
            
            page_content = all_elements
            all_pages_content.append(page_content)
            
            logger.info(f"  ✅ Found {len(text_elements)} text blocks and {len(image_elements)} images")
        
        pdf_document.close()        
        return all_pages_content


    def process_html_file(self, html_path: str) -> str:
        """
        Process HTML file by converting to image first, then to markdown.
        
        Args:
            html_path: Path to the HTML file
            
        Returns:
            Markdown text extracted from the HTML
        """
        try:
            # For HTML files, we could use libraries like wkhtmltopdf or selenium
            # For now, we'll read the HTML content and extract text
            with open(html_path, 'r', encoding='utf-8') as file:
                html_content = file.read()
            
            # Use BeautifulSoup to extract text from HTML
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html_content, 'html.parser')
                text_content = soup.get_text()
                
                # Clean up the text
                text_content = re.sub(r'\s+', ' ', text_content).strip()
                
                logger.info(f"✅ Extracted text from HTML: {len(text_content)} characters")
                return text_content
                
            except ImportError:
                logger.warning("⚠️ BeautifulSoup not available, using basic HTML processing")
                # Basic HTML tag removal
                text_content = re.sub(r'<[^>]+>', '', html_content)
                text_content = re.sub(r'\s+', ' ', text_content).strip()
                return text_content
                
        except Exception as e:
            logger.error(f"❌ Error processing HTML file: {e}")
            return ""
