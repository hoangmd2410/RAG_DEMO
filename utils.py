import base64
import os
import tempfile
from PIL import Image
import fitz
import os

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Function to convert PDF pages to images
def crop_all_pages(pdf_path):
    try:
        import fitz  # PyMuPDF
        from PIL import Image
    except ImportError:
        raise ImportError("PyMuPDF (fitz) and PIL are required for PDF processing. Please install: pip install PyMuPDF Pillow")
    
    doc = fitz.open(pdf_path)
    encoded_images = []
    
    for page_num, page in enumerate(doc):
        pix = page.get_pixmap()
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        pix.save(temp_file.name)
        image = Image.open(temp_file.name)
        image.save(temp_file.name, "PNG")
        
        encoded_image = encode_image(temp_file.name)
        encoded_images.append(encoded_image)
        
        # Clean up temp file
        os.unlink(temp_file.name)
    
    doc.close()
    return encoded_images
