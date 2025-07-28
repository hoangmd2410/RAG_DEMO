#!/usr/bin/env python3
"""
Test script for Nanonets OCR preprocessing functionality.
This script demonstrates how the new preprocessing pipeline works with various document formats.
"""

import os
import logging
from utils import DocumentProcessor, NanoNetsOCRProcessor
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_nanonets_ocr():
    """Test the Nanonets OCR preprocessing functionality."""
    
    logger.info("🚀 Testing Nanonets OCR preprocessing pipeline")
    
    # Initialize the document processor
    processor = DocumentProcessor()
    
    # Test cases for different file formats
    test_files = [
        # Add your test files here
        "test.png",
        # "sample_documents/test.png", 
        # "sample_documents/test.html",
        # "sample_documents/test.jpg"
    ]
    
    if not test_files:
        logger.warning("⚠️ No test files specified. Please add test files to the test_files list.")
        logger.info("📝 Supported formats: " + ", ".join(Config.SUPPORTED_FORMATS))
        return
    
    for file_path in test_files:
        if not os.path.exists(file_path):
            logger.warning(f"⚠️ Test file not found: {file_path}")
            continue
            
        logger.info(f"\n🔍 Processing: {file_path}")
        logger.info(f"📂 File format: {os.path.splitext(file_path)[1]}")
        
        try:
            # Extract text using the new preprocessing pipeline
            extracted_text = processor.extract_text(file_path)
            
            if extracted_text:
                logger.info(f"✅ Successfully extracted text: {len(extracted_text)} characters")
                
                # Show a preview of the extracted text
                preview = extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text
                logger.info(f"📄 Preview:\n{preview}")
                
                # Save extracted text to file for review
                output_file = f"output_{os.path.basename(file_path)}.txt"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(extracted_text)
                logger.info(f"💾 Saved extracted text to: {output_file}")
                
            else:
                logger.error(f"❌ Failed to extract text from: {file_path}")
                
        except Exception as e:
            logger.error(f"❌ Error processing {file_path}: {e}")

def test_nanonets_processor_directly():
    """Test the Nanonets processor directly with an image."""
    
    logger.info("\n🔬 Testing Nanonets processor directly")
    
    # Initialize the Nanonets processor
    nanonets_processor = NanoNetsOCRProcessor()
    
    if not nanonets_processor.model:
        logger.error("❌ Nanonets OCR model not loaded. Please check your GPU and dependencies.")
        return
    
    # Test with a sample image (you need to provide this)
    test_image = "sample_image.png"  # Replace with your test image path
    
    if not os.path.exists(test_image):
        logger.warning(f"⚠️ Test image not found: {test_image}")
        logger.info("📝 Please provide a test image file to test the Nanonets OCR functionality.")
        return
    
    logger.info(f"🖼️ Processing image: {test_image}")
    
    try:
        # Convert image to markdown
        markdown_output = nanonets_processor.convert_to_markdown(test_image)
        
        if markdown_output:
            logger.info(f"✅ Successfully converted to markdown: {len(markdown_output)} characters")
            
            # Save markdown output
            output_file = f"nanonets_output_{os.path.basename(test_image)}.md"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(markdown_output)
            logger.info(f"💾 Saved markdown output to: {output_file}")
            
            # Show preview
            preview = markdown_output[:500] + "..." if len(markdown_output) > 500 else markdown_output
            logger.info(f"📄 Markdown preview:\n{preview}")
            
        else:
            logger.error("❌ Failed to convert image to markdown")
            
    except Exception as e:
        logger.error(f"❌ Error processing image: {e}")

def check_system_requirements():
    """Check if the system meets the requirements for Nanonets OCR."""
    
    logger.info("\n🔧 Checking system requirements")
    
    # Check GPU availability
    import torch
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"✅ GPU available: {gpu_count} GPU(s)")
        logger.info(f"💾 GPU memory: {gpu_memory:.1f} GB")
        
        if gpu_memory < 16:
            logger.warning("⚠️ WARNING: Nanonets OCR requires at least 16GB GPU memory for optimal performance")
        
    else:
        logger.warning("⚠️ No GPU available. Nanonets OCR will use CPU (much slower)")
    
    # Check dependencies
    required_packages = ['transformers', 'torch', 'PIL', 'pdf2image']
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package} is installed")
        except ImportError:
            logger.error(f"❌ {package} is missing. Please install it.")

if __name__ == "__main__":
    logger.info("🔬 Nanonets OCR Test Suite")
    logger.info("=" * 50)
    
    # Check system requirements first
    check_system_requirements()
    
    # Test the document processor
    test_nanonets_ocr()
    
    # Test the Nanonets processor directly
    test_nanonets_processor_directly()
    
    logger.info("\n🏁 Test completed!")
    logger.info("💡 Tips:")
    logger.info("   - Ensure you have sufficient GPU memory (16GB+ recommended)")
    logger.info("   - Place test files in the same directory as this script")
    logger.info("   - Check the output files for extracted text and markdown") 