import logging
from logging_config import setup_logging
import pymupdf4llm as pdf
import re

# Try to import pymupdf_layout for enhanced layout analysis
try:
    from pymupdf_layout import get_page_layout
    HAS_PYMUPDF_LAYOUT = True
except ImportError:
    HAS_PYMUPDF_LAYOUT = False
    logging.warning("pymupdf_layout not available, using pymupdf4llm for PDF extraction")


def remove_references_section(md_text):
    """
    Remove references section and everything after it.
    
    Args:
        md_text: Markdown text extracted from PDF
    
    Returns:
        Text with references section removed
    """
    # Common reference section headers (case insensitive)
    patterns = [
        r'\n#+\s*References?\s*\n',
        r'\n#+\s*Bibliography\s*\n',
        r'\n#+\s*Literature\s+Cited\s*\n',
        r'\n#+\s*Works\s+Cited\s*\n',
        r'\n\*\*References?\*\*\s*\n',
        r'\n\*\*Bibliography\*\*\s*\n',
    ]
    
    earliest_match = None
    earliest_pos = len(md_text)
    
    # Find the earliest reference section
    for pattern in patterns:
        match = re.search(pattern, md_text, re.IGNORECASE)
        if match and match.start() < earliest_pos:
            earliest_pos = match.start()
            earliest_match = match
    
    if earliest_match:
        logging.info(f"Found references section at position {earliest_pos}, removing rest of document")
        return md_text[:earliest_match.start()].strip()
    
    logging.info("No references section found, keeping full text")
    return md_text


def extract_markdown_from_pdf(pdf_path, remove_references=True, use_layout_analysis=True):
    """
    Extract markdown from PDF with optional reference removal and layout analysis.
    
    Uses pymupdf_layout for improved page layout analysis when available,
    falls back to pymupdf4llm for compatibility.
    
    Args:
        pdf_path: Path to PDF file
        remove_references: If True, remove references section (default: True)
        use_layout_analysis: If True, use pymupdf_layout for better layout analysis (default: True)
    
    Returns:
        Markdown text
    """
    # Use pymupdf_layout if available and requested
    if use_layout_analysis and HAS_PYMUPDF_LAYOUT:
        try:
            import pymupdf
            doc = pymupdf.open(pdf_path)
            md_text = ""
            
            for page_num, page in enumerate(doc):
                # Get layout information for better structure
                layout = get_page_layout(page)
                # Convert to markdown with layout context
                md_text += f"\n\n## Page {page_num + 1}\n"
                md_text += layout.get_text("markdown")
            
            doc.close()
            logging.info(f"Extracted markdown from PDF using pymupdf_layout: {pdf_path}")
        except Exception as e:
            logging.warning(f"pymupdf_layout extraction failed, falling back to pymupdf4llm: {e}")
            md_text = pdf.to_markdown(pdf_path)
    else:
        # Fallback to pymupdf4llm
        md_text = pdf.to_markdown(pdf_path)
    
    if remove_references:
        md_text = remove_references_section(md_text)
    
    logging.info(f"Extracted markdown from PDF: {pdf_path}")
    return md_text
