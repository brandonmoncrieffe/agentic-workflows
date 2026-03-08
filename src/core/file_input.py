import logging
from logging_config import setup_logging
import pymupdf4llm as pdf
import re


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


def extract_markdown_from_pdf(pdf_path, remove_references=True):
    """
    Extract markdown from PDF with optional reference removal.
    
    Args:
        pdf_path: Path to PDF file
        remove_references: If True, remove references section (default: True)
    
    Returns:
        Markdown text
    """
    md_text = pdf.to_markdown(pdf_path)
    
    if remove_references:
        md_text = remove_references_section(md_text)
    
    logging.info(f"Extracted markdown from PDF: {pdf_path}")
    return md_text
