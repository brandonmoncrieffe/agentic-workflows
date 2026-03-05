import logging
from logging_config import setup_logging
import pymupdf4llm as pdf

def extract_markdown_from_pdf(pdf_path):
    md_text = pdf.to_markdown(pdf_path)
    logging.info(f"Extracted markdown from PDF: {pdf_path}")
    return md_text
