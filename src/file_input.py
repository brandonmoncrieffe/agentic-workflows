import pymupdf4llm as pdf

def extract_markdown_from_pdf(pdf_path):
    md_text = pdf.to_markdown(pdf_path)
    return md_text
