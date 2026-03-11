import logging
import pymupdf4llm as pdf
import re
from typing import Optional

def clean_markdown_text(md_text: str) -> str:
    """
    Clean markdown text by removing common PDF artifacts.
    Aggressively removes tables, equations, and complex notation.
    
    Args:
        md_text: Raw markdown text extracted from PDF
    
    Returns:
        Cleaned markdown text
    """
    # Remove "Machine Translated by Google" headers
    md_text = re.sub(r'Machine Translated by Google\s*\n+', '', md_text, flags=re.IGNORECASE)
    md_text = re.sub(r'Translated by Google\s*\n+', '', md_text, flags=re.IGNORECASE)
    
    # Remove OCR artifact characters
    md_text = md_text.replace('ÿ', '')
    md_text = md_text.replace('\uffff', '')
    md_text = md_text.replace('\ufffe', '')
    md_text = md_text.replace('�', '')  # Unicode replacement character
    
    lines = md_text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Skip empty lines (but keep them)
        if not line.strip():
            cleaned_lines.append(line)
            continue
            
        # Skip lines with Unicode replacement characters
        if '�' in line or '���' in line:
            continue
            
        # Skip tables (lines with multiple pipes)
        if line.count('|') >= 2:
            continue
            
        # Skip table separator lines
        if re.match(r'^\s*[\|\-\s:]+\s*$', line):
            continue
            
        # Skip lines with excessive underscores (mathematical notation)
        if line.count('_') > 5:
            continue
            
        # Skip lines with complex superscript references [1], [2]–[5], etc.
        if re.search(r'\[\d+\][\[\–\-\]]*\[\d+\]', line):
            continue
            
        # Skip lines with broken LaTeX/equation brackets like [+][jk][(][L]
        if line.count('][') >= 3:
            continue
            
        # Skip lines with single-character brackets like [+] [−] [jk]
        if re.search(r'\[[^\]]{1,3}\].*\[[^\]]{1,3}\].*\[[^\]]{1,3}\]', line):
            continue
            
        # Skip lines that look like equation numbers: (1), (2), (11), etc. standing alone or with minimal text
        if re.match(r'^\s*\(\d+\)\s*$', line):
            continue
            
        # Skip lines starting with equation-like patterns
        if re.match(r'^\s*[A-Z]\s*=\s*j\s*\d+\s*sin', line):
            continue
            
        # Skip lines that are just a single number (chart axes, figure labels)
        if re.match(r'^\s*\d+\s*$', line):
            continue
            
        # Skip lines with sequential numbers (chart axes like "100 90 80 70")
        # Check if line is mostly just numbers separated by whitespace
        stripped = line.strip()
        if stripped and re.match(r'^[\d\s]+$', stripped):
            # Line contains only digits and spaces
            nums = stripped.split()
            if len(nums) >= 3:  # At least 3 numbers in sequence
                continue
            
        # Skip lines that are mostly mathematical symbols
        math_chars = sum(1 for c in line if c in '_^*√∫∑∏∂∇≈≠≤≥±×÷∈∉⊂⊃∪∩')
        if len(line.strip()) > 0 and math_chars / len(line.strip()) > 0.3:
            continue
            
        # Skip lines with complex equation-like patterns
        if re.search(r'[_\^]{2,}|_\{.*?\}|\^\{.*?\}', line):
            continue
            
        # Skip lines with excessive mathematical formatting like log10����
        if re.search(r'log\d+[^\w\s]{3,}', line):
            continue
            
        # Skip lines that are mostly just symbols and minimal text
        text_chars = sum(1 for c in line if c.isalpha())
        if len(line.strip()) > 5 and text_chars < len(line.strip()) * 0.4:
            continue
            
        cleaned_lines.append(line)
    
    md_text = '\n'.join(cleaned_lines)
    
    # Remove excessive blank lines (more than 2 in a row)
    md_text = re.sub(r'\n{3,}', '\n\n', md_text)
    
    md_text = md_text.strip()
    
    logging.info("Applied aggressive markdown cleaning")
    return md_text


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
    r'\nReferences\s*\n',
    r'\nBibliography\s*\n',
    r'\nREFERENCES\s*\n',
    r'\nReferences\b',
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
        print(f"Found references section at position {earliest_pos}, removing rest of document")
        return md_text[:earliest_match.start()].strip()
    
    logging.info("No references section found, keeping full text")
    return md_text


def extract_markdown_from_pdf(pdf_path: str, remove_references: bool = True, 
                             clean_text: bool = False, **kwargs) -> str:
    """
    Extract markdown from PDF with cleaning and optional reference removal.
    
    Args:
        pdf_path: Path to PDF file
        remove_references: If True, remove references section (default: True)
        clean_text: If True, apply text cleaning (default: True)
        **kwargs: Additional arguments to pass to pymupdf4llm.to_markdown()
                 Common options:
                 - page_chunks: bool - Return page-by-page chunks (default: False)
                 - margins: tuple - Page margins (left, top, right, bottom)
                 - dpi: int - Image resolution (default: 150)
    
    Returns:
        Cleaned markdown text
    """
    # Extract markdown with better default settings
    default_kwargs = {
        'page_chunks': False,  # Get single document instead of page chunks
    }
    default_kwargs.update(kwargs)
    
    md_text = pdf.to_markdown(pdf_path, **default_kwargs)
    
    # Apply text cleaning first (before reference removal)
    if clean_text:
        md_text = clean_markdown_text(md_text)
    
    # Remove references section
    if remove_references:
        md_text = clean_markdown_text(md_text)
    
    # Remove references section
    if remove_references:
        md_text = remove_references_section(md_text)
    
    logging.info(f"Extracted and processed markdown from PDF: {pdf_path}")
    return md_text

if __name__ == "__main__":
    pdf_test_path = 'lram_papers/design/2017 Jiang.pdf'
    md = extract_markdown_from_pdf(pdf_test_path, clean_text=True)
    
    with open('outputs/test.md', 'w') as f:
        f.write(md)
    print('Saved to outputs/test.md')