"""
Utility functions for saving LLM responses to JSON and Markdown formats.
"""
import json
import logging
from pathlib import Path
from datetime import datetime


def save_response(response, pdf_path, response_format, output_dir='outputs'):
    """
    Save LLM response as both JSON and Markdown.
    
    Args:
        response: The raw response from ollama.chat
        pdf_path: Path to the source PDF file
        response_format: The Pydantic model class used for extraction
        output_dir: Directory to save outputs (default: 'outputs')
    
    Returns:
        Parsed Pydantic model instance
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Parse response into Pydantic model
    content = json.loads(response['message']['content'])
    paper = response_format(**content)
    
    # Create filename from PDF name and timestamp
    pdf_name = Path(pdf_path).stem
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_filename = f"{pdf_name}_{timestamp}"
    
    # Save JSON
    json_data = {
        'metadata': {
            'timestamp': timestamp,
            'source_pdf': str(pdf_path),
            'schema': response_format.__name__
        },
        'extracted': paper.model_dump()
    }
    json_path = output_path / f"{base_filename}.json"
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    logging.info(f"Saved JSON to: {json_path}")
    
    # Save Markdown - dynamically format based on available fields
    md_content = f"""# Extraction Results

**Source:** {pdf_path}  
**Extracted:** {timestamp}
**Schema:** {response_format.__name__}

---

"""
    
    # Add all fields from the model
    for field_name, field_value in paper.model_dump().items():
        if field_value is not None:
            md_content += f"## {field_name.replace('_', ' ').title()}\n"
            if isinstance(field_value, list):
                for item in field_value:
                    md_content += f"- {item}\n"
            else:
                md_content += f"{field_value}\n"
            md_content += "\n"
    
    md_path = output_path / f"{base_filename}.md"
    md_path.write_text(md_content)
    logging.info(f"Saved Markdown to: {md_path}")
    
    return paper


def save_raw_markdown(md_text, pdf_path, output_dir='outputs'):
    """
    Save raw markdown extracted from PDF.
    
    Args:
        md_text: Raw markdown text
        pdf_path: Path to the source PDF file
        output_dir: Directory to save outputs (default: 'outputs')
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create filename from PDF name
    pdf_name = Path(pdf_path).stem
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    md_path = output_path / f"{pdf_name}_raw_{timestamp}.md"
    md_path.write_text(md_text)
    logging.info(f"Saved raw markdown to: {md_path}")


def save_batch_summary(results, output_dir='outputs'):
    """
    Save a summary of batch processing results.
    
    Args:
        results: List of LRAM_paper instances
        output_dir: Directory to save the summary
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_path = output_path / f"batch_summary_{timestamp}.md"
    
    summary = f"""# Batch Processing Summary

**Date:** {timestamp}  
**Papers Processed:** {len(results)}

---

"""
    for i, paper in enumerate(results, 1):
        summary += f"""## {i}. {paper.title}

**Authors:** {paper.authors}

**Key Points:**
- Method: {paper.method[:100]}...
- Results: {paper.results[:100]}...

---

"""
    
    summary_path.write_text(summary)
    logging.info(f"Saved batch summary to: {summary_path}")
