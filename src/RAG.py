import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import ollama
import core.vetor_db as vetor_db
import logging
from logging_config import setup_logging
from logging_config.save_outputs import save_response, save_raw_markdown
from core import chunk_embed
from core import file_chunk
from core import file_input
from pathlib import Path
from pydantic import BaseModel
from templates.prompts import SUMMARY_PROMPT, COMPARISON_PROMPT, LRAM_EXTRACTION_PROMPT
from templates.schemas import LRAM_paper


def ingest(pdf_path, dev_mode, chunk_size=1000, chunk_overlap=200):
    md_text = file_input.extract_markdown_from_pdf(pdf_path)
    chunks = file_chunk.chunk_markdown(md_text, chunk_size, chunk_overlap)
    if dev_mode == False:
        summary = file_chunk.summarizer(md_text, pdf_path)
        contextualized_chunks = file_chunk.contextual_chunker(chunks, summary, pdf_path)
    else: 
        summary = None
        contextualized_chunks = None

    return {
        'md_text': md_text,
        'chunks': chunks,
        'summary': summary,
        'contextualized_chunks': contextualized_chunks
    } 

def vectorize(collection_name, pdf_paths, dev_mode, chunk_size=1000, chunk_overlap=200, batch=10):
    chroma_client = vetor_db.initialize_client()
    pdf_paths = list(Path(pdf_paths).glob('*.pdf'))
    logging.info(f"Found {len(pdf_paths)} PDF files to process")

    try: 
        collection = vetor_db.open_collection(chroma_client, collection_name)
    except:
        collection = vetor_db.create_collection(chroma_client, collection_name)
    
    collection_data = collection.get()
    existing_paths = {
        doc_id.rsplit(':', 1)[0] 
        for doc_id in collection_data['ids']
    }

    for pdf_path in pdf_paths:
        if str(pdf_path) in existing_paths:
            logging.info(f"Skipping: {pdf_path}")
            continue
        
        ingested = ingest(pdf_path, dev_mode, chunk_size, chunk_overlap)
        
        if dev_mode:
            chunks_to_embed = ingested['chunks']
        else:
            chunks_to_embed = ingested['contextualized_chunks']
        
        embeds = chunk_embed.embed_chunks(chunks_to_embed, batch)
        vetor_db.add_embeds(collection, embeds, chunks_to_embed, pdf_path)
    
    logging.info("Vectorization complete.")

def query(collection_name, query_pdf, dev_mode, chunk_size=400, chunk_overlap=80, batch=10, top_k=5):
    chroma_client = vetor_db.initialize_client()
    query_pdf = list(Path(query_pdf).glob('*.pdf'))[0]
    
    ingested = ingest(query_pdf, dev_mode, chunk_size, chunk_overlap)
    
    if dev_mode:
        query_chunks = ingested['chunks']
    else:
        query_chunks = ingested['contextualized_chunks']
    
    embeds = chunk_embed.embed_chunks(query_chunks, batch)
    collection = vetor_db.open_collection(chroma_client, collection_name)
    ids, context_chunks, context_embeds = vetor_db.retrieve(collection, embeds, top_k)
    return ids, context_chunks, context_embeds, query_chunks

def synthesize_response(input_chunks, context_chunks, prompt_template, response_format):
    input_chunks = file_chunk.format_chunks(input_chunks)
    context_chunks = file_chunk.format_chunks(context_chunks)

    response = ollama.chat(
        model='qwen3.5:9b',
        messages=[{"role": "user", "content": prompt_template.format(input_chunks=input_chunks, context_chunks=context_chunks)}],
        format=response_format.model_json_schema(),
        options={ "temperature": 0.0 }    
    )
    return response

def parameter_sweep(collection_name, query_pdfs, dev_mode, output_dir='outputs'):
    query_pdfs = list(Path(query_pdfs).glob('*.pdf'))
    results = []
    for query_pdf in query_pdfs:
        # For extraction, use FULL TEXT or much larger chunks
        paper_extracted = ingest(query_pdf, dev_mode=True, chunk_size=8000, chunk_overlap=500)
        input_chunks = paper_extracted['chunks']
        
        # Save raw markdown for inspection
        save_raw_markdown(paper_extracted['md_text'], query_pdf, output_dir)
        
        # If paper is small enough, just use the full text instead of chunks
        full_text = paper_extracted['md_text']
        if len(full_text) < 100000:  # ~25k tokens
            input_chunks = [full_text]
        
        # SUMMARY_PROMPT doesn't use context_chunks, so pass empty list
        response = synthesize_response(input_chunks, [], prompt_template=LRAM_EXTRACTION_PROMPT, response_format=LRAM_paper)
        
        # Save to JSON and Markdown
        paper = save_response(response, query_pdf, LRAM_paper, output_dir)
        results.append(paper)
    
    return results

def RAG(collection_name, query_pdf, dev_mode):
    ids, context_chunks, context_embeds, query_chunk = query(collection_name, query_pdf, dev_mode)
    response = synthesize_response(query_chunk, context_chunks)
    print(response['message']['content'])
    pass
    
if __name__ == "__main__":
    collection_name = "testing_functionality"
    pdf_paths = 'tests/lram_papers'
    query_path = 'tests/lram_papers/test_query'
    parameter_sweep(collection_name, pdf_paths, dev_mode=True)