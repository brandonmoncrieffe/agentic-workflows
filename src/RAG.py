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
from templates.prompts import LRAM_PARAMETER_PROMPT, LRAM_BUCKET_PROMPT
from templates.schemas import LRAM_paper_buckets, LRAM_paper_parameters


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
        model='qwen2.5:7b-instruct',
        messages=[{"role": "user", "content": prompt_template.format(input_chunks=input_chunks, context_chunks=context_chunks)}],
        format=response_format.model_json_schema(),
        options={ "temperature": 0.0 }    
    )
    return response

def parameter_sweep(query_pdfs, bucket_schema, parameter_schema, bucket_prompt, parameter_prompt,output_dir='outputs'):
    #retrieve raw pdf(s) location
    query_pdfs = list(Path(query_pdfs).glob('*.pdf'))
    md_contents = {}

    #interate through each paper
    for query_pdf in query_pdfs:
        #clean markdown
        paper_extracted = ingest(query_pdf, dev_mode=True, chunk_size=8000, chunk_overlap=500)
        input_chunks = paper_extracted['chunks']
        save_raw_markdown(paper_extracted['md_text'], query_pdf, output_dir)
        full_text = paper_extracted['md_text']
        if len(full_text) < 100000:  # ~25k tokens
            input_chunks = [full_text]

        bucket_response = synthesize_response(input_chunks, context_chunks=[], prompt_template=bucket_prompt, response_format=bucket_schema)

        # Save to JSON and Markdown
        md_content = save_response(bucket_response, query_pdf, bucket_schema, md=True, output_dir=output_dir)

        parameter_response = synthesize_response(input_chunks, context_chunks=[], prompt_template=parameter_prompt, response_format=parameter_schema)

        save_response(parameter_response, query_pdf, parameter_schema, md=False, output_dir=output_dir)

    pass

def RAG(collection_name, query_pdf, dev_mode):
    ids, context_chunks, context_embeds, query_chunk = query(collection_name, query_pdf, dev_mode)
    response = synthesize_response(query_chunk, context_chunks)
    print(response['message']['content'])
    pass
    
if __name__ == "__main__":
    pdf_paths = 'lram_papers/design'
    parameter_sweep(pdf_paths, LRAM_paper_buckets, LRAM_paper_parameters, LRAM_BUCKET_PROMPT, LRAM_PARAMETER_PROMPT)