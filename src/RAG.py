import ollama
import core.vetor_db as vetor_db
import logging
from logging_config import setup_logging
from core import chunk_embed
from core import file_chunk
from core import file_input
from pathlib import Path
from pydantic import BaseModel


class LRAM_paper(BaseModel):
    title: str
    authors: str
    abstract: str
    method: str
    results: str
    conclusion: str

def vectorize(collection_name, pdf_paths, dev_mode, chunk_size=1000, chunk_overlap=200, batch=10):
    chroma_client = vetor_db.initialize_client()
    pdf_paths = list(Path(pdf_paths).glob('*.pdf'))
    
    try: 
        collection = vetor_db.open_collection(chroma_client, collection_name)
    except:
        collection = vetor_db.create_collection(chroma_client, collection_name)
    
    collection_data = collection.get(collection_name) 
    existing_paths = collection_data['documents']
    
    for pdf_path in pdf_paths:
        if str(pdf_path) in existing_paths:
            logging.info(f"PDF already processed, skipping: {pdf_path}")
            continue
        md_text = file_input.extract_markdown_from_pdf(pdf_path)
        chunks = file_chunk.chunk_markdown(md_text, chunk_size, chunk_overlap)
        if dev_mode:
            embeds = chunk_embed.embed_chunks(chunks, batch)
            vetor_db.add_embeds(collection, embeds, chunks, pdf_path)
        else: 
            summary = file_chunk.summarizer(md_text, pdf_path)
            contextualized_chunks = file_chunk.contextual_chunker(chunks, summary, pdf_path)
            embeds = chunk_embed.embed_chunks(contextualized_chunks, batch)
            vetor_db.add_embeds(collection, embeds, contextualized_chunks, pdf_path)
    logging.info("Vectorization complete.")

def query(collection_name, query_pdf, chunk_size=400, chunk_overlap=80, batch=10, top_k=5):
    chroma_client = vetor_db.initialize_client()
    md_text = file_input.extract_markdown_from_pdf(query_pdf)
    chunks = file_chunk.chunk_markdown(md_text, chunk_size, chunk_overlap)
    contextualized_chunks = file_chunk.contextual_chunker(chunks, md_text, query_pdf)
    embeds = chunk_embed.embed_chunks(contextualized_chunks, batch)
    collection = vetor_db.open_collection(chroma_client, collection_name)
    ids, context_chunks, context_embeds = vetor_db.retrieve(collection, embeds, top_k)
    return ids, context_chunks, context_embeds, contextualized_chunks

def synthesize_response(input_chunks, context_chunks):
    input_chunks = file_chunk.format_chunks(input_chunks)
    context_chunks = file_chunk.format_chunks(context_chunks)

    prompt = f"""
    INPUT:
    {input_chunks}

    CONTEXT:
    {context_chunks}

    OUTPUT RULES:
    - If the answer is not explicitly present, return "unknown".
    """
    response = ollama.chat(
        model='qwen2.5:7b-instruct',
        messages=[{"role": "user", "content": prompt}],
        format=LRAM_paper.model_json_schema(),
        options={ "temperature": 0.0 }    
    )
    return response

def RAG(collection_name, query_pdf):
    ids, context_chunks, context_embeds, query_chunk = query(collection_name, query_pdf)
    response = synthesize_response(query_chunk, context_chunks)
    pass
    
if __name__ == "__main__":
    # vetor_db.remove_collection(vetor_db.initialize_client(), "lram_papers"
    collection_name = "LRAM-database"
    pdf_paths = 'tests\lram_papers'
    vectorize(collection_name, pdf_paths, dev_mode=False)