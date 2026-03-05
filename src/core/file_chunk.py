import logging
import os
from logging_config import setup_logging
import ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_markdown(md_text, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(md_text)
    logging.info(f"Markdown text split into {len(chunks)} chunks.")
    return chunks

def format_chunks(chunks):
    formatted = []
    chunks = chunks[0]

    for i, chunk in enumerate(chunks):
        text = chunk.strip()
        formatted.append(f"CHUNK {i}:\n{text}")
    logging.info(f"Formatted {len(formatted)} chunks for prompt.")
    return "\n\n".join(formatted)

def summarizer(md_text, file_name):
    prompt = f"""
    You are helping build a retrieval system.

    Here is the markdown text of a document:

    <markdown>
    {md_text}
    </markdown>

    Please give a short succinct summary of the document for the purposes of improving search retrieval. Answer only with the succinct summary and nothing else. 
    """
        
    response = ollama.chat(
        model='qwen2.5:7b-instruct',
        messages=[{"role": "system", "content": prompt}]
    )
    summary = response['message']['content']
    logging.info(f"Generated summary for {os.path.basename(file_name)}.")
    return summary

def contextual_chunker(chunks, summary, file_name):
    DOCUMENT_SUMMARY = summary
    FILE_NAME = file_name
    contextualized_chunks = []
    
    for idx, chunk in enumerate(chunks, 1):
        logging.info(f"Processing chunk {idx}/{len(chunks)}...")
        chunk_number = chunk 
        prompt = f"""
    You are helping build a retrieval system.

    <document>
    {DOCUMENT_SUMMARY}
    </document>

    Here is the chunk we want to situate within the whole document:

    <chunk>
    {chunk}
    </chunk>

    Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else. 
    """
        
        response = ollama.chat(
            model='qwen2.5:7b-instruct',
            messages=[{"role": "system", "content": prompt}]
        )
        context = response['message']['content']
        contextualized_chunk = context + "\n\n" + chunk
        contextualized_chunks.append(contextualized_chunk)
    logging.info(f"Contextualized {len(contextualized_chunks)} chunks.")
    return contextualized_chunks



