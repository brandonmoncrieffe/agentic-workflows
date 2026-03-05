from core import file_input
from core import file_chunk
from core import chunk_embed
import core.vetor_db as vetor_db
from pydantic import BaseModel
import ollama
import os


def embed_pdf(pdf_path: str, chunk_size: int, chunk_overlap: int, batch: int) -> list[list[float]]:
    md_text = file_input.extract_markdown_from_pdf(pdf_path)
    chunks = file_chunk.chunk_markdown(md_text, chunk_size, chunk_overlap)
    contextualized_chunks = file_chunk.contextual_chunker(chunks, md_text, pdf_path)
    embeds = chunk_embed.embed_chunks(contextualized_chunks, batch)
    return embeds, contextualized_chunks

def create_db(collection_name: str):
    chroma_client = vetor_db.initialize_client()
    collection = vetor_db.create_collection(chroma_client, collection_name)
    return collection

def open_db(collection_name: str):
    chroma_client = vetor_db.initialize_client()
    collection = vetor_db.open_collection(chroma_client, collection_name)
    return collection

def remove_db(collection_name: str):
    chroma_client = vetor_db.initialize_client()
    vetor_db.remove_collection(chroma_client, collection_name) 
    return

def store_embeds(collection, chunks, embeds: list[list[float]], paper_name: str):
    vetor_db.add_embeds(collection, embeds, chunks, paper_name)
    return

def retrieve(collection, input_embeds, top_k):
    results = vetor_db.retrieve(collection, input_embeds, top_k)
    return results