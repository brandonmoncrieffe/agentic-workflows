import ollama
import logging
from logging_config import setup_logging

def embed_chunks(chunks, batch):
    embeds = []
    logging.info(f"Embedding {len(chunks)} chunks with batch size {batch}")
    for i in range(0, len(chunks), batch):
        batch_chunks = chunks[i:i+batch]
        embed = ollama.embed(
                model='mxbai-embed-large',
                input=batch_chunks,
                )
        embeds += embed['embeddings']
        logging.info(f"{i+len(batch_chunks)}/{len(chunks)} chunks embedded")
    logging.info(f"Final embeds: {len(embeds)} chunks: {len(chunks)}")
    return embeds

