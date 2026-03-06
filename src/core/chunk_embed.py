import ollama
import logging
from logging_config import setup_logging

def embed_chunks(chunks, batch):
    embeds = []
    logging.info(f"Embedding {len(chunks)} chunks with batch size {batch}")
    
    for i in range(0, len(chunks), batch):
        batch_chunks = chunks[i:i+batch]
        
        try:
            # Try batch embedding first (most efficient)
            embed = ollama.embed(
                    model='mxbai-embed-large',
                    input=batch_chunks,
                    )
            embeds += embed['embeddings']
            logging.info(f"{i+len(batch_chunks)}/{len(chunks)} chunks embedded")
            
        except Exception as e:
            # If batch fails, try individual chunks
            logging.warning(f"Batch {i//batch + 1} failed: {e}. Trying chunks individually...")
            
            for j, single_chunk in enumerate(batch_chunks):
                try:
                    # Truncate if too long (rough estimate: 512 tokens = 2048 chars)
                    if len(single_chunk) > 2000:
                        logging.warning(f"Chunk {i+j+1} is {len(single_chunk)} chars, truncating to 2000")
                        single_chunk = single_chunk[:2000]
                    
                    embed = ollama.embed(
                            model='mxbai-embed-large',
                            input=[single_chunk],
                            )
                    embeds += embed['embeddings']
                    logging.info(f"{i+j+1}/{len(chunks)} chunks embedded (individual)")
                    
                except Exception as e2:
                    # Last resort: use zero vector to keep going
                    logging.error(f"Failed to embed chunk {i+j+1}: {e2}")
                    logging.warning(f"Using zero vector for chunk {i+j+1}")
                    embeds.append([0.0] * 1024)  # mxbai-embed-large dimension
    
    logging.info(f"Final embeds: {len(embeds)} chunks: {len(chunks)}")
    return embeds

