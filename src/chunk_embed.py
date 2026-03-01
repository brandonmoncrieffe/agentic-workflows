import ollama

def embed_chunks(chunks, batch):
    embeds = []
    for i in range(0, len(chunks), batch):
        batch_chunks = chunks[i:i+batch]
        embed = ollama.embed(
                model='mxbai-embed-large',
                input=batch_chunks,
                )
        embeds += embed['embeddings']
        print(f"processed {i+len(batch_chunks)}/{len(chunks)}")
    print("Final embeds:", len(embeds), "chunks:", len(chunks))
    return embeds

