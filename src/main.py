import file_input
import file_chunk
import chunk_embed
import vetor_db
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


def test_fruit(database_name, query_pdf):
    try: 
        database = create_db(database_name)
        stored_docs = set()
    except:
        database = open_db(database_name)
        all_data = database.get()
        stored_docs = set(id.split(':')[0] for id in all_data['ids'])
    
    for filename in os.listdir("test_docs"):
        if filename.endswith(".pdf"):
            if filename in stored_docs:
                print(f"Skipping {filename} - already in database")
                continue
            pdf_path = os.path.join("test_docs", filename)
            embeds, chunks = embed_pdf(pdf_path, chunk_size=1000, chunk_overlap=200, batch=10)
            store_embeds(database, chunks, embeds, filename)

    query_embeds, query_chunks = embed_pdf(query_pdf, chunk_size=400, chunk_overlap=80, batch=10)
    print(f"Query has {len(query_chunks)} chunks, {len(query_embeds)} embeddings")
    
    ids, retrieve_chunks, embeds = retrieve(database, query_embeds, top_k=5)
    print(f"Retrieved structure: {len(retrieve_chunks)} lists")
    print(f"First list has {len(retrieve_chunks[0]) if retrieve_chunks else 0} items")
    total_docs = sum(len(sublist) for sublist in retrieve_chunks) if isinstance(retrieve_chunks[0], list) else len(retrieve_chunks)
    print(f"Total retrieved docs: {total_docs}")
    
    class Fruit(BaseModel):
        fruit_name: str
        location: str
        color: str
        size: str
        taste: str
        winter_solution: str

    input = file_chunk.format_chunks(query_chunks)
    context = file_chunk.format_chunks(retrieve_chunks)

    prompt = f"""
    INPUT:
    {input}

    CONTEXT:
    {context}

    OUTPUT RULES:
    - If the answer is not explicitly present, return "unknown".
    - Never copy unrelated sentences just to fill a field.

    """
    response = ollama.chat(
        model='qwen2.5:7b-instruct',
        messages=[{"role": "user", "content": prompt}],
        format=Fruit.model_json_schema(),
        options={ "temperature": 0.0 }    
    )

    print(response['message']['content'])
    
if __name__ == "__main__":
    test_fruit('test_database', "test_docs/test_query/apples_query.pdf")