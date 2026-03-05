import pytest
import ollama
import os
import sys
from pathlib import Path
from pydantic import BaseModel

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from core import file_chunk
import tests.pipeline_functions as pf

def test_fruit(database_name, query_pdf):
    try: 
        database = pf.create_db(database_name)
        stored_docs = set()
    except:
        database = pf.open_db(database_name)
        all_data = database.get()
        stored_docs = set(id.split(':')[0] for id in all_data['ids'])
    
    for filename in os.listdir("test_docs"):
        if filename.endswith(".pdf"):
            if filename in stored_docs:
                print(f"Skipping {filename} - already in database")
                continue
            pdf_path = os.path.join("test_docs", filename)
            embeds, chunks = pf.embed_pdf(pdf_path, chunk_size=1000, chunk_overlap=200, batch=10)
            pf.store_embeds(database, chunks, embeds, filename)

    query_embeds, query_chunks = pf.embed_pdf(query_pdf, chunk_size=400, chunk_overlap=80, batch=10)
    print(f"Query has {len(query_chunks)} chunks, {len(query_embeds)} embeddings")
    
    ids, retrieve_chunks, embeds = pf.retrieve(database, query_embeds, top_k=5)
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