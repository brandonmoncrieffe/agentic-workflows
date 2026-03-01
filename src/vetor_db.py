from pathlib import Path
import chromadb
 
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PERSIST_DIR = PROJECT_ROOT / "chroma_db"

def initialize_client():
    chroma_client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    print("ChromaDB client initialized at:", PERSIST_DIR)
    return chroma_client

def create_collection(chroma_client, collection_name):
    collection = chroma_client.create_collection(name=collection_name)
    print(f"Collection '{collection_name}' created.")
    return collection

def open_collection(chroma_client, collection_name):
    collection = chroma_client.get_collection(name=collection_name)
    print(f"Collection '{collection_name}' opened.")
    return collection

def remove_collection(chroma_client, collection_name):
    chroma_client.delete_collection(name=collection_name)
    print(f"Collection '{collection_name}' removed.")
    return
def add_embeds(collection, embeds, chunks, paper_name):
    ids = [f"{paper_name}:{i}" for i in range(len(embeds))]

    collection.add(
        ids=ids,
        embeddings=embeds,
        documents=chunks,
    )
    return

def retrieve(collection, input_embeds, top_k):
    results = []
    results = collection.query(
        query_embeddings=input_embeds,
        n_results=top_k,
        include=["documents", "embeddings"]
    )
    ids = results['ids']
    documents = results['documents']
    embeds = results['embeddings']
    return ids, documents, embeds