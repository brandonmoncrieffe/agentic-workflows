import ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_markdown(md_text, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(md_text)
    print(f"Markdown text split into {len(chunks)} chunks.")
    return chunks

def format_chunks(chunks):
    formatted = []
    chunks = chunks[0]

    for i, chunk in enumerate(chunks):
        text = chunk.strip()
        formatted.append(f"CHUNK {i}:\n{text}")

    return "\n\n".join(formatted)

def contextual_chunker(chunks, md_text, file_name):
    WHOLE_DOCUMENT = md_text
    FILE_NAME = file_name
    contextualized_chunks = []
    
    for chunk in chunks:
        chunk_number = chunk 
        prompt = f"""
    You are helping build a retrieval system.

    <document>
    {WHOLE_DOCUMENT}
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
    
    return contextualized_chunks



