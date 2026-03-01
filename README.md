# va-research-agent

A RAG-powered research workflow that processes PDF documents using contextual chunking and vector embeddings for intelligent information retrieval from an internal database.

## Setup

Ensure [Ollama](https://ollama.ai) is installed and running. Pull the required models: `ollama pull mxbai-embed-large` and `ollama pull qwen2.5:7b-instruct`. Clone the repository, and install dependencies using `uv sync` from the project root. 