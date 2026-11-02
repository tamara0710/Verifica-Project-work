# config.py – UNICO punto in cui si sceglie il modello
import os
from langchain_ollama import ChatOllama, OllamaEmbeddings
 
MODEL = os.getenv("OLLAMA_MODEL")              # chat, tag "tools"
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL")  # embedding (RAG)
BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
 
 
def get_llm(temperature: float = 0) -> ChatOllama:
    if not MODEL:
        raise RuntimeError("Imposta la variabile OLLAMA_MODEL")
    return ChatOllama(model=MODEL, temperature=temperature,base_url=BASE_URL)
 
 
def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=EMBED_MODEL, base_url=BASE_URL)