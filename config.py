import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings

# Carica il file .env
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "bge-m3:latest")
BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def get_llm(temperature: float = 0) -> ChatOllama:
    if not MODEL:
        raise RuntimeError("Imposta la variabile OLLAMA_MODEL nel file .env")
    return ChatOllama(model=MODEL, temperature=temperature, base_url=BASE_URL)


def get_embeddings() -> OllamaEmbeddings:
    if not EMBED_MODEL:
        raise RuntimeError("Imposta la variabile OLLAMA_EMBED_MODEL nel file .env")
    return OllamaEmbeddings(model=EMBED_MODEL, base_url=BASE_URL)