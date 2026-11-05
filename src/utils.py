from pathlib import Path
import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter  # Import aggiornato
from langchain_core.documents import Document
#from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from config import get_llm, get_embeddings

def da_pdf_a_md():
    base_dir = Path(__file__).parent.parent
    
    pdf_path = base_dir / "data" / "B_menu_trattoria_nonna_rosa.pdf"
    md_path = base_dir / "data" / "menu_trattoria.md"

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"Errore: Il file PDF non esiste in: {pdf_path.resolve()}\n"
            f"Verifica che la cartella 'data' sia nella radice principale del progetto."
        )

    md_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Conversione in corso di: {pdf_path.name}...")
    testo_markdown = pymupdf4llm.to_markdown(str(pdf_path))

    md_path.write_text(testo_markdown, encoding="utf-8")
    print(f"File Markdown generato con successo in: {md_path.resolve()}")


def carica_md_in_vectorstore():
    base_dir = Path(__file__).parent.parent
    md_path = base_dir / "data" / "menu_trattoria.md"
    persist_dir = base_dir / "chroma_db"

    # 1. Verifica esistenza file Markdown
    if not md_path.exists():
        raise FileNotFoundError(
            f"Il file Markdown non esiste in {md_path.resolve()}. "
            f"Esegui prima da_pdf_a_md()."
        )

    # 2. Leggi il contenuto del file Markdown
    contenuto_markdown = md_path.read_text(encoding="utf-8")

    # 3. Configura lo splitter per intestazioni Markdown (# e ##)
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    # 4. Dividi per sezioni (genera direttamente una lista di Document)
    chunks = markdown_splitter.split_text(contenuto_markdown)

    # Arricchisci ogni chunk mantenendo il metadato della fonte originali
    for doc in chunks:
        doc.metadata["source"] = md_path.name

    print(f"Creati {len(chunks)} chunk a partire dal file Markdown.")

    # 5. Inizializza o aggiorna il database Chroma
    vectorstore = Chroma.from_documents(
        chunks, 
        embedding=get_embeddings(),  
        collection_name="menu_trattoria", 
        persist_directory=str(persist_dir),
        ids=[f"chunk-{i}" for i in range(len(chunks))], 
    )
    
    print(f"Database vettoriale salvato/aggiornato in: {persist_dir.resolve()}")
    
    # 6. Configura il retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    return retriever

import shutil
from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.stores import InMemoryStore
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import get_embeddings
import gc
import shutil
import time

# Store globale in memoria per i documenti padre
store_doc_padre = InMemoryStore()

# def carica_md_in_vectorstore_con_ParentDocumentRetriever():
#     # Determinazione dinamica e sicura della root del progetto
#     current_path = Path(__file__).resolve() if "__file__" in globals() else Path.cwd()
#     base_dir = current_path.parent.parent if current_path.parent.name == "src" else current_path.parent
    
#     md_path = base_dir / "data" / "menu_trattoria.md"
#     persist_dir = base_dir / "chroma_db"

#     # Reset completo sia del DB vettoriale su disco sia dello Store in RAM
#     global store_doc_padre
#     store_doc_padre = InMemoryStore()
    
#     if persist_dir.exists():
#         shutil.rmtree(persist_dir)
#         print("🗑️ Vecchio ChromaDB rimosso con successo.")

#     if not md_path.exists():
#         raise FileNotFoundError(f"Il file Markdown non esiste in {md_path.resolve()}.")

#     contenuto_markdown = md_path.write_text(encoding="utf-8") if not md_path.exists() else md_path.read_text(encoding="utf-8")
#     doc_padre = [Document(page_content=contenuto_markdown, metadata={"source": md_path.name})]

#     # 1. Inizializzazione Vectorstore Chroma
#     vectorstore = Chroma(
#         collection_name="menu_trattoria_childs",
#         embedding_function=get_embeddings(),
#         persist_directory=str(persist_dir)
#     )

#     # 2. Splitter per i frammenti figli (150 caratteri per alta precisione vettoriale)
#     child_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
#     #child_splitter = RecursiveCharacterTextSplitter(chunk_size=550, chunk_overlap=100)

#     # 3. Inizializzazione ParentDocumentRetriever
#     retriever = ParentDocumentRetriever(
#         vectorstore=vectorstore,
#         docstore=store_doc_padre,
#         child_splitter=child_splitter,
#         parent_splitter=None, # Mantiene l'intero menu .md come documento padre
#     )

#     # 4. Indicizzazione automatica del padre e generazione figli in Chroma
#     retriever.add_documents(doc_padre, ids=["menu_completo_parent"])
#     #retriever.search_kwargs = {"k": 6}
#     print("✅ ParentDocumentRetriever inizializzato e sincronizzato con successo!")

#     return retriever


def carica_md_in_vectorstore_con_ParentDocumentRetriever():
    # Determinazione dinamica e sicura della root del progetto
    current_path = (
        Path(__file__).resolve() if "__file__" in globals() else Path.cwd()
    )
    base_dir = (
        current_path.parent.parent
        if current_path.parent.name == "src"
        else current_path.parent
    )

    md_path = base_dir / "data" / "menu_trattoria.md"
    persist_dir = base_dir / "chroma_db"

    # Reset completo dello Store in RAM
    global store_doc_padre
    store_doc_padre = InMemoryStore()

    # 1. Rilascio risorse e cancellazione sicura per Windows (gestione PermissionError)
    gc.collect()  # Chiude eventuali handle rimasti aperti su Chroma/SQLite

    if persist_dir.exists():
        try:
            shutil.rmtree(persist_dir)
            print("🗑️ Vecchio ChromaDB rimosso con successo.")
        except PermissionError:
            # Fallback se il file è ancora temporaneamente bloccato dall'OS
            time.sleep(0.5)
            shutil.rmtree(persist_dir, ignore_errors=True)
            print(
                "⚠️ Rimozione ChromaDB forzata a causa di file in uso da Windows."
            )

    # 2. Verifica ed estrazione contenuto Markdown
    if not md_path.exists():
        raise FileNotFoundError(
            f"Il file Markdown non esiste in {md_path.resolve()}."
        )

    contenuto_markdown = md_path.read_text(encoding="utf-8")
    doc_padre = [
        Document(
            page_content=contenuto_markdown, metadata={"source": md_path.name}
        )
    ]

    # 3. Inizializzazione Vectorstore Chroma
    vectorstore = Chroma(
        collection_name="menu_trattoria_childs",
        embedding_function=get_embeddings(),
        persist_directory=str(persist_dir),
    )

    # 4. Splitter per i frammenti figli
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=150, chunk_overlap=20
    )

    # 5. Inizializzazione ParentDocumentRetriever
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store_doc_padre,
        child_splitter=child_splitter,
        parent_splitter=None,  # Mantiene l'intero menu .md come documento padre
    )

    # 6. Indicizzazione automatica del padre e generazione figli in Chroma
    retriever.add_documents(doc_padre, ids=["menu_completo_parent"])
    print("✅ ParentDocumentRetriever inizializzato e sincronizzato con successo!")

    return retriever