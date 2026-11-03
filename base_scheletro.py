"""RAG agentico su PDF: indicizzazione, retriever come tool, RAG correttivo."""
from pathlib import Path
from typing import TypedDict
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from config import get_llm, get_embeddings

PDF = Path(__file__).parent / "data" / "B_menu_trattoria_nonna_rosa.pdf"

# [slide]
from pypdf import PdfReader                                  # pip install pypdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# 1 Document per pagina, con metadati
docs = [Document(page_content=p.extract_text(), metadata={"page": i})
        for i, p in enumerate(PdfReader(PDF).pages)]
splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
chunks = splitter.split_documents(docs)                    # pezzi sovrapposti

vectorstore = Chroma.from_documents(
    chunks, embedding=get_embeddings(),
    collection_name="regolamento", persist_directory="./chroma_db",
    ids=[f"chunk-{i}" for i in range(len(chunks))],   # niente duplicati se rilanci
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

@tool
def cerca_regolamento(query: str) -> str:
    """Cerca nel regolamento del coworking: orari, prezzi, disdette, regole."""
    risultati = retriever.invoke(query)
    return "\n\n".join(f"[pag. {d.metadata.get('page', 0) + 1}] {d.page_content}"
                       for d in risultati)
# [/slide]

llm = get_llm()

class State(TypedDict):
    domanda: str
    query: str
    documenti: list[str]
    pertinente: bool
    tentativi: int
    risposta: str

# [slide:crag]
class Giudizio(BaseModel):
    pertinente: bool = Field(description="I documenti contengono la risposta?")

grader = get_llm().with_structured_output(Giudizio)

def recupera(state: State) -> dict:
    q = state.get("query") or state["domanda"]
    return {"documenti": [d.page_content for d in retriever.invoke(q)]}

def valuta(state: State) -> dict:
    g = grader.invoke(f"Domanda: {state['domanda']}\n"
                      f"Documenti: {state['documenti']}")
    return {"pertinente": g.pertinente}

def riscrivi(state: State) -> dict:
    q = llm.invoke(f"Riformula come query di ricerca breve: {state['domanda']}").content
    return {"query": q, "tentativi": state.get("tentativi", 0) + 1}

def genera(state: State) -> dict:
    ctx = "\n".join(state["documenti"])
    r = llm.invoke(f"Rispondi SOLO con queste fonti:\n{ctx}\n"
                   f"Domanda: {state['domanda']}")
    return {"risposta": r.content}

def non_so(state: State) -> dict:
    return {"risposta": "Non ho trovato l'informazione nel regolamento."}

def instrada(state: State) -> str:
    if state["pertinente"]:
        return "genera"
    return "riscrivi" if state.get("tentativi", 0) < 2 else "non_so"
# [/slide:crag]

builder = StateGraph(State)
for n, f in [("recupera", recupera), ("valuta", valuta), ("riscrivi", riscrivi),
             ("genera", genera), ("non_so", non_so)]:
    builder.add_node(n, f)
builder.add_edge(START, "recupera")
builder.add_edge("recupera", "valuta")
builder.add_conditional_edges("valuta", instrada, ["genera", "riscrivi", "non_so"])
builder.add_edge("riscrivi", "recupera")
builder.add_edge("genera", END)
builder.add_edge("non_so", END)
graph = builder.compile()

if __name__ == "__main__":
    print(graph.invoke({"domanda": "Qualli primi ci sono nel menu?"})["risposta"])

