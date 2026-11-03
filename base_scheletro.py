""""RAG agentico su PDF: indicizzazione menu trattoria, retriever come tool, CRAG correttivo con Checkpointer."""
from pathlib import Path
from typing import TypedDict, Annotated
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from config import get_llm, get_embeddings
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from pypdf import PdfReader                                  
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

PDF = Path(__file__).parent / "data" / "B_menu_trattoria_nonna_rosa.pdf"

docs = [Document(page_content=p.extract_text(), metadata={"page": i})
        for i, p in enumerate(PdfReader(PDF).pages)]
splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
chunks = splitter.split_documents(docs)                    

vectorstore = Chroma.from_documents(
    chunks, 
    embedding=get_embeddings(),
    collection_name="menu_trattoria", 
    persist_directory="./chroma_db",
    ids=[f"chunk-{i}" for i in range(len(chunks))],   # niente duplicati se rilanci
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

@tool
def cerca_menu(query: str) -> str:
    """Cerca nel menu della Trattoria Nonna Rosa: piatti, ingredienti, prezzi, allergeni e intolleranze."""
    risultati = retriever.invoke(query)
    return "\n\n".join(f"[pag. {d.metadata.get('page', 0) + 1}] {d.page_content}"
                       for d in risultati)

llm = get_llm()

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    domanda: str
    query: str
    documenti: list[str]
    pertinente: bool
    tentativi: int
    risposta: str

class Giudizio(BaseModel):
    pertinente: bool = Field(description="I documenti contengono le informazioni per rispondere alla domanda?")

grader = get_llm().with_structured_output(Giudizio)

def recupera(state: State) -> dict:
    q = state.get("query") or state["domanda"]
    return {"documenti": [d.page_content for d in retriever.invoke(q)]}

def valuta(state: State) -> dict:
    g = grader.invoke(
        f"Domanda: {state['domanda']}\n"
        f"Documenti: {state['documenti']}\n"
        "Valuta se i documenti estratti consentono di rispondere adeguatamente alla domanda."
    )
    return {"pertinente": g.pertinente}

def riscrivi(state: State) -> dict:
    q = llm.invoke(
        f"Riformula come query di ricerca breve focalizzata su piatti, ingredienti o allergeni del menu: {state['domanda']}"
    ).content
    return {"query": q, "tentativi": state.get("tentativi", 0) + 1}

def genera(state: State) -> dict:
    ctx = "\n\n".join(state["documenti"])
    prompt = (
        "Sei l'assistente virtuale della Trattoria Nonna Rosa.\n"
        "Rispondi alla domanda dell'utente basandoti ESCLUSIVAMENTE sul seguente testo estratto dal menu.\n\n"
        "REGOLE RIGIDE:\n"
        "1. Per domande su ALLERGENI e INTOLLERANZE: riporta solo ed esclusivamente ciò che è esplicitamente indicato nel testo.\n"
        "2. Non fare supposizioni o inferenze basate su ricette tradizionali se l'allergene non è scritto chiaramente nel testo fornito.\n"
        "3. Se l'informazione sull'allergene non è specificata nel testo, dichiara chiaramente che nel menu non viene specificata.\n\n"
        f"--- CONTESTO ESTRATTO DAL MENU ---\n{ctx}\n-----------------------------------\n\n"
        f"Domanda: {state['domanda']}"
    )
    r = llm.invoke(prompt)
    return {"risposta": r.content}

def non_so(state: State) -> dict:
    return {"risposta": "Mi dispiace, non ho trovato questa informazione nel menu della trattoria."}

def instrada(state: State) -> str:
    if state["pertinente"]:
        return "genera"
    return "riscrivi" if state.get("tentativi", 0) < 2 else "non_so"

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

# if __name__ == "__main__":
#     config = {"configurable": {"thread_id": "tavolo_1"}}
    
#     domanda_test = "Ci sono piatti privi di glutine o indicazioni sugli allergeni della pasta?"
#     risultato = graph.invoke({"domanda": domanda_test}, config=config)
    
#     print(f"Risposta: {risultato['risposta']}")

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "tavolo_1"}}

    # Prima domanda
    domanda_1 = "Ci sono primi piatti nel menu?"
    print(f"=== DOMANDA 1: {domanda_1} ===")
    risultato_1 = graph.invoke({"domanda": domanda_1}, config=config)
    print(f"Risposta 1:\n{risultato_1['risposta']}\n")

    # Seconda domanda (fa riferimento contestuale alla prima)
    domanda_2 = "Quali di questi contengono allergeni indicati nel menu?"
    print(f"=== DOMANDA 2: {domanda_2} ===")
    risultato_2 = graph.invoke({"domanda": domanda_2}, config=config)
    print(f"Risposta 2:\n{risultato_2['risposta']}\n")