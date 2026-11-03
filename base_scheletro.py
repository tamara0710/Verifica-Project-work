"""RAG agentico su PDF e Gestione Prenotazioni SQLite con StateGraph standard di LangGraph."""
from pathlib import Path
from typing import TypedDict, Annotated, Literal, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from config import get_llm, get_embeddings
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langgraph.graph.message import add_messages

from pypdf import PdfReader                                  
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from src.database.connection import crea_tabelle_complete
from src.database import repositories


# 1. INIZIALIZZAZIONE DATABASE E VETTORIALE CHROMA (MENÙ)

crea_tabelle_complete()

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
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


# 2. STRUMENTI PER IL DATABASE (PRENOTAZIONI)

@tool
def controlla_disponibilita_tavoli(data: str, turno_id: int, persone: int) -> str:
    """Verifica disponibilità tavoli per una data (YYYY-MM-DD), turno (1=Pranzo, 2=Cena) e numero persone."""
    tavoli = repositories.verifica_disponibilita(data, turno_id, persone)
    if not tavoli:
        return f"Nessun tavolo disponibile per {persone} persone il {data} (turno {turno_id})."
    return f"Tavolo disponibile trovato (ID Tavolo: {tavoli[0][0]}, Zona: {tavoli[0][2]})."

@tool
def salva_prenotazione(nome: str, telefono: str, data: str, turno_id: int, persone: int, email: Optional[str] = None, note: Optional[str] = "", celiachia: bool = False) -> str:
    """Registra e salva una prenotazione nel database della trattoria.
    
    Parametri:
    - nome: Il nome del cliente.
    - telefono: Il recapito telefonico.
    - data: Data della prenotazione in formato YYYY-MM-DD.
    - turno_id: ID del turno (1=Pranzo, 2=Cena).
    - persone: Numero di coperti (da 1 a 12).
    - email: Facoltativa, indirizzo email del cliente.
    - note: Facoltative, richieste speciali o intolleranze generiche.
    - celiachia: Impostare su True se è presente almeno una persona celiaca.
    """
    tavoli = repositories.verifica_disponibilita(data, turno_id, persone)
    if not tavoli:
        return "Impossibile prenotare: nessun tavolo libero."
    
    cliente_id = repositories.trova_o_crea_cliente(nome, telefono, email)
    acconto = (persone * 10.0) if persone > 8 else 0.0
    testo_note = note if note is not None else ""
    prenotazione_id = repositories.inserisci_prenotazione(
        cliente_id, tavoli[0][0], turno_id, data, persone, testo_note, 1 if celiachia else 0, acconto
    )
    return f"Prenotazione confermata con ID #{prenotazione_id} a nome di {nome} per {persone} persone."

tools_db = [controlla_disponibilita_tavoli, salva_prenotazione]
tools_map = {t.name: t for t in tools_db}

@tool
def cerca_menu(query: str) -> str:
    """Cerca nel menu della Trattoria Nonna Rosa: piatti, ingredienti, prezzi, allergeni e intolleranze."""
    risultati = retriever.invoke(query)
    return "\n\n".join(f"[pag. {d.metadata.get('page', 0) + 1}] {d.page_content}" for d in risultati)


# 3. STATO E MODELLI DI STRUTTURAZIONE

llm = get_llm()

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    domanda: str
    intento: str
    query: str
    documenti: list[str]
    pertinente: bool
    tentativi: int
    risposta: str

class ClassificaIntento(BaseModel):
    intento: Literal["menu", "prenotazione"] = Field(
        description="Classifica se la richiesta riguarda il menù/cibo ('menu') oppure prenotazioni/disponibilità tavoli ('prenotazione')."
    )

class Giudizio(BaseModel):
    pertinente: bool = Field(description="I documenti contengono le informazioni per rispondere alla domanda?")

classifier = llm.with_structured_output(ClassificaIntento)
grader = get_llm().with_structured_output(Giudizio)


# 4. NODI DEL GRAFO

def classifica_intento(state: State) -> dict:
    """Determina la natura della richiesta per instradare verso il menù o verso il DB."""
    res = classifier.invoke(
        f"Analizza la richiesta dell'utente e stabilisci l'intento ('menu' o 'prenotazione'):\n\n{state['domanda']}"
    )
    return {"intento": res.intento}

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
    # prompt = (
    #     "Sei l'assistente virtuale della Trattoria Nonna Rosa.\n"
    #     "Rispondi alla domanda dell'utente basandoti ESCLUSIVAMENTE sul seguente testo estratto dal menu.\n\n"
    #     "REGOLE RIGIDE:\n"
    #     "1. Per domande su ALLERGENI e INTOLLERANZE: riporta solo ed esclusivamente ciò che è esplicitamente indicato nel testo.\n"
    #     "2. Non fare supposizioni o inferenze basate su ricette tradizionali se l'allergene non è scritto chiaramente nel testo fornito.\n"
    #     "3. Se l'informazione sull'allergene non è specificata nel testo, dichiara chiaramente che nel menu non viene specificata.\n\n"
    #     f"--- CONTESTO ESTRATTO DAL MENU ---\n{ctx}\n-----------------------------------\n\n"
    #     f"Domanda: {state['domanda']}"
    # )
#     prompt = (
    #     "Sei l'assistente virtuale della Trattoria Nonna Rosa.\n"
    #     "Rispondi alla domanda dell'utente basandoti ESCLUSIVAMENTE sul testo del menu fornito.\n"
    #     "Controlla attentamente la colonna 'Allergeni' prima di rispondere.\n"
    #     "Se un piatto ha scritto '(senza glutine)' nel nome o non presenta 'glutine' tra gli allergeni, indicalo chiaramente.\n"
    #     "Non inventare ingredienti.\n\n"
    #     f"--- CONTESTO ESTRATTO DAL MENU ---\n{ctx}\n-----------------------------------\n\n"
    #     f"Domanda: {state['domanda']}"
#     ) 
    prompt = (
        "Sei l'assistente virtuale della Trattoria Nonna Rosa.\n"
        "Rispondi alla domanda dell'utente basandoti ESCLUSIVAMENTE sul testo del menu fornito sotto.\n\n"
        "REGOLE DI ANALISI RIGIDE:\n"
        "1. Analizza il testo riga per riga. Ogni riga rappresenta un piatto con il proprio prezzo e i propri allergeni.\n"
        "2. Identifica chiaramente il blocco denominato 'Primi'.\n"
        "3. Per ogni piatto sotto la sezione 'Primi', verifica ESPLICITAMENTE la presenza della parola 'glutine' nella colonna degli allergeni.\n"
        "4. Se la parola 'glutine' è elencata tra gli allergeni di quel piatto, esso CONTIENE glutine. Non puoi includerlo come 'senza glutine'.\n"
        "5. Rispondi elencando solo ed esclusivamente i primi piatti che NON presentano l'allergene glutine.\n\n"
        f"--- CONTESTO ESTRATTO DAL MENU ---\n{ctx}\n-----------------------------------\n\n"
        f"Domanda: {state['domanda']}"
    )  
      
    r = llm.invoke(prompt)
    return {"risposta": r.content}

def non_so(state: State) -> dict:
    return {"risposta": "Mi dispiace, non ho trovato questa informazione nel menu della trattoria."}

def gestisci_prenotazione(state: State) -> dict:
    """Invoca l'LLM collegato ai tool del DB ed esegue le operazioni SQL necessarie."""
    llm_with_tools = llm.bind_tools(tools_db)
    prompt = (
        "Sei l'assistente virtuale della Trattoria Nonna Rosa per la gestione delle prenotazioni.\n"
        "Usa gli strumenti a disposizione per verificare i tavoli o completare la prenotazione nel DB.\n"
        "Turni validi: 1 = Pranzo (12:30), 2 = Cena.\n"
        "Se mancano dettagli obbligatori (nome, telefono, data YYYY-MM-DD, turno, numero persone), richiedili al cliente."
    )
    msg = llm_with_tools.invoke([SystemMessage(content=prompt), HumanMessage(content=state["domanda"])])
    
    # Se l'LLM seleziona uno o più tool per interagire con il DB
    if msg.tool_calls:
        esiti = []
        for call in msg.tool_calls:
            tool_func = tools_map[call["name"]]
            esiti.append(tool_func.invoke(call["args"]))
        
        # Genera la risposta finale all'utente sintetizzando il risultato della query SQL
        risposta_finale = llm.invoke(
            f"Domanda utente: {state['domanda']}\n"
            f"Esito operazione Database: {' | '.join(esiti)}\n\n"
            "Fornisci un riscontro chiaro e ben formattato al cliente."
        ).content
        return {"risposta": risposta_finale}
    
    return {"risposta": msg.content}


# 5. INSTRADAMENTO E COMPILAZIONE GRAFO

def instrada_intento(state: State) -> str:
    if state["intento"] == "prenotazione":
        return "gestisci_prenotazione"
    return "recupera"

def instrada_valutazione(state: State) -> str:
    if state["pertinente"]:
        return "genera"
    return "riscrivi" if state.get("tentativi", 0) < 2 else "non_so"

builder = StateGraph(State)
# for n, f in [("recupera", recupera), ("valuta", valuta), ("riscrivi", riscrivi),
#              ("genera", genera), ("non_so", non_so)]:
#     builder.add_node(n, f)
# builder.add_edge(START, "recupera")
# builder.add_edge("recupera", "valuta")
# builder.add_conditional_edges("valuta", instrada, ["genera", "riscrivi", "non_so"])
# builder.add_edge("riscrivi", "recupera")
# builder.add_edge("genera", END)
# builder.add_edge("non_so", END)
# graph = builder.compile()

# Registrazione Nodi
builder.add_node("classifica_intento", classifica_intento)
builder.add_node("recupera", recupera)
builder.add_node("valuta", valuta)
builder.add_node("riscrivi", riscrivi)
builder.add_node("genera", genera)
builder.add_node("non_so", non_so)
builder.add_node("gestisci_prenotazione", gestisci_prenotazione)

# Connessioni e Condizioni
builder.add_edge(START, "classifica_intento")
builder.add_conditional_edges("classifica_intento", instrada_intento, ["recupera", "gestisci_prenotazione"])

# Percorso CRAG (Menù)
builder.add_edge("recupera", "valuta")
builder.add_conditional_edges("valuta", instrada_valutazione, ["genera", "riscrivi", "non_so"])
builder.add_edge("riscrivi", "recupera")
builder.add_edge("genera", END)
builder.add_edge("non_so", END)

# Percorso DB (Prenotazione)
builder.add_edge("gestisci_prenotazione", END)

# Compilazione del Grafo con Memoria
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)


# if __name__ == "__main__":
#     config = {"configurable": {"thread_id": "tavolo_1"}}

#     # Prima domanda
#     domanda_1 = "Ci sono primi piatti nel menu?"
#     print(f"=== DOMANDA 1: {domanda_1} ===")
#     risultato_1 = graph.invoke({"domanda": domanda_1}, config=config)
#     print(f"Risposta 1:\n{risultato_1['risposta']}\n")

#     # Seconda domanda (fa riferimento contestuale alla prima)
#     domanda_2 = "Quali di questi contengono allergeni indicati nel menu?"
#     print(f"=== DOMANDA 2: {domanda_2} ===")
#     risultato_2 = graph.invoke({"domanda": domanda_2}, config=config)
#     print(f"Risposta 2:\n{risultato_2['risposta']}\n")


# 6. TEST ESECUZIONE

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "sessione_tavolo_1"}}

    # Test 1: Query Menù (Ramo CRAG)
    domanda_1 = "Quali primi senza glutine ci sono?"
    print(f"=== TEST 1 (MENÙ): {domanda_1} ===")
    r1 = graph.invoke({"domanda": domanda_1}, config=config)
    print(f"Risposta 1:\n{r1['risposta']}\n")

    # # Test 2: Inserimento Prenotazione (Ramo DB SQLite)
    # domanda_2 = "Vorrei prenotare un tavolo per 4 persone il 2026-11-14 per il pranzo (turno 1). Nome: Mario Rossi, tel: 3331234567. Tra noi c'è una persona celiaca."
    # print(f"=== TEST 2 (PRENOTAZIONE DB): {domanda_2} ===")
    # r2 = graph.invoke({"domanda": domanda_2}, config=config)
    # print(f"Risposta 2:\n{r2['risposta']}\n")