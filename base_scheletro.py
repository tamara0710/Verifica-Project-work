"""RAG agentico su PDF e Gestione Prenotazioni SQLite con StateGraph standard di LangGraph."""
from pathlib import Path
from typing import TypedDict, Annotated, Literal
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from config import get_llm, get_embeddings
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.types import Command

from pypdf import PdfReader                                  
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from src.database.connection import crea_tabelle_complete
from src.database.repositories import (
    verifica_disponibilita,
    crea_prenotazione,
    annulla_o_consulta
)


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
    ids=[f"chunk-{i}" for i in range(len(chunks))],
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


# 2. REGISTRAZIONE STRUMENTI DEL DATABASE (PRENOTAZIONI)

tools_db = [verifica_disponibilita, crea_prenotazione, annulla_o_consulta]
tools_map = {t.name: t for t in tools_db}


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
        description="Classifica se la richiesta riguarda il menù/cibo ('menu') oppure prenotazioni/disponibilità/annullamenti tavoli ('prenotazione')."
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
    print("\n" + "="*50)
    print("🧠 CLASSIFICAZIONE INTENTO DALL'LLM")
    print("="*50)
    print(f"🎯 Intento rilevato: '{res.intento}'")
    print("-" * 50)
    return {"intento": res.intento}

def recupera(state: State) -> dict:
    q = state.get("query") or state["domanda"]
    print("\n🔍 RAG: Recupero documenti da Chroma Vectorstore...")
    print(f"📥 Query di ricerca: '{q}'")
    docs_estrattti = retriever.invoke(q)
    print(f"📄 Documenti trovati: {len(docs_estrattti)} chunk")
    print("-" * 50)
    return {"documenti": [d.page_content for d in docs_estrattti]}

def valuta(state: State) -> dict:
    g = grader.invoke(
        f"Domanda: {state['domanda']}\n"
        f"Documenti: {state['documenti']}\n"
        "Valuta se i documenti estratti consentono di rispondere adeguatamente alla domanda."
    )
    print(f"⚖️ Valutazione Pertinenza Documenti: {g.pertinente}")
    print("-" * 50)
    return {"pertinente": g.pertinente}

def riscrivi(state: State) -> dict:
    q = llm.invoke(
        f"Riformula come query di ricerca breve focalizzata su piatti, ingredienti o allergeni del menu: {state['domanda']}"
    ).content
    print(f"🔄 Riformulazione Query RAG: '{q}'")
    print("-" * 50)
    return {"query": q, "tentativi": state.get("tentativi", 0) + 1}

def genera(state: State) -> dict:
    ctx = "\n\n".join(state["documenti"])
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
    
    # SYSTEM PROMPT DIRETTIVO E RIGIDO
    prompt = (
        "Sei l'assistente virtuale della Trattoria Nonna Rosa per la gestione delle prenotazioni.\n\n"
        "ISTRUZIONI OBBLIGATORIE PER LA SCELTA DEGLI STRUMENTI:\n"
        "1. Se il cliente chiede di PRENOTARE / CREARE una prenotazione e ti ha fornito i 5 dati fondamentali "
        "(Nome, Telefono, Data YYYY-MM-DD, Turno ID 1 o 2, Numero Persone), DEVI CHIAMARE TASSATIVAMENTE 'crea_prenotazione'.\n"
        "   - NON chiamare 'verifica_disponibilita': il tool 'crea_prenotazione' fa già il controllo di disponibilità da solo!\n"
        "2. Usa 'verifica_disponibilita' ESCLUSIVAMENTE se la domanda è solo informativa (es. 'C'è posto per 4 persone domani?').\n"
        "3. Usa 'annulla_o_consulta' solo se il cliente vuole vedere o cancellare sue prenotazioni esistenti.\n\n"
        "MAPPATURA TURNI: 1 = Pranzo (12:30), 2 = Cena.\n"
        "Se mancano dati obbligatori per completare la prenotazione, NON usare strumenti e chiedi i dati mancanti all'utente."
    )
    msg = llm_with_tools.invoke([SystemMessage(content=prompt), HumanMessage(content=state["domanda"])])
    
    # Se l'LLM seleziona uno o più tool per interagire con il DB
    if msg.tool_calls:
        print("\n" + "="*50)
        print("🛠️  TOOL CALLING RILEVATA DALL'LLM (DATABASE)")
        print("="*50)
        
        esiti = []
        for call in msg.tool_calls:
            nome_tool = call["name"]
            argomenti = call["args"]
            
            # --- STAMPA A TERMINALE DEL TOOL USATO E DEI PARAMETRI ---
            print(f"🔧 Tool invocato: {nome_tool}")
            print(f"📥 Argomenti passati: {argomenti}")
            
            tool_func = tools_map[nome_tool]
            
            # Nota: poiché crea_prenotazione contiene un interrupt(),
            # l'esecuzione scatenerà l'interruzione di LangGraph per la conferma.
            esito = tool_func.invoke(argomenti)
            
            print(f"📤 Risultato del Tool: {esito}")
            print("-" * 50)
            
            esiti.append(esito)
        
        # Genera la risposta finale all'utente sintetizzando il risultato dell'operazione
        risposta_finale = llm.invoke(
            f"Domanda utente: {state['domanda']}\n"
            f"Esito operazione Database: {' | '.join(esiti)}\n\n"
            "Fornisci un riscontro chiaro e ben formattato al cliente."
        ).content
        return {"risposta": risposta_finale}
    
    # Se l'LLM risponde direttamente senza usare tool
    print("\nℹ️  Nessun tool invocato per questa richiesta.")
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


# 6. TEST ESECUZIONE

# 6. TEST ESECUZIONE (CON GESTIONE INTERRUPT)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "sessione_tavolo_2"}}

    domanda_1 = "Ciao, mi chiamo Cristiano Ronaldo, voglio prenotare un tavolo per 5 persone per il 2026-11-10 a pranzo (turno 1), il mio numero è 347557777"
    print(f"=== TEST PRENOTAZIONE DB: {domanda_1} ===")
    
    # 1. Prima invocazione del grafo
    r1 = graph.invoke({"domanda": domanda_1}, config=config)
    
    # 2. Controlliamo se il grafo si è fermato all'interrupt
    stato = graph.get_state(config)
    
    if stato.next:
        # Recuperiamo i dati passati dall'interrupt di crea_prenotazione
        dati_interrupt = stato.tasks[0].interrupts[0].value
        print("\n" + "="*50)
        print("⏸️  INTERRUPT ATTIVATO (Richiesta conferma)")
        print("="*50)
        print(f"📋 Dettagli: {dati_interrupt}")
        print("\n✅ Risposta utente all'interrupt: 'True' (Confermo)")
        print("-" * 50)
        
        # 3. Riprendiamo l'esecuzione passando la conferma (resume=True)
        r1 = graph.invoke(Command(resume=True), config=config)

    # 4. Ora r1['risposta'] contiene l'esito finale dell'INSERT
    print(f"\n💬 Risposta finale:\n{r1['risposta']}\n")