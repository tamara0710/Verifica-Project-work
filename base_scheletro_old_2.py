"""Pattern CRAG."""
from pathlib import Path
from typing import TypedDict, Annotated, Literal, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from config import get_llm, get_embeddings
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langgraph.graph.message import add_messages
                               
from langchain_core.documents import Document
from langchain_chroma import Chroma

from src.database.connection import crea_tabelle_complete
from src.database import repositories
from src.utils import da_pdf_a_md, carica_md_in_vectorstore


# 1. INIZIALIZZAZIONE DB, DB VETTORIALE CHROMA (MENÙ), INIZIALIZZAZIONE CAG in memoria RAM

da_pdf_a_md()
crea_tabelle_complete()
retriever = carica_md_in_vectorstore()

BASE_DIR = Path(__file__).parent
PATH_INFO = BASE_DIR / "data" / "info_trattoria.md"

# def carica_info_cag() -> str:
#     """Carica in RAM il file statico delle informazioni generali e FAQ."""
#     if PATH_INFO.exists():
#         return PATH_INFO.read_text(encoding="utf-8")
#     return "Trattoria Nonna Rosa - Bari. Telefono: 080 1234567."

def carica_info_cag() -> str:
    """Carica in RAM il file statico delle FAQ se esiste, altrimenti stringa vuota."""
    if PATH_INFO.exists():
        return PATH_INFO.read_text(encoding="utf-8")
    print("WARNING: info_trattoria.md non trovato. Le richieste CAG passeranno direttamente al RAG.")
    return ""

CONTESTO_CAG = carica_info_cag()

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
    # Ritorna il testo estratto dai chunk del file Markdown
    return "\n\n".join(f"[Fonte: {d.metadata.get('source', 'menu_trattoria.md')}] {d.page_content}" for d in risultati)


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
    esito_cag: bool

# class ClassificaIntento(BaseModel):
#     intento: Literal["menu", "prenotazione"] = Field(
#         description="Classifica se la richiesta riguarda il menù/cibo ('menu') oppure prenotazioni/disponibilità tavoli ('prenotazione')."
#     )

class ClassificaIntento(BaseModel):
    intento: Literal["info_generali", "prenotazione"] = Field(
        description=(
            "Scegli 'info_generali' per domande informative (numero di telefono, orari, indirizzo, "
            "parcheggio, FAQ, menu, allergeni, piatti).\n"
            "Scegli 'prenotazione' SOLO SE l'utente dichiara esplicitamente di voler prenotare o "
            "verificare la disponibilità di un tavolo."
        )
    )

class ValutazioneCAG(BaseModel):
    presente: bool = Field(
        description="Imposta su True SOLO SE l'informazione esatta si trova nelle FAQ/Info generali. Se la domanda chiede l'elenco dei piatti o dettagli del menu (es. quali primi ci sono), imposta tassativamente a False."
    )
    risposta: Optional[str] = Field(
        default="",
        description="Se presente è True, scrivi la risposta. Se presente è False, lascia vuoto."
    )

# class GiudizioCAG(BaseModel):
#     trovato: bool = Field(description="L'informazione richiesta è chiaramente presente nel testo statico delle FAQ/Info?")
#     risposta: str = Field(description="Risposta completa e ben formattata se trovato, altrimenti stringa vuota.")

class Giudizio(BaseModel):
    pertinente: bool = Field(description="I documenti contengono le informazioni per rispondere alla domanda?")

classifier = llm.with_structured_output(ClassificaIntento)
# grader_cag = llm.with_structured_output(GiudizioCAG)
grader = get_llm().with_structured_output(Giudizio)
grader_cag = llm.with_structured_output(ValutazioneCAG)


# 4. NODI DEL GRAFO

# def classifica_intento(state: State) -> dict:
#     """Determina la natura della richiesta per instradare verso il menù o verso il DB."""
#     res = classifier.invoke(
#         f"Analizza la richiesta dell'utente e stabilisci l'intento ('menu' o 'prenotazione'):\n\n{state['domanda']}"
#     )
#     return {"intento": res.intento}

def classifica_intento(state: State) -> dict:
    """Determina se instradare verso le informazioni (CAG/RAG) o il Database."""
    prompt = (
        "Classifica la richiesta dell'utente:\n"
        "- 'info_generali': domande su contatti (telefono, email), indirizzo, orari, FAQ, menu o piatti.\n"
        "- 'prenotazione': intenzione esplicita di prenotare un tavolo o verificare la disponibilità.\n\n"
        f"Domanda: {state['domanda']}"
    )
    res = classifier.invoke(prompt)
    return {"intento": res.intento}

# NODO CAG AGGIUNTO
# def gestisci_cag(state: State) -> dict:
#     """Verifica se la domanda trova risposta nelle FAQ/Info generali (CAG)."""
#     prompt = (
#         "Rispondi alla domanda dell'utente basandoti ESCLUSIVAMENTE sulle informazioni generali e FAQ seguenti.\n"
#         "Se l'informazione NON è presente o la domanda richiede il menu dettagliato dei piatti, rispondi tassativamente con: 'NON_TROVATO'.\n\n"
#         f"--- INFORMAZIONI GENERALI ---\n{CONTESTO_CAG}\n-----------------------------\n\n"
#         f"Domanda: {state['domanda']}"
#     )
#     res = llm.invoke(prompt).content.strip()
    
#     if "NON_TROVATO" not in res.upper():
#         return {"risposta": res, "esito_cag": True}
#     return {"esito_cag": False}

def gestisci_cag(state: State) -> dict:
    print("--- [NODO] ESECUZIONE CAG ---")
    
    # GUARDRAIL: Se il contesto CAG è vuoto o mancante, passa subito al RAG
    if not CONTESTO_CAG or not CONTESTO_CAG.strip():
        print("--> CAG: Contesto statico vuoto/mancante. Passaggio immediato al RAG...")
        return {"esito_cag": False}

    prompt = (
        "Sei l'assistente della Trattoria Nonna Rosa.\n"
        "Verifica se la domanda trova risposta completa nelle FAQ/Info generali statiche.\n"
        "REGOLE IMPORTANTI:\n"
        "- Se la domanda chiede specificamente argomenti collegati al menu o agli allergeni dei piatti, "
        "imposta 'presente' su False per inoltrare la ricerca al menu RAG.\n\n"
        f"--- INFORMAZIONI GENERALI ---\n{CONTESTO_CAG}\n-----------------------------"
    )
    res = grader_cag.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=state["domanda"])
    ])
    
    if res.presente and res.risposta:
        print("--> CAG: Informazione trovata nelle FAQ generali.")
        return {"risposta": res.risposta, "esito_cag": True}
    
    print("--> CAG: Informazione NON trovata nelle FAQ. Passaggio al RAG...")
    return {"esito_cag": False}

# NODI RAG ESISTENTI (CRAG)
def recupera(state: State) -> dict:
    print("--- [NODO] ESECUZIONE RAG (recupera) ---")
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
    print("--- [NODO] ESECUZIONE RAG (genera) ---")
    documenti = state.get("documenti", [])
    if not documenti:
        return {"risposta": "Mi dispiace, non ho trovato informazioni rilevanti nel menu per rispondere alla tua richiesta."}

    ctx = "\n\n".join(documenti)

        # prompt = (
    #     "Sei l'assistente virtuale della Trattoria Nonna Rosa.\n"
    #     "Rispondi alla domanda dell'utente basandoti ESCLUSIVAMENTE sul testo del menu fornito sotto.\n\n"
    #     "REGOLE CRITICHE E IMPERATIVE PER GLI ALLERGENI:\n"
    #     "1. Analizza ogni singola riga del menu fornito nel contesto in modo isolato.\n"
    #     "2. Per ogni piatto, controlla attentamente la sezione finale dopo la parola 'Allergeni:'.\n"
    #     "3. Se la parola 'glutine' compare nell'elenco degli allergeni di un piatto, quel piatto CONTIENE GLUTINE. È un errore gravissimo per la salute del cliente affermare il contrario.\n"
    #     "4. Ogni piatto che ha gli allergeni dedicati significa che continene quegli allergeni.\n"
    #     "5. Gli unici piatti senza glutine, vegetariani o vegani sono specificati accanto al nome del piatto.\n"
    #     f"--- CONTESTO ESTRATTO DAL MENU ---\n{ctx}\n-----------------------------------\n\n"
    #     f"Domanda: {state['domanda']}"
    # )   

    prompt = (
        "Sei l'assistente virtuale della Trattoria Nonna Rosa.\n"
        "Rispondi alla domanda dell'utente basandoti ESCLUSIVAMENTE sul contesto estratto dal menu fornito sottostante.\n\n"
        "REGOLE TASSATIVE SUL MENU E SUGLI ALLERGENI:\n"
        "1. Elenca SOLO ed esclusivamente i piatti e le opzioni citati nel testo.\n"
        "2. NON inventare, supporre o suggerire mai piatti, ingredienti o consigli non presenti nel testo.\n"
        "3. Interpretazione tabelle ed allergeni:\n"
        "   - Se la colonna Allergeni di un piatto contiene 'glutine', quel piatto CONTIENE GLUTINE.\n"
        "   - Gli unici piatti idonei per celiaci/senza glutine sono quelli contrassegnati espressamente come '(senza glutine)' nel nome oppure se il testo indica opzioni modificabili su richiesta (es. pasta senza glutine).\n"
        "4. Se l'informazione richiesta non è presente nel testo, rispondi chiaramente di non avere tale dato nel menu.\n\n"
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
    # prompt = (
    #     "Sei l'assistente virtuale della Trattoria Nonna Rosa per la gestione delle prenotazioni.\n"
    #     "Usa gli strumenti a disposizione per verificare i tavoli o completare la prenotazione nel DB.\n"
    #     "Turni validi: 1 = Pranzo (12:30), 2 = Cena.\n"
    #     "Se mancano dettagli obbligatori (nome, telefono, data YYYY-MM-DD, turno, numero persone), richiedili al cliente."
    # )
    prompt = (
        "Sei l'assistente virtuale della Trattoria Nonna Rosa per la gestione delle prenotazioni.\n\n"
        "REGOLE FONDAMENTALI PER I TOOL:\n"
        "1. NON invocare 'salva_prenotazione' o 'controlla_disponibilita_tavoli' se l'utente non richiede esplicitamente di voler fare una prenotazione.\n"
        "2. NON invocare 'salva_prenotazione' o 'controlla_disponibilita_tavoli' se l'utente non ha specificato "
        "tutti i parametri obbligatori (data YYYY-MM-DD, turno 1/2, numero persone, nome e telefono).\n"
        "3. Se mancano dettagli obbligatori o l'utente fa solo domande generali, rispondi chiedendo i dati mancanti "
        "SENZA invocare strumenti.\n"
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
    return "gestisci_cag"

def instrada_dopo_cag(state: State) -> str:
    """Se il CAG ha risolto la risposta termina, altrimenti passa al RAG su ChromaDB."""
    if state.get("esito_cag", False):
        return END
    return "recupera"

def instrada_valutazione(state: State) -> str:
    if state["pertinente"]:
        return "genera"
    return "riscrivi" if state.get("tentativi", 0) < 2 else "non_so"

builder = StateGraph(State)

# Registrazione Nodi
builder.add_node("classifica_intento", classifica_intento)
builder.add_node("gestisci_cag", gestisci_cag)  # NODO CAG
builder.add_node("recupera", recupera)           # NODO RAG ESISTENTE
builder.add_node("valuta", valuta)
builder.add_node("riscrivi", riscrivi)
builder.add_node("genera", genera)
builder.add_node("non_so", non_so)
builder.add_node("gestisci_prenotazione", gestisci_prenotazione)

# Connessioni e Condizioni
builder.add_edge(START, "classifica_intento")
builder.add_conditional_edges("classifica_intento", instrada_intento, ["gestisci_cag", "gestisci_prenotazione"])

# Percorso CAG -> Fallback RAG
builder.add_conditional_edges("gestisci_cag", instrada_dopo_cag, [END, "recupera"])

# Percorso CRAG (Menù RAG su Retriever)
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

if __name__ == "__main__":
    # Test 1 con thread dedicato
    # config_cag = {"configurable": {"thread_id": "sessione_test_cag"}}
    # domanda_1 = "Qual è il numero di telefono per contattarvi?"
    # r1 = graph.invoke({"domanda": domanda_1}, config=config_cag)
    # print(f"Risposta 1:\n{r1['risposta']}\n")

    # Test 2 con thread dedicato
    config_rag = {"configurable": {"thread_id": "sessione_test_rag"}}
    domanda_2 = "Quali primi senza glutine ci sono?"
    r2 = graph.invoke({"domanda": domanda_2}, config=config_rag)
    print(f"Risposta 2:\n{r2['risposta']}\n")


# # 6. TEST ESECUZIONE

# if __name__ == "__main__":
#     config = {"configurable": {"thread_id": "sessione_tavolo_1"}}

#     # Test 1: Query Menù (Ramo CRAG)
#     domanda_1 = "Quali primi senza glutine ci sono?"
#     print(f"=== TEST 1 (MENÙ): {domanda_1} ===")
#     r1 = graph.invoke({"domanda": domanda_1}, config=config)
#     print(f"Risposta 1:\n{r1['risposta']}\n")

    # # Test 2: Inserimento Prenotazione (Ramo DB SQLite)
    # domanda_2 = "Vorrei prenotare un tavolo per 4 persone il 2026-11-14 per il pranzo (turno 1). Nome: Mario Rossi, tel: 3331234567. Tra noi c'è una persona celiaca."
    # print(f"=== TEST 2 (PRENOTAZIONE DB): {domanda_2} ===")
    # r2 = graph.invoke({"domanda": domanda_2}, config=config)
    # print(f"Risposta 2:\n{r2['risposta']}\n")

    # print("\n=== VERIFICA CONTENUTO MEMORIZZATO IN CHROMA ===")
    # # Estrae tutti i documenti memorizzati nella collezione corrente
    # tutti_i_docs = retriever.vectorstore.get()
    # for id_doc, testo in zip(tutti_i_docs['ids'], tutti_i_docs['documents']):
    #     print(f"ID: {id_doc} --> Contenuto: {testo}")
    # print("================================================\n")