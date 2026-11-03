"""Pattern CRAG con routing a 3 vie (CAG / RAG / DB)."""
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


# 1. INIZIALIZZAZIONE DB, DB VETTORIALE CHROMA (MENÙ), CAG IN RAM

da_pdf_a_md()
crea_tabelle_complete()
retriever = carica_md_in_vectorstore()

# Path corretto per lo script posizionato nella root di progetto
BASE_DIR = Path(__file__).parent
PATH_INFO = BASE_DIR / "data" / "info_trattoria.md"

def carica_info_cag() -> str:
    """Carica in RAM il file statico delle FAQ se esiste."""
    if PATH_INFO.exists():
        return PATH_INFO.read_text(encoding="utf-8")
    print("WARNING: info_trattoria.md non trovato.")
    return ""

CONTESTO_CAG = carica_info_cag()


# 2. STRUMENTI DB (PRENOTAZIONI)

@tool
def controlla_disponibilita_tavoli(data: str, turno_id: int, persone: int) -> str:
    """Verifica disponibilità tavoli per una data (YYYY-MM-DD), turno (1=Pranzo, 2=Cena) e numero persone."""
    tavoli = repositories.verifica_disponibilita(data, turno_id, persone)
    if not tavoli:
        return f"Nessun tavolo disponibile per {persone} persone il {data} (turno {turno_id})."
    return f"Tavolo disponibile trovato (ID Tavolo: {tavoli[0][0]}, Zona: {tavoli[0][2]})."

@tool
def salva_prenotazione(nome: str, telefono: str, data: str, turno_id: int, persone: int, email: Optional[str] = None, note: Optional[str] = "", celiachia: bool = False) -> str:
    """Registra e salva una prenotazione nel database della trattoria."""
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


# 3. STATO E CLASSIFICAZIONE INTENTO (3 VIE)

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

class ClassificaIntento(BaseModel):
    intento: Literal["info_generali", "menu", "prenotazione"] = Field(
        description=(
            "- 'info_generali': contatti (telefono, email), indirizzo, password Wi-Fi, parcheggio convenzionato, accessibilità disabili.\n"
            "- 'menu': piatti, prezzi, allergeni, celiachia/senza glutine, orari di apertura, turni, regole su animali domestici, coperto, acconti e disdette.\n"
            "- 'prenotazione': intenzione esplicita di prenotare un tavolo o verificare disponibilità."
        )
    )

class Giudizio(BaseModel):
    pertinente: bool = Field(description="I documenti estratti consentono di rispondere alla domanda?")

classifier = llm.with_structured_output(ClassificaIntento)
grader = get_llm().with_structured_output(Giudizio)


# 4. NODI DEL GRAFO

# def classifica_intento(state: State) -> dict:
#     prompt = (
#         "Classifica la richiesta dell'utente:\n"
#         "- 'info_generali': contatti (telefono, email), indirizzo, orari, parcheggio, FAQ.\n"
#         "- 'menu': piatti, primi, secondi, dolci, prezzi, allergeni, opzioni senza glutine o vegetariane.\n"
#         "- 'prenotazione': intenzione esplicita di prenotare un tavolo.\n\n"
#         f"Domanda: {state['domanda']}"
#     )
#     res = classifier.invoke(prompt)
#     print(f"--- [ROUTING] Intento rilevato: {res.intento} ---")
#     return {"intento": res.intento}

def classifica_intento(state: State) -> dict:
    # system_prompt = (
    #     "Sei un classificatore di intenti rigido per l'assistente di una trattoria.\n\n"
    #     "REGOLE TASSATIVE PER LA CLASSIFICAZIONE:\n"
    #     "1. 'menu': Scegli QUESTA opzione per qualsiasi domanda su piatti, ingredienti, allergeni, "
    #     "intolleranze, celiachia, opzioni senza glutine/vegetariane, prezzi dei piatti o bevande.\n"
    #     "   (Esempio: 'Quali primi senza glutine ci sono?' -> 'menu')\n"
    #     "2. 'info_generali': Scegli QUESTA opzione per domande su numero di telefono, email, indirizzo, "
    #     "orari di apertura, parcheggio, wifi o regole del locale.\n"
    #     "3. 'prenotazione': Scegli QUESTA opzione SOLO ED ESCLUSIVAMENTE se l'utente dichiara "
    #     "ESPLICITAMENTE di voler prenotare un tavolo o verificare la disponibilità di posti "
    #     "(parole chiave: 'prenotare', 'tavolo', 'posto', 'disponibilità'). Se l'utente non menziona la prenotazione di un tavolo, NON scegliere MAI 'prenotazione'."
    # )

    system_prompt = (
        "Sei un classificatore di intenti rigido per l'assistente della Trattoria Nonna Rosa.\n\n"
        "REGOLE TASSATIVE PER LA CLASSIFICAZIONE:\n"
        "1. 'info_generali' (CAG): Domande su contatti (telefono, email), indirizzo, rete/password Wi-Fi, "
        "parcheggio o accessibilità per disabili.\n"
        "2. 'menu' (RAG): Domande su piatti, ingredienti, allergeni, opzioni senza glutine/vegano, prezzi, "
        "ORARI DI APERTURA, turni pranzo/cena, presenza di animali domestici/cani, costo coperto, o regole su acconti e disdette.\n"
        "3. 'prenotazione' (DB): SOLO ED ESCLUSIVAMENTE se l'utente dichiara esplicitamente di voler "
        "prenotare un tavolo o verificare la disponibilità di posti per una determinata data/ora."
    )
    res = classifier.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Domanda dell'utente: {state['domanda']}")
    ])
    print(f"--- [ROUTING] Intento rilevato: {res.intento} ---")
    return {"intento": res.intento}

def gestisci_cag(state: State) -> dict:
    print("--- [NODO] ESECUZIONE CAG ---")
    if not CONTESTO_CAG or not CONTESTO_CAG.strip():
        print("--> CAG vuoto, passaggio al RAG...")
        return {"esito_cag": False}

    prompt = (
        "Sei l'assistente della Trattoria Nonna Rosa.\n"
        "Rispondi alla domanda usando ESCLUSIVAMENTE il testo delle informazioni generali seguenti.\n"
        "Se l'informazione NON è presente nel testo, rispondi tassativamente: 'NON_TROVATO'.\n\n"
        f"--- INFORMAZIONI GENERALI ---\n{CONTESTO_CAG}\n-----------------------------\n\n"
        f"Domanda: {state['domanda']}"
    )
    res = llm.invoke(prompt).content.strip()
    
    if "NON_TROVATO" not in res.upper():
        print("--> CAG: Risposta generata con successo dalle FAQ.")
        return {"risposta": res, "esito_cag": True}
    
    print("--> CAG: Dato non presente nelle FAQ. Passaggio al RAG...")
    return {"esito_cag": False}

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

# def genera(state: State) -> dict:
#     print("--- [NODO] ESECUZIONE RAG (genera) ---")
#     documenti = state.get("documenti", [])
#     if not documenti:
#         return {"risposta": "Mi dispiace, non ho trovato informazioni rilevanti nel menu per rispondere alla tua richiesta."}

#     ctx = "\n\n".join(documenti)

#     prompt = (
#         "Sei l'assistente virtuale della Trattoria Nonna Rosa.\n"
#         "Rispondi alla domanda dell'utente basandoti ESCLUSIVAMENTE sul contesto estratto dal menu fornito sottostante.\n\n"
#         "REGOLE TASSATIVE SUL MENU E SUGLI ALLERGENI:\n"
#         "1. Elenca SOLO ed esclusivamente i piatti e le opzioni espressamente citati nel contesto.\n"
#         "2. NON inventare, supporre o suggerire mai piatti, ricette o consigli che non siano citati nel testo.\n"
#         "3. Interpretazione allergeni:\n"
#         "   - Se la colonna Allergeni di un piatto contiene 'glutine', quel piatto CONTIENE GLUTINE.\n"
#         "   - Gli unici piatti idonei per celiaci/senza glutine sono quelli contrassegnati come '(senza glutine)' nel nome oppure le varianti su richiesta (es. pasta senza glutine citata nelle note).\n"
#         "4. Se l'informazione non è presente nel testo, dichiara di non avere tale dato nel menu.\n\n"
#         f"--- CONTESTO ESTRATTO DAL MENU ---\n{ctx}\n-----------------------------------\n\n"
#         f"Domanda: {state['domanda']}"
#     )

#     r = llm.invoke(prompt)
#     return {"risposta": r.content}

def genera(state: State) -> dict:
    print("--- [NODO] ESECUZIONE RAG (genera) ---")
    documenti = state.get("documenti", [])
    if not documenti:
        return {"risposta": "Mi dispiace, non ho trovato informazioni rilevanti nel menu per rispondere alla tua richiesta."}

    ctx = "\n\n".join(documenti)

    system_prompt = (
        "Sei l'assistente virtuale della Trattoria Nonna Rosa.\n"
        "Rispondi alla domanda dell'utente basandoti ESCLUSIVAMENTE sul contesto estratto dal menu fornito sottostante.\n\n"
        "REGOLE TASSATIVE ED IMPERATIVE SULLA SALUTE (ALLERGENI E CELIACHIA):\n"
        "1. La colonna 'Allergeni' indica gli allergeni PRESENTI nel piatto. Se la colonna contiene 'glutine', quel piatto CONTIENE GLUTINE ed è VIETATO CATEGORICAMENTE indicarlo o proporlo come senza glutine.\n"
        "2. NON aggiungere MAI l'etichetta '(senza glutine)' a piatti che contengono glutine (es. Orecchiette alle cime di rapa, Spaghetti all'assassina, Cavatelli).\n"
        "3. Elenca SOLO ed esclusivamente i piatti realmente presenti nel testo, senza inventare ricette o variazioni non citate."
    )
    
    user_prompt = (
        f"--- CONTESTO ESTRATTO DAL MENU ---\n{ctx}\n-----------------------------------\n\n"
        f"Domanda dell'utente: {state['domanda']}"
    )

    r = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])
    return {"risposta": r.content}

def non_so(state: State) -> dict:
    return {"risposta": "Mi dispiace, non ho trovato questa informazione nel menu della trattoria."}

def gestisci_prenotazione(state: State) -> dict:
    print("--- [NODO] ESECUZIONE DB (prenotazione) ---")
    llm_with_tools = llm.bind_tools(tools_db)
    prompt = (
        "Sei l'assistente virtuale della Trattoria Nonna Rosa per la gestione delle prenotazioni.\n\n"
        "REGOLE FONDAMENTALI PER I TOOL:\n"
        "1. NON invocare 'salva_prenotazione' o 'controlla_disponibilita_tavoli' se l'utente non richiede esplicitamente di voler fare una prenotazione.\n"
        "2. NON invocare i tool se l'utente non ha specificato tutti i parametri obbligatori (data YYYY-MM-DD, turno 1/2, persone, nome, telefono).\n"
        "3. Se mancano dettagli obbligatori, rispondi chiedendo i dati mancanti SENZA invocare strumenti.\n"
    )
    msg = llm_with_tools.invoke([SystemMessage(content=prompt), HumanMessage(content=state["domanda"])])
    
    if msg.tool_calls:
        esiti = []
        for call in msg.tool_calls:
            tool_func = tools_map[call["name"]]
            esiti.append(tool_func.invoke(call["args"]))
        
        risposta_finale = llm.invoke(
            f"Domanda utente: {state['domanda']}\n"
            f"Esito operazione Database: {' | '.join(esiti)}\n\n"
            "Fornisci un riscontro chiaro al cliente."
        ).content
        return {"risposta": risposta_finale}
    
    return {"risposta": msg.content}


# 5. INSTRADAMENTO E COMPILAZIONE GRAFO

def instrada_intento(state: State) -> str:
    if state["intento"] == "prenotazione":
        return "gestisci_prenotazione"
    elif state["intento"] == "menu":
        return "recupera"
    return "gestisci_cag"

def instrada_dopo_cag(state: State) -> str:
    if state.get("esito_cag", False):
        return END
    return "recupera"

def instrada_valutazione(state: State) -> str:
    if state["pertinente"]:
        return "genera"
    return "riscrivi" if state.get("tentativi", 0) < 2 else "non_so"

builder = StateGraph(State)

builder.add_node("classifica_intento", classifica_intento)
builder.add_node("gestisci_cag", gestisci_cag)
builder.add_node("recupera", recupera)
builder.add_node("valuta", valuta)
builder.add_node("riscrivi", riscrivi)
builder.add_node("genera", genera)
builder.add_node("non_so", non_so)
builder.add_node("gestisci_prenotazione", gestisci_prenotazione)

builder.add_edge(START, "classifica_intento")
builder.add_conditional_edges("classifica_intento", instrada_intento, ["gestisci_cag", "recupera", "gestisci_prenotazione"])
builder.add_conditional_edges("gestisci_cag", instrada_dopo_cag, [END, "recupera"])
builder.add_edge("recupera", "valuta")
builder.add_conditional_edges("valuta", instrada_valutazione, ["genera", "riscrivi", "non_so"])
builder.add_edge("riscrivi", "recupera")
builder.add_edge("genera", END)
builder.add_edge("non_so", END)
builder.add_edge("gestisci_prenotazione", END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)


# 6. ESECUZIONE TEST

# if __name__ == "__main__":
#     # Test 1: Telefono (Ramo CAG)
#     config_cag = {"configurable": {"thread_id": "sessione_test_cag"}}
#     domanda_1 = "Qual è il numero di telefono per contattarvi?"
#     print(f"=== TEST 1: {domanda_1} ===")
#     r1 = graph.invoke({"domanda": domanda_1}, config=config_cag)
#     print(f"Risposta 1:\n{r1['risposta']}\n")

#     # Test 2: Primi senza glutine (Ramo RAG)
#     config_rag = {"configurable": {"thread_id": "sessione_test_rag"}}
#     domanda_2 = "Quali primi senza glutine ci sono?"
#     print(f"=== TEST 2: {domanda_2} ===")
#     r2 = graph.invoke({"domanda": domanda_2}, config=config_rag)
#     print(f"Risposta 2:\n{r2['risposta']}\n")

if __name__ == "__main__":
    test_cases = [
        # (ID, Domanda, Ramo Atteso)
        ("TC-01", "Qual è il numero di telefono e l'indirizzo?", "CAG (FAQ)"),
        ("TC-02", "Quali sono gli orari di apertura del weekend?", "CAG (FAQ)"),
        ("TC-03", "Posso portare un cane di taglia grande in sala?", "CAG (FAQ)"),
        ("TC-04", "Quali primi senza glutine ci sono?", "RAG (Menu SG)"),
        ("TC-05", "Ci sono piatti vegani tra i dolci o gli antipasti?", "RAG (Menu Vegan)"),
        ("TC-06", "Le orecchiette alle cime di rapa contengono pesce o lattosio?", "RAG (Allergeni)"),
        ("TC-07", "Avete la pizza margherita o i panini?", "RAG (Piatto assente)"),
        ("TC-08", "Vorrei informazioni sulla caparra per i gruppi", "CAG (Regolamento)"),
        #("TC-09", "Vorrei prenotare un tavolo per 2 persone", "DB (Incompleto - Richiesta Dati)"),
        #("TC-10", "Vorrei prenotare per 4 persone il 2026-11-20 per cena (turno 2). Nome: Mario Rossi, tel: 3331234567.", "DB (Completo - Tool Calling)"),
    ]

    print("==================================================")
    print("      AVVIO BATTERIA DI TEST LANGGRAPH           ")
    print("==================================================\n")

    for test_id, domanda, descrizione in test_cases:
        # thread_id univoco per evitare inquinamento della memoria
        config = {"configurable": {"thread_id": f"sessione_{test_id}"}}
        
        print(f"[{test_id}] TEST: {descrizione}")
        print(f"Domanda: \"{domanda}\"")
        
        risultato = graph.invoke({"domanda": domanda}, config=config)
        
        print(f"Risposta ottenuta:\n{risultato['risposta']}")
        print("-" * 60 + "\n")