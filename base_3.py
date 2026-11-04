"""Pattern CRAG con routing a 3 vie (CAG / RAG / DB), guardrail anti-allucinazione severo e memoria persistente."""
import json
import re
import sqlite3
from pathlib import Path
from typing import TypedDict, Annotated, Literal, Optional
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore
from langgraph.types import Command, interrupt
from langgraph.graph.message import add_messages

from config import get_llm, get_embeddings
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessageChunk
from langchain_core.tools import tool
import sqlite3
from langgraph.store.sqlite import SqliteStore

from src.database.connection import crea_tabelle_complete
from src.database import repositories
from src.database.repositories import (
    verifica_disponibilita,
    crea_prenotazione,
    annulla_o_consulta,
)
from src.utils import da_pdf_a_md, carica_md_in_vectorstore, carica_md_in_vectorstore_con_ParentDocumentRetriever

# 1. INIZIALIZZAZIONE DB, CAG E PERSISTENZA

BASE_DIR = Path(__file__).parent
PATH_INFO = BASE_DIR / "data" / "info_trattoria.md"

# 1. PERCORSI DB SEPARATI
PATH_TRATTORIA = BASE_DIR / "src" / "database" / "trattoria.db"
PATH_STORE = BASE_DIR / "src" / "database" / "store.sqlite"
PATH_CHECKPOINTS = BASE_DIR / "src" / "database" / "checkpoints.sqlite"

# Assicura la creazione della cartella
PATH_TRATTORIA.parent.mkdir(parents=True, exist_ok=True)

# 2. INIZIALIZZAZIONE TRATTORIA DB (Dati di Dominio: Tavoli, Menu, Prenotazioni)
crea_tabelle_complete()

# 3. INIZIALIZZAZIONE STORE SQLITE (Memoria Cross-Conversazione / Allergie)
store_conn = sqlite3.connect(
    str(PATH_STORE), 
    check_same_thread=False, 
    timeout=20.0,
    isolation_level=None  # <-- AGGIUNGI QUESTO PARAMETRO (Autocommit Mode)
)
store_conn.execute("PRAGMA journal_mode=WAL;")
store = SqliteStore(store_conn)
store.setup()

# 4. INIZIALIZZAZIONE CHECKPOINTER SQLITE (Stato Conversazione / Resume Interrupt)
sqlite_conn = sqlite3.connect(str(PATH_CHECKPOINTS), check_same_thread=False, timeout=20.0, isolation_level=None)
sqlite_conn.execute("PRAGMA journal_mode=WAL;")
checkpointer = SqliteSaver(sqlite_conn)
checkpointer.setup()

# Caricamento RAG e CAG
da_pdf_a_md()
retriever = carica_md_in_vectorstore_con_ParentDocumentRetriever()

def carica_info_cag() -> str:
    if PATH_INFO.exists():
        return PATH_INFO.read_text(encoding="utf-8")
    print("⚠️ WARNING: info_trattoria.md non trovato.")
    return ""

CONTESTO_CAG = carica_info_cag()


# 2. I 4 STRUMENTI RICHIESTI DALLA TRACCIA (Obbligatori: cerca_menu, verifica_disponibilita, crea_prenotazione, annulla_o_consulta)

@tool
def cerca_menu(query: str) -> str:
    """Cerca nel menu e nel regolamento della Trattoria Nonna Rosa (PDF): piatti, ingredienti, prezzi, allergeni, orari e regole su acconti e cani."""
    docs = retriever.invoke(query)
    if not docs:
        return "Nessuna informazione trovata nel menu della trattoria."
    return "\n\n".join(f"- {d.page_content}" for d in docs)

tools_db = [
    cerca_menu,
    verifica_disponibilita,
    crea_prenotazione,
    annulla_o_consulta,
]
tools_map = {t.name: t for t in tools_db}

# Alias e retrocompatibilità per chiamate pregresse
salva_prenotazione = crea_prenotazione
controlla_disponibilita_tavoli = verifica_disponibilita
tools_map["salva_prenotazione"] = crea_prenotazione
tools_map["controlla_disponibilita_tavoli"] = verifica_disponibilita


# def sanifica_argomenti_tool(nome_tool: str, argomenti: dict) -> dict:
#     """Normalizza e sanifica i tipi dei parametri prima dell'invocazione del tool."""
#     args = dict(argomenti)

#     # Normalizzazione chiavi frequenti
#     if "data_prenotazione" in args and "data" not in args:
#         args["data"] = args.pop("data_prenotazione")
#     if "numero_persone" in args and "persone" not in args:
#         args["persone"] = args.pop("numero_persone")
#     if "num_persone" in args and "persone" not in args:
#         args["persone"] = args.pop("num_persone")
#     if "id_turno" in args and "turno_id" not in args:
#         args["turno_id"] = args.pop("id_turno")

#     # Pulizia e casting tipi per evitare errori Pydantic
#     if "email" in args and args["email"] is None:
#         args["email"] = ""
#     if "note" in args and args["note"] is None:
#         args["note"] = ""
#     if "celiachia" in args:
#         if args["celiachia"] is None:
#             args["celiachia"] = False
#         elif isinstance(args["celiachia"], str):
#             args["celiachia"] = args["celiachia"].lower() in ("true", "1", "si", "sì")
#     if "turno_id" in args and args["turno_id"] is not None:
#         try:
#             args["turno_id"] = int(args["turno_id"])
#         except (ValueError, TypeError):
#             pass
#     if "persone" in args and args["persone"] is not None:
#         try:
#             args["persone"] = int(args["persone"])
#         except (ValueError, TypeError):
#             pass
#     if "prenotazione_id" in args and args["prenotazione_id"] is not None:
#         try:
#             args["prenotazione_id"] = int(args["prenotazione_id"])
#         except (ValueError, TypeError):
#             pass
#     if "conferma_perdita_acconto" in args:
#         if args["conferma_perdita_acconto"] is None:
#             args["conferma_perdita_acconto"] = False
#         elif isinstance(args["conferma_perdita_acconto"], str):
#             args["conferma_perdita_acconto"] = args["conferma_perdita_acconto"].lower() in ("true", "1", "si", "sì")

#     return args

from datetime import datetime, timedelta

def sanifica_argomenti_tool(nome_tool: str, argomenti: dict) -> dict:
    """Normalizza e sanifica i tipi dei parametri prima dell'invocazione del tool."""
    args = dict(argomenti)

    # Normalizzazione chiavi frequenti
    if "data_prenotazione" in args and "data" not in args:
        args["data"] = args.pop("data_prenotazione")
    if "numero_persone" in args and "persone" not in args:
        args["persone"] = args.pop("numero_persone")
    if "num_persone" in args and "persone" not in args:
        args["persone"] = args.pop("num_persone")
    if "id_turno" in args and "turno_id" not in args:
        args["turno_id"] = args.pop("id_turno")

    # === 1. RISOLUZIONE VALORI TESTUALI PER LA DATA (Oggi/Domani) ===
    if "data" in args and isinstance(args["data"], str):
        data_pulita = args["data"].lower().strip()
        if "domani" in data_pulita:
            args["data"] = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
            print(f"🔧 [SANIFICAZIONE] Convertito 'domani' in data ISO: {args['data']}")
        elif "oggi" in data_pulita:
            args["data"] = datetime.now().strftime("%Y-%m-%d")
            print(f"🔧 [SANIFICAZIONE] Convertito 'oggi' in data ISO: {args['data']}")

    # === 2. FALLBACK DI SICUREZZA PER TURNO_ID (Evita Pydantic Null Error) ===
    if "turno_id" in args:
        # Se l'LLM ha impostato None o non è riuscito a fare il parsing
        if args["turno_id"] is None:
            # Fallback predefinito di sicurezza (es. 2 per la Cena se non specificato, o 1 per Pranzo)
            # Puoi anche controllare se nella conversazione appare la parola 'cena' o 'pranzo'
            args["turno_id"] = 2  # Imposta un default intero anziché lasciare None
            print(f"🔧 [SANIFICAZIONE] Rilevato turno_id=None. Applicato fallback di sicurezza: {args['turno_id']}")

    # Pulizia e casting tipi per evitare errori Pydantic
    if "email" in args and args["email"] is None:
        args["email"] = ""
    if "note" in args and args["note"] is None:
        args["note"] = ""
    if "celiachia" in args:
        if args["celiachia"] is None:
            args["celiachia"] = False
        elif isinstance(args["celiachia"], str):
            args["celiachia"] = args["celiachia"].lower() in ("true", "1", "si", "sì")
    if "turno_id" in args and args["turno_id"] is not None:
        try:
            args["turno_id"] = int(args["turno_id"])
        except (ValueError, TypeError):
            args["turno_id"] = 2  # Fallback definitivo in caso di stringhe non convertibili
    if "persone" in args and args["persone"] is not None:
        try:
            args["persone"] = int(args["persone"])
        except (ValueError, TypeError):
            pass
    if "prenotazione_id" in args and args["prenotazione_id"] is not None:
        try:
            args["prenotazione_id"] = int(args["prenotazione_id"])
        except (ValueError, TypeError):
            pass
    if "conferma_perdita_acconto" in args:
        if args["conferma_perdita_acconto"] is None:
            args["conferma_perdita_acconto"] = False
        elif isinstance(args["conferma_perdita_acconto"], str):
            args["conferma_perdita_acconto"] = args["conferma_perdita_acconto"].lower() in ("true", "1", "si", "sì")

    return args



# 3. CLASSIFICAZIONE INTENTO E STRUTTURE DATI

llm = get_llm()

class State(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    domanda: str
    telefono: Optional[str]
    intento: Optional[str]
    query: Optional[str]
    documenti: Optional[list[str]]
    pertinente: Optional[bool]
    tentativi: Optional[int]
    risposta: Optional[str]
    esito_cag: Optional[bool]

class ClassificaIntento(BaseModel):
    intento: Literal["info_generali", "menu", "prenotazione"] = Field(
        description=(
            "- 'info_generali': contatti (telefono, email), indirizzo, Wi-Fi, parcheggio, accessibilità disabili.\n"
            "- 'menu': piatti, prezzi, allergeni, celiachia/glutine, vegani, ORARI, turni, CANI/ANIMALI DOMESTICI, coperto, acconti e disdette.\n"
            "- 'prenotazione': intenzione ESPLICITA di prenotare un tavolo, verificare disponibilità o consultare/annullare prenotazioni."
        )
    )

class Giudizio(BaseModel):
    pertinente: bool = Field(description="I documenti estratti consentono di rispondere alla domanda?")

classifier = llm.with_structured_output(ClassificaIntento)
grader = get_llm().with_structured_output(Giudizio)


# 4. NODI DEL GRAFO (CUSTOM GRAPH, NON create_agent)

def classifica_intento(state: State) -> dict:
    system_prompt = (
        "Sei un classificatore di intenti rigido per l'assistente della Trattoria Nonna Rosa.\n\n"
        "REGOLE TASSATIVE PER LA CLASSIFICAZIONE:\n"
        "1. 'info_generali' (CAG): Solo per contatti (telefono, email), indirizzo, Wi-Fi, parcheggio convenzionato e accessibilità disabili.\n"
        "2. 'menu' (RAG): Per QUALSIASI domanda su piatti, ingredienti, allergeni, celiachia, piatti vegani, "
        "ORARI DI APERTURA, turni, CANI/ANIMALI DOMESTICI, costo coperto, acconti/caparra e disdette.\n"
        "3. 'prenotazione' (DB): SOLO ED ESCLUSIVAMENTE se l'utente esprime la VOLONTÀ ESPLICITA di PRENOTARE UN TAVOLO, "
        "chiedere disponibilità posti, o consultare/annullare prenotazioni."
    )
    res = classifier.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Domanda dell'utente: {state['domanda']}")
    ])
    print(f"--- [ROUTING] Intento rilevato: {res.intento} ---")
    return {"intento": res.intento, "tentativi": 0}

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
    print("--- [NODO] ESECUZIONE RAG (recupera con cerca_menu) ---")
    q = state.get("query") or state["domanda"]
    print(f"📥 Query di ricerca: '{q}'")
    docs_estratti = retriever.invoke(q)
    print(f"📄 Documenti trovati: {len(docs_estratti)} chunk")
    return {"documenti": [d.page_content for d in docs_estratti]}

def valuta(state: State) -> dict:
    prompt = (
        f"Domanda: {state['domanda']}\n"
        f"Documenti tratti dal menu: {state.get('documenti', [])}\n\n"
        "Valuta se i documenti contengono informazioni utili per rispondere alla domanda."
    )
    g = grader.invoke(prompt)
    print(f"⚖️ Valutazione Pertinenza Documenti: {g.pertinente}")
    return {"pertinente": g.pertinente}

def riscrivi(state: State) -> dict:
    q = llm.invoke(
        f"Riformula come query di ricerca breve focalizzata su piatti, ingredienti o allergeni del menu: {state['domanda']}"
    ).content
    print(f"🔄 Riformulazione Query RAG: '{q}'")
    return {"query": q, "tentativi": state.get("tentativi", 0) + 1}

def genera(state: State, *, store: Optional[BaseStore] = None) -> dict:
    print("--- [NODO] ESECUZIONE RAG (genera) ---")
    documenti = state.get("documenti", [])
    if not documenti:
        return {"risposta": "Mi dispiace, non ho trovato informazioni rilevanti nel menu per rispondere alla tua richiesta."}

    ctx = "\n\n".join(documenti)

    # Memoria cross-conversazione (Store) per preferenze/allergie
    telefono = state.get("telefono")
    if not telefono and "domanda" in state:
        m = re.search(r'\b(3\d{7,10})\b', state["domanda"])
        if m:
            telefono = m.group(1)

    memoria_allergie = ""
    if store and telefono:
        item = store.get(("clienti", "allergie"), telefono)
        if item and item.value:
            memoria_allergie = item.value.get("allergie", "")
            print(f"🧠 [STORE] Allergie cliente {telefono} usate nella risposta menu: {memoria_allergie}")

    system_prompt = (
        "Sei l'assistente virtuale della Trattoria Nonna Rosa.\n"
        "Rispondi alla domanda dell'utente basandoti ESCLUSIVAMENTE sul testo del menu e del regolamento forniti nel contesto.\n\n"
        "REGOLE RIGIDE DI LETTURA E LOGICA:\n"
        "1. LETTURA RIGIDA DELLA COLONNA ALLERGENI: Se la colonna contiene un allergene, il piatto lo contiene.\n"
        "2. OPZIONI SPECIALI: Considera idonei per celiaci solo piatti segnati '(senza glutine)' o privi di glutine negli allergeni.\n"
        "3. REGOLA ACCONTO (GRUPPI > 8): Gruppi superiori a 8 persone devono versare un acconto di 10 € a persona entro 48 ore. Disdetta gratuita fino a 24 ore prima.\n"
        "4. COMPLETEZZA SU ORARI E REGOLAMENTO: Riporta tutti i dettagli senza omettere orari o cifre.\n"
        "5. DIVIETO ASSOLUTO DI ALLUCINAZIONE: Basa la risposta SOLO sul testo del PDF."
    )
    if memoria_allergie:
        system_prompt += f"\n\n[MEMORIA STORE CLIENTE]: Il cliente ha segnalato in precedenza l'allergia/intolleranza: '{memoria_allergie}'. Personalizza la risposta ricordando questa sua esigenza!"

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

def gestisci_prenotazione(state: State, *, store: Optional[BaseStore] = None) -> dict:
    print("--- [NODO] ESECUZIONE DB (prenotazione) ---")
    llm_with_tools = llm.bind_tools(tools_db)

    # Recupero eventuale numero di telefono per lo Store cross-conversazione
    telefono = state.get("telefono")
    if not telefono and "domanda" in state:
        m = re.search(r'\b(3\d{7,10})\b', state["domanda"])
        if m:
            telefono = m.group(1)

    memoria_allergie = ""
    if store and telefono:
        item = store.get(("clienti", "allergie"), telefono)
        if item and item.value:
            memoria_allergie = item.value.get("allergie", "")
            print(f"🧠 [STORE] Allergie recuperate per cliente {telefono}: {memoria_allergie}")

    adesso = datetime.now()
    data_oggi = adesso.strftime("%Y-%m-%d")
    data_domani = (adesso + timedelta(days=1)).strftime("%Y-%m-%d")
    giorno_settimana = adesso.strftime("%A")

    prompt = (
        "Sei l'assistente virtuale della Trattoria Nonna Rosa per la gestione delle prenotazioni.\n\n"
        f"📅 CONTESTO TEMPORALE CORRENTE (USA QUESTI DATI PER LE DATE RELATIVE):\n"
        f"   - Oggi è: {data_oggi} ({giorno_settimana})\n"
        f"   - Domani sarà: {data_domani}\n\n"
        "REGOLE TASSATIVE PER L'ESTRAZIONE DELLA DATA:\n"
        f"1. Se l'utente scrive 'domani', devi valorizzare il parametro 'data' del tool tassativamente con: {data_domani}.\n"
        f"2. Se l'utente scrive 'oggi', devi valorizzare il parametro 'data' del tool tassativamente con: {data_oggi}.\n"
        "3. Se l'utente non specifica l'anno, usa sempre l'anno corrente corrente.\n\n"
        "ISTRUZIONI OBBLIGATORIE PER LA SCELTA DEGLI STRUMENTI:\n"
        "1. Se il cliente chiede di PRENOTARE / CREARE una prenotazione e ti ha fornito i dati fondamentali "
        "(Nome, Telefono, Data YYYY-MM-DD, Turno ID 1 o 2, Numero Persone), DEVI CHIAMARE TASSATIVAMENTE 'crea_prenotazione'.\n"
        "   - NON chiamare 'verifica_disponibilita': il tool 'crea_prenotazione' fa già il controllo di disponibilità da solo!\n"
        "2. Usa 'verifica_disponibilita' ESCLUSIVAMENTE se la domanda è solo informativa (es. 'C'è posto per 4 persone domani?').\n"
        "3. Usa 'annulla_o_consulta' solo se il cliente vuole consultare o cancellare sue prenotazioni esistenti.\n"
        "4. Usa 'cerca_menu' se la richiesta riguarda chiarimenti su piatti, allergeni o regole generali.\n\n"
        "MAPPATURA TURNI: 1 = Pranzo (12:30), 2 = Cena (19:30).\n"
        "REGOLA ACCONTO (GRUPPI > 8 PERSONE dal PDF): per gruppi superiori a 8 persone è obbligatorio un acconto di 10 € a persona (da versare entro 48 ore).\n"
        "Se mancano dati obbligatori per completare la prenotazione, NON usare strumenti e richiedi cortesemente i dati mancanti all'utente."
        "Se ti viene fornita una data di prenotazione senza l'anno (es. 5 Novembre, 13/04, domani) o altre informazioni da cui puoi estrapolare la data ma senza avere certezza dell'anno DEVI CONSIDERARE L'ANNO IN CORSO."
    )
    if memoria_allergie:
        prompt += f"\n\n[MEMORIA STORE CLIENTE]: Questo cliente ({telefono}) ha segnalato in precedenza allergie/intolleranze: '{memoria_allergie}'. Tieni conto di questa informazione (impostando celiachia=True o inserendola nelle note della prenotazione)!"

    msg = llm_with_tools.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=state["domanda"])
    ])
    
    if msg.tool_calls:
        print("\n" + "=" * 50)
        print("🛠️  TOOL CALLING RILEVATA DALL'LLM (DATABASE)")
        print("=" * 50)
        esiti = []
        for call in msg.tool_calls:
            nome_tool = call["name"]
            argomenti = sanifica_argomenti_tool(nome_tool, call["args"])

            # Se il cliente ha allergia memorizzata e non è specificata esplicitamente, la applichiamo
            if memoria_allergie and "celiachia" in argomenti and not argomenti["celiachia"]:
                if any(k in memoria_allergie.lower() for k in ["celiac", "glutine"]):
                    argomenti["celiachia"] = True

            print(f"🔧 Tool invocato: {nome_tool}")
            print(f"📥 Argomenti passati: {argomenti}")
            
            tool_func = tools_map.get(nome_tool)
            if not tool_func:
                esito = f"Errore: tool '{nome_tool}' non riconosciuto."
            else:
                # Esecuzione tool (scatena interrupt per conferma cliente)
                esito = tool_func.invoke(argomenti)
                
            print(f"📤 Risultato del Tool: {esito}")
            print("-" * 50)
            esiti.append(str(esito))

            # Memorizzazione cross-conversazione nello Store delle allergie/note comunicate
            if store and "telefono" in argomenti and (argomenti.get("celiachia") or argomenti.get("note")):
                allergia_salvata = "Celiachia (senza glutine)" if argomenti.get("celiachia") else str(argomenti.get("note", ""))
                store.put(("clienti", "allergie"), str(argomenti["telefono"]), {
                    "nome": argomenti.get("nome", ""),
                    "allergie": allergia_salvata
                })
                print(f"💾 [STORE] Allergie salvate con successo per {argomenti['telefono']}: {allergia_salvata}")

        risposta_finale = llm.invoke(
            f"Domanda dell'utente: {state['domanda']}\n"
            f"Esito operazione/i Database: {' | '.join(esiti)}\n\n"
            "Fornisci una risposta chiara, cortese e professionale al cliente confermando o spiegando l'esito."
        ).content

        risposta_finale = llm.invoke(
            f"Domanda dell'utente: {state['domanda']}\n"
            f"Esito operazione/i Database: {' | '.join(esiti)}\n\n"
            "Genera la risposta finale per il cliente. REGOLE RIGIDE:\n"
            "- Non usare placeholder come '[Tuo Nome]', '[Il tuo nome]' o firma finale.\n"
            "- Non dire all'utente 'ti consiglio di contattarmi' o 'contattaci', tu SEI l'assistente virtuale del locale, proponi tu di fare i controlli.\n"
            "- Mantieni un tono cordiale da trattoria e spiega chiaramente se il rifiuto è dovuto al numero di persone in base alla disponibilità dei tavoli." # (9 persone richiedono l'acconto o non entrano nei tavoli da 6)
        ).content


        # Se tra gli esiti è presente la ricevuta JSON di crea_prenotazione, la alleghiamo formattata
        for e in esiti:
            if '"esito": "ok"' in e or '"codice":' in e:
                try:
                    dati_json = json.loads(e)
                    json_formattato = json.dumps(dati_json, indent=2, ensure_ascii=False)
                    risposta_finale += f"\n\n🧾 **RICEVUTA DI PRENOTAZIONE (JSON):**\n```json\n{json_formattato}\n```"
                except Exception:
                    pass

        return {"risposta": risposta_finale}
    
    print("\nℹ️  Nessun tool invocato per questa richiesta.")
    return {"risposta": msg.content}


# 5. COSTRUZIONE E COMPILAZIONE GRAFO

def instrada_intento(state: State) -> str:
    if state.get("intento") == "prenotazione":
        return "gestisci_prenotazione"
    elif state.get("intento") == "menu":
        return "recupera"
    return "gestisci_cag"

def instrada_dopo_cag(state: State) -> str:
    if state.get("esito_cag", False):
        return END
    return "recupera"

def instrada_valutazione(state: State) -> str:
    if state.get("pertinente"):
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

# Compilazione del grafo con SqliteSaver e Store cross-conversazione
graph = builder.compile(checkpointer=checkpointer, store=store)


# 6. FUNZIONE DI STREAMING (Facoltativo +5 pt: Streaming dei token in output)

def stream_risposta(graph_instance, input_data: dict, config: dict):
    """Esegue lo streaming dei token in output in tempo reale."""
    print("🌊 [STREAMING OUTPUT]: ", end="", flush=True)
    full_content = ""
    for msg, metadata in graph_instance.stream(input_data, config=config, stream_mode="messages"):
        if isinstance(msg, AIMessageChunk) and msg.content:
            print(msg.content, end="", flush=True)
            full_content += msg.content
    print("\n" + "-" * 50)
    return full_content


# 7. BATTERIA DI TEST COMPLETA (TUTTI I PUNTI DELLA TRACCIA)

# if __name__ == "__main__":
#     print("\n" + "=" * 60)
#     print("      VERIFICA COMPLETA TRACCIA B (OBBLIGATORI + FACOLTATIVI)")
#     print("=" * 60)

#     # ---------------------------------------------------------
#     # TEST 1: PRENOTAZIONE CON INTERRUPT & SQLITESAVER (Obbligatori + Facoltativo SqliteSaver)
#     # ---------------------------------------------------------
#     config_1 = {"configurable": {"thread_id": "sessione_prenotazione_1"}}
#     domanda_1 = "Ciao, mi chiamo Cristiano Ronaldo, voglio prenotare un tavolo per 5 persone per il 2026-11-12 a pranzo (turno 1), il mio numero è 347557777, tra noi c'è una persona celiaca"
#     print(f"\n[TEST 1] PRENOTAZIONE CON INTERRUPT & MEMORIZZAZIONE ALLERGIE")
#     print(f"Domanda: \"{domanda_1}\"")
    
#     r1 = graph.invoke({"domanda": domanda_1}, config=config_1)
#     stato_1 = graph.get_state(config_1)
    
#     if stato_1.next:
#         dati_interrupt = stato_1.tasks[0].interrupts[0].value
#         print("\n" + "="*50)
#         print("⏸️  INTERRUPT ATTIVATO (Richiesta conferma)")
#         print("="*50)
#         print(f"📋 Dettagli: {dati_interrupt}")
#         print("\n✅ Risposta utente all'interrupt: 'True' (Confermo)")
#         print("-" * 50)
        
#         r1 = graph.invoke(Command(resume=True), config=config_1)

#     print(f"\n💬 Risposta finale Test 1:\n{r1['risposta']}\n")

#     # ---------------------------------------------------------
#     # TEST 2: STORE CROSS-CONVERSAZIONE (Facoltativo +5 pt)
#     # Lo stesso cliente fa una nuova domanda in un Thread/Sessione DIVERSA.
#     # Il grafo ricorderà che è celiaco grazie allo Store!
#     # ---------------------------------------------------------
#     config_2 = {"configurable": {"thread_id": "sessione_nuova_thread_2"}}
#     domanda_2 = "Ciao sono Cristiano Ronaldo (tel 347557777), quali primi mi consigliate?"
#     print("\n" + "=" * 60)
#     print("[TEST 2] MEMORIA STORE CROSS-CONVERSAZIONE (Nuovo Thread)")
#     print(f"Domanda: \"{domanda_2}\"")
#     r2 = graph.invoke({"domanda": domanda_2}, config=config_2)
#     print(f"\n💬 Risposta Test 2 (con memoria allergie dallo Store):\n{r2['risposta']}\n")

#     # ---------------------------------------------------------
#     # TEST 3: REGOLA ACCONTO GRUPPI > 8 PERSONE (Facoltativo +5 pt)
#     # ---------------------------------------------------------
#     config_3 = {"configurable": {"thread_id": "sessione_acconto_gruppi"}}
#     domanda_3 = "Vorrei informazioni sulla caparra e l'acconto per i gruppi di 10 persone"
#     print("\n" + "=" * 60)
#     print("[TEST 3] REGOLA ACCONTO GRUPPI > 8 PERSONE (da PDF)")
#     print(f"Domanda: \"{domanda_3}\"")
#     r3 = graph.invoke({"domanda": domanda_3}, config=config_3)
#     print(f"\n💬 Risposta Test 3:\n{r3['risposta']}\n")

#     # ---------------------------------------------------------
#     # TEST 4: STREAMING DEI TOKEN IN OUTPUT (Facoltativo +5 pt)
#     # ---------------------------------------------------------
#     config_4 = {"configurable": {"thread_id": "sessione_streaming"}}
#     domanda_4 = "Qual è il numero di telefono e l'indirizzo della trattoria?"
#     print("\n" + "=" * 60)
#     print("[TEST 4] STREAMING TOKEN IN TEMPO REALE")
#     print(f"Domanda: \"{domanda_4}\"")
#     stream_risposta(graph, {"domanda": domanda_4}, config=config_4)
#     print("\n🎉 TUTTI I TEST DELLA TRACCIA COMPLETATI CON SUCCESSO!")


# 7. CHATBOT INTERATTIVO CLI CON COMANDI E GESTIONE RESUME/INTERRUPT

def avvia_chatbot():
    print("\n" + "=" * 65)
    print(" 🍷 BENVENUTO NEL CHATBOT DI TRATTORIA NONNA ROSA (CRAG Architecture)")
    print("=" * 65)
    print(" Scrivi una domanda oppure usa uno dei seguenti comandi:")
    print("   • 'help'   : Mostra la guida ai comandi speciali")
    print("   • 'nuovo'  : Avvia un nuovo thread/sessione di chat")
    print("   • 'utente' : Cambia il numero di telefono dell'utente attivo")
    print("   • 'stato'  : Mostra sessione attiva e memoria dello Store")
    print("   • 'esci'   : Chiude la conversazione")
    print("=" * 65 + "\n")

    # Identificativi predefiniti per la sessione
    session_count = 1
    telefono_corrente = input("📱 Inserisci il tuo numero di telefono (es. 347557777): ").strip()
    if not telefono_corrente:
        telefono_corrente = "347557777"
        print(f"👉 Usato numero predefinito: {telefono_corrente}")

    thread_id = f"sessione_{telefono_corrente}_{session_count}"
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        try:
            # # Controllo se il grafo si trova in uno stato interrotto (Interrupt in sospeso)
            # stato_attuale = graph.get_state(config)
            
            # if stato_attuale.next:
            #     # Gestione ripresa da Interrupt (es. conferma prenotazione)
            #     dati_interrupt = stato_attuale.tasks[0].interrupts[0].value
            #     print("\n" + "=" * 55)
            #     print("⚠️  RICHIESTA DI CONFERMA IN SOSPESO")
            #     print("=" * 55)
            #     print(f"📋 Dettagli: {dati_interrupt}")
                
            #     conferma = input("\n👉 Inserisci 'si' per confermare o 'no' per annullare: ").strip().lower()
            #     esito_bool = conferma in ("si", "sì", "yes", "true", "1")
                
            #     print("\n🔄 Ripresa esecuzione grafo...")
            #     risposta = graph.invoke(Command(resume=esito_bool), config=config)
            #     if "risposta" in risposta and risposta["risposta"]:
            #         print(f"\n🤖 Assistente:\n{risposta['risposta']}\n")
            #     continue

            # Controllo se il grafo si trova in uno stato interrotto (Interrupt in sospeso)
            stato_attuale = graph.get_state(config)
            
            if stato_attuale.next and stato_attuale.tasks:
                task = stato_attuale.tasks[0]
                # VERIFICA SICURA: controlla che ci siano interrupt presenti prima di accedere all'indice 0
                if hasattr(task, "interrupts") and task.interrupts:
                    dati_interrupt = task.interrupts[0].value
                    print("\n" + "=" * 55)
                    print("⚠️  RICHIESTA DI CONFERMA IN SOSPESO")
                    print("=" * 55)
                    print(f"📋 Dettagli: {dati_interrupt}")
                    
                    conferma = input("\n👉 Inserisci 'si' per confermare o 'no' per annullare: ").strip().lower()
                    esito_bool = conferma in ("si", "sì", "yes", "true", "1")
                    
                    print("\n🔄 Ripresa esecuzione grafo...")
                    risposta = graph.invoke(Command(resume=esito_bool), config=config)
                    if "risposta" in risposta and risposta["risposta"]:
                        print(f"\n🤖 Assistente:\n{risposta['risposta']}\n")
                    continue

            # Prompt utente standard
            user_input = input(f"👤 [{telefono_corrente} @ {thread_id}] > ").strip()

            if not user_input:
                continue

            # --- GESTIONE COMANDI SPECIALI ---
            cmd = user_input.lower()

            if cmd in ("esci", "exit", "quit"):
                print("\n👋 Grazie per aver contattato la Trattoria Nonna Rosa. Arrivederci!")
                break

            elif cmd in ("nuovo", "reset"):
                session_count += 1
                thread_id = f"sessione_{telefono_corrente}_{session_count}"
                config = {"configurable": {"thread_id": thread_id}}
                print(f"\n🔄 Nuova sessione avviata! Nuovo Thread ID: '{thread_id}'")
                print("💡 Nota: Le allergie e preferenze sono mantenute nello Store cross-conversazione.\n")
                continue

            elif cmd == "utente":
                nuovo_tel = input("📱 Inserisci il nuovo numero di telefono: ").strip()
                if nuovo_tel:
                    telefono_corrente = nuovo_tel
                    session_count = 1
                    thread_id = f"sessione_{telefono_corrente}_{session_count}"
                    config = {"configurable": {"thread_id": thread_id}}
                    print(f"\n👤 Utente aggiornato: {telefono_corrente} (Thread ID: {thread_id})\n")
                else:
                    print("⚠️ Numero non valido. Operazione annullata.\n")
                continue

            elif cmd == "stato":
                print("\n" + "-" * 40)
                print(f"📌 STATO CORRENTE:")
                print(f"   • Telefono Utente : {telefono_corrente}")
                print(f"   • Thread ID Checkpoint: {thread_id}")
                
                # Recupero dati memorizzati nello Store per l'utente corrente
                item = store.get(("clienti", "allergie"), telefono_corrente)
                if item and item.value:
                    print(f"   • Memoria Store   : {item.value}")
                else:
                    print("   • Memoria Store   : Nessun dato/allergia salvato per questo numero.")
                print("-" * 40 + "\n")
                continue

            elif cmd in ("help", "aiuto"):
                print("\n📖 COMANDI DISPONIBILI:")
                print("  • 'nuovo'  : Genera un nuovo ID thread per simularne il riavvio o una nuova chat.")
                print("  • 'utente' : Cambia il numero di telefono (utile per testare clienti differenti).")
                print("  • 'stato'  : Mostra il Thread ID attivo e i dati salvati nello Store per il cliente.")
                print("  • 'esci'   : Termina l'esecuzione del chatbot.\n")
                continue

            # --- ESECUZIONE NORMALE DEL GRAFO ---
            # Si iniettano domanda e telefono nello stato
            input_data = {
                "domanda": user_input,
                "telefono": telefono_corrente
            }

            res = graph.invoke(input_data, config=config)
            
            # Stampa della risposta se non è già stata gestita da un eventuale interrupt
            if "risposta" in res and res["risposta"]:
                print(f"\n🤖 Assistente:\n{res['risposta']}\n")

        except KeyboardInterrupt:
            print("\n\n👋 Interruzione rilevata. Chiusura del chatbot...")
            break
        except Exception as e:
            print(f"\n❌ Si è verificato un errore: {e}\n")

    # Chiusura pulita delle risorse
    sqlite_conn.close()


if __name__ == "__main__":
    avvia_chatbot()