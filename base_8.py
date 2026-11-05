"""Chatbot Trattoria "Da Nonna Rosa" (Traccia B).

Grafo custom con routing a 3 vie:
  - info_generali -> CAG (info_trattoria.md)
  - menu          -> CRAG (RAG sul PDF con valutazione dei documenti)
  - prenotazione  -> tool su SQLite (verifica_disponibilita, crea_prenotazione, annulla_o_consulta)
Memoria: checkpointer SQLite (thread_id) + Store SQLite (allergie tra conversazioni).
"""
import json
import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, Literal, Optional, TypedDict

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import GraphBubbleUp
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.store.base import BaseStore
from langgraph.store.sqlite import SqliteStore
from langgraph.types import Command
from pydantic import BaseModel, Field

from config import get_llm
from src.database.connection import crea_tabelle_complete
from src.database.repositories import (
    annulla_o_consulta,
    crea_prenotazione,
    elenco_turni,
    normalizza_telefono,
    verifica_disponibilita,
)
from src.utils import carica_md_in_vectorstore_con_ParentDocumentRetriever, da_pdf_a_md

# ==========================================================
# 1. INIZIALIZZAZIONE: DB, STORE, CHECKPOINTER, RAG, CAG
# ==========================================================
BASE_DIR = Path(__file__).parent
PATH_INFO = BASE_DIR / "data" / "info_trattoria.md"
PATH_STORE = BASE_DIR / "src" / "database" / "store.sqlite"
PATH_CHECKPOINTS = BASE_DIR / "src" / "database" / "checkpoints.sqlite"
PATH_STORE.parent.mkdir(parents=True, exist_ok=True)

crea_tabelle_complete()   # idempotente: crea/aggiorna tavoli, turni, menù, prenotazioni
# file del database che le prenotazioni usano davvero (è quello accanto al modulo connection)
DB_IN_USO = Path(crea_tabelle_complete.__globals__["DB_PATH"])
print(f"📂 Database prenotazioni in uso: {DB_IN_USO}")

store_conn = sqlite3.connect(str(PATH_STORE), check_same_thread=False, timeout=20.0, isolation_level=None)
store_conn.execute("PRAGMA journal_mode=WAL;")
store = SqliteStore(store_conn)
store.setup()

sqlite_conn = sqlite3.connect(str(PATH_CHECKPOINTS), check_same_thread=False, timeout=20.0, isolation_level=None)
sqlite_conn.execute("PRAGMA journal_mode=WAL;")
checkpointer = SqliteSaver(sqlite_conn)
checkpointer.setup()

try:
    da_pdf_a_md()
except Exception as e:   # se il PDF non si converte si usa l'MD già presente
    print(f"⚠️ Conversione PDF -> MD non riuscita ({e}): uso il file MD esistente.")
retriever = carica_md_in_vectorstore_con_ParentDocumentRetriever()


def carica_info_cag() -> str:
    for percorso in (PATH_INFO, BASE_DIR / "info_trattoria.md"):
        if percorso.exists():
            return percorso.read_text(encoding="utf-8")
    print("⚠️ WARNING: info_trattoria.md non trovato (cercato in data/ e nella cartella del progetto).")
    return ""


CONTESTO_CAG = carica_info_cag()

# ==========================================================
# 2. TOOL (cerca_menu + i 3 tool del database)
# ==========================================================
@tool
def cerca_menu(query: str) -> str:
    """Cerca nel menu e nel regolamento della Trattoria Nonna Rosa (PDF): piatti, ingredienti, prezzi, allergeni, orari e regole su acconti e cani."""
    docs = retriever.invoke(query)
    if not docs:
        return "Nessuna informazione trovata nel menu della trattoria."
    return "\n\n".join(f"- {d.page_content}" for d in docs)


tools_db = [cerca_menu, verifica_disponibilita, crea_prenotazione, annulla_o_consulta]
tools_map = {t.name: t for t in tools_db}
tools_map["salva_prenotazione"] = crea_prenotazione
tools_map["controlla_disponibilita_tavoli"] = verifica_disponibilita

ALIAS_ARGOMENTI = {
    "data_prenotazione": "data", "numero_persone": "persone", "num_persone": "persone",
    "id_turno": "turno_id", "turno": "turno_id", "id_prenotazione": "prenotazione_id",
    "codice": "prenotazione_id", "codice_prenotazione": "prenotazione_id", "telefono_cliente": "telefono",
}
NOMI_CAMPI = {"nome": "il nome", "telefono": "il numero di telefono", "data": "la data",
              "turno_id": "il turno (pranzo 12:30 o 14:00, cena 19:30 o 21:30)",
              "persone": "il numero di persone", "azione": "l'azione (consulta o annulla)"}


def sanifica_argomenti_tool(nome_tool: str, argomenti: dict, telefono_sessione: Optional[str] = None) -> dict:
    """Pulisce gli argomenti scelti dall'LLM: alias, campi inventati, tipi, telefono di sessione."""
    args = dict(argomenti or {})
    for vecchio, nuovo in ALIAS_ARGOMENTI.items():
        if vecchio in args and nuovo not in args:
            args[nuovo] = args.pop(vecchio)

    tool_obj = tools_map.get(nome_tool)
    if tool_obj is not None:                       # tolgo i parametri che il tool non ha
        args = {k: v for k, v in args.items() if k in tool_obj.args}
    args = {k: v for k, v in args.items() if v is not None}   # None = non fornito

    for k in ("nome", "telefono", "data", "email", "note", "azione", "query"):
        if k in args:
            args[k] = str(args[k]).strip()
    for k in ("turno_id", "persone", "prenotazione_id"):
        if k in args:
            try:
                args[k] = int(float(str(args[k]).strip()))
            except ValueError:
                pass                               # resta com'è: il tool segnalerà l'errore
    if "celiachia" in args and not isinstance(args["celiachia"], bool):
        args["celiachia"] = str(args["celiachia"]).strip().lower() in ("true", "1", "si", "sì", "yes")

    # identità: consulta/annulla valgono SEMPRE per il telefono della sessione
    if telefono_sessione:
        if nome_tool == "annulla_o_consulta":
            args["telefono"] = telefono_sessione
        elif nome_tool in ("crea_prenotazione",) and not args.get("telefono"):
            args["telefono"] = telefono_sessione
    return args


# ==========================================================
# 3. STATO, LLM, CLASSIFICATORE
# ==========================================================
llm = get_llm()


class State(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    domanda: str
    telefono: Optional[str]
    intento: Optional[str]
    ultimo_intento: Optional[str]
    query: Optional[str]
    documenti: Optional[list[str]]
    pertinente: Optional[bool]
    tentativi: Optional[int]
    risposta: Optional[str]
    esito_cag: Optional[bool]
    tool_calls: Optional[list]


class ClassificaIntento(BaseModel):
    intento: Literal["info_generali", "menu", "prenotazione"] = Field(
        description=(
            "- 'info_generali': contatti (telefono, email), indirizzo, Wi-Fi, parcheggio, accessibilità disabili.\n"
            "- 'menu': piatti, prezzi, allergeni, celiachia/glutine, vegani, ORARI, turni, CANI/ANIMALI, coperto, acconti e disdette (regole).\n"
            "- 'prenotazione': prenotare un tavolo, verificare disponibilità, consultare o annullare prenotazioni, "
            "oppure fornire dati richiesti per una prenotazione in corso (nome, telefono, data, turno, persone)."
        )
    )


class Giudizio(BaseModel):
    pertinente: bool = Field(description="I documenti estratti consentono di rispondere alla domanda?")


classifier = llm.with_structured_output(ClassificaIntento)
grader = get_llm().with_structured_output(Giudizio)

GIORNI = ["lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica"]


def testo_cronologia(messaggi: list, n: int = 6) -> str:
    righe = []
    for m in messaggi[-n:]:
        ruolo = "Cliente" if isinstance(m, HumanMessage) else "Assistente"
        righe.append(f"{ruolo}: {str(m.content)[:400]}")
    return "\n".join(righe)


def intento_euristico(testo: str) -> str:
    """Piano B se il classificatore LLM fallisce."""
    t = testo.lower()
    if re.search(r"prenot|disdett|annull|cancell|disponibil|tavolo|\bposti?\b", t) and not re.search(r"acconto|caparra|regol", t):
        return "prenotazione"
    if re.search(r"telefon|indirizzo|wi-?fi|parcheggi|e-?mail|accessib|dove si trova", t):
        return "info_generali"
    return "menu"


def aggiorna_memoria_allergie(st: Optional[BaseStore], telefono: Optional[str], testo: str) -> None:
    """Salva nello Store le allergie dette in prima persona (es. 'sono celiaco', 'sono allergico alle noci')."""
    if not st or not telefono:
        return
    low = testo.lower()
    trovate = []
    if re.search(r"\b(sono|sia|ho la|soffro di)\s+(celiac\w*|celiachia)|intolleran\w+ al glutine", low):
        trovate.append("Celiachia (senza glutine)")
    for m in re.finditer(r"allergic[oa]\s+(?:a|ai|alle|alla|al|agli|all')\s*([a-zàèéìòù ]{3,40})", low):
        trovate.append(f"Allergia: {m.group(1).strip()}")
    if trovate:
        salva_allergia(st, telefono, trovate)


def salva_allergia(st: BaseStore, telefono: str, nuove: list, nome: str = "") -> None:
    item = st.get(("clienti", "allergie"), telefono)
    esistenti = [x.strip() for x in (item.value.get("allergie", "").split(";") if item and item.value else []) if x.strip()]
    for n in nuove:
        if n not in esistenti:
            esistenti.append(n)
    st.put(("clienti", "allergie"), telefono,
           {"nome": nome or (item.value.get("nome", "") if item and item.value else ""), "allergie": "; ".join(esistenti)})
    print(f"💾 [STORE] Allergie per {telefono}: {'; '.join(esistenti)}")


def leggi_allergie(st: Optional[BaseStore], telefono: Optional[str]) -> str:
    if st and telefono:
        item = st.get(("clienti", "allergie"), telefono)
        if item and item.value:
            return item.value.get("allergie", "")
    return ""


def trova_telefono(state: State) -> Optional[str]:
    tel = state.get("telefono")
    if tel:
        return normalizza_telefono(tel)
    m = re.search(r"(?<!\d)(3\d{7,10})(?!\d)", state.get("domanda", ""))
    return m.group(1) if m else None


# ==========================================================
# 4. NODI
# ==========================================================
def classifica_intento(state: State, *, store: Optional[BaseStore] = None) -> dict:
    domanda = state["domanda"]
    telefono = trova_telefono(state)
    aggiorna_memoria_allergie(store, telefono, domanda)

    storico = state.get("messages", [])
    contesto = testo_cronologia(storico, 4)
    system_prompt = (
        "Sei un classificatore di intenti per l'assistente della Trattoria Nonna Rosa.\n"
        "1. 'info_generali': solo contatti (telefono, email), indirizzo, Wi-Fi, parcheggio, accessibilità.\n"
        "2. 'menu': piatti, ingredienti, allergeni, celiachia, vegani, ORARI DI APERTURA, turni, CANI/ANIMALI, "
        "coperto, regole su acconti/caparra e disdette.\n"
        "3. 'prenotazione': l'utente vuole PRENOTARE, chiede se c'è posto/disponibilità, vuole CONSULTARE o ANNULLARE "
        "una prenotazione, OPPURE sta rispondendo con i dati richiesti per una prenotazione già avviata "
        "(nome, telefono, data, turno, numero di persone)."
    )
    try:
        res = classifier.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=(f"Conversazione recente:\n{contesto}\n\n" if contesto else "") +
                                 f"Ultimo messaggio dell'utente: {domanda}"),
        ])
        intento = res.intento
    except Exception as e:
        intento = intento_euristico(domanda)
        print(f"⚠️ Classificatore non riuscito ({e}): uso l'euristica -> {intento}")

    # Se si stava prenotando e il messaggio contiene numeri/date, sta fornendo dati.
    if state.get("ultimo_intento") == "prenotazione" and intento != "prenotazione" and re.search(r"\d", domanda) \
            and not re.search(r"acconto|caparra|prezzo|€|euro|orari", domanda.lower()):
        intento = "prenotazione"

    print(f"--- [ROUTING] Intento rilevato: {intento} ---")
    # reset dei campi dei turni precedenti (il checkpointer li conserva!)
    return {
        "intento": intento, "ultimo_intento": intento, "telefono": telefono,
        "tentativi": 0, "query": None, "documenti": [], "pertinente": None,
        "esito_cag": None, "risposta": None, "tool_calls": [],
        "messages": [HumanMessage(content=domanda)],
    }


def gestisci_cag(state: State) -> dict:
    print("--- [NODO] ESECUZIONE CAG ---")
    if not CONTESTO_CAG.strip():
        return {"esito_cag": False}
    prompt = (
        "Sei l'assistente della Trattoria Nonna Rosa.\n"
        "Rispondi usando ESCLUSIVAMENTE le informazioni generali seguenti.\n"
        "Se l'informazione NON è presente nel testo, rispondi solo: NON_TROVATO.\n\n"
        f"--- INFORMAZIONI GENERALI ---\n{CONTESTO_CAG}\n-----------------------------\n\n"
        f"Domanda: {state['domanda']}"
    )
    try:
        res = str(llm.invoke(prompt).content).strip()
    except Exception as e:
        print(f"⚠️ CAG non riuscito ({e}): passo al RAG")
        return {"esito_cag": False}
    if "NON_TROVATO" in res.upper() or not res:
        return {"esito_cag": False}
    return {"risposta": res, "esito_cag": True, "messages": [AIMessage(content=res)]}


def recupera(state: State) -> dict:
    print("--- [NODO] ESECUZIONE RAG (recupera con cerca_menu) ---")
    q = state.get("query") or state["domanda"]
    print(f"📥 Query di ricerca: '{q}'")
    try:
        docs = retriever.invoke(q)
    except Exception as e:
        print(f"⚠️ Errore nel retriever: {e}")
        docs = []
    print(f"📄 Documenti trovati: {len(docs)} chunk")
    return {"documenti": [d.page_content for d in docs]}


def valuta(state: State) -> dict:
    documenti = state.get("documenti") or []
    if not documenti:
        return {"pertinente": False}
    prompt = (
        f"Domanda: {state['domanda']}\n"
        f"Documenti tratti dal menu: {documenti}\n\n"
        "Valuta se i documenti contengono informazioni utili per rispondere alla domanda."
    )
    try:
        g = grader.invoke(prompt)
        pertinente = bool(g.pertinente)
    except Exception as e:
        print(f"⚠️ Valutazione non riuscita ({e}): considero i documenti pertinenti")
        pertinente = True
    print(f"⚖️ Valutazione Pertinenza Documenti: {pertinente}")
    return {"pertinente": pertinente}


def riscrivi(state: State) -> dict:
    try:
        q = str(llm.invoke(
            "Riformula la domanda come query di ricerca breve (max 12 parole) su piatti, ingredienti, allergeni o "
            f"regole del menu. Rispondi SOLO con la query.\nDomanda: {state['domanda']}"
        ).content)
        q = q.strip().splitlines()[0].strip(" \"'") if q.strip() else state["domanda"]
    except Exception:
        q = state["domanda"]
    print(f"🔄 Riformulazione Query RAG: '{q}'")
    return {"query": q[:200], "tentativi": (state.get("tentativi") or 0) + 1}


def risposta_dal_menu(domanda: str, documenti: list, telefono: Optional[str], st: Optional[BaseStore]) -> str:
    ctx = "\n\n".join(documenti)
    memoria = leggi_allergie(st, telefono)
    system_prompt = (
        "Sei l'assistente virtuale della Trattoria Nonna Rosa.\n"
        "Rispondi basandoti ESCLUSIVAMENTE sul testo del menu e del regolamento forniti nel contesto.\n\n"
        "REGOLE:\n"
        "1. Colonna ALLERGENI: se contiene un allergene, il piatto lo contiene.\n"
        "2. Per celiaci sono idonei solo i piatti '(senza glutine)' o senza glutine tra gli allergeni.\n"
        "3. Acconto: gruppi oltre 8 persone, 10 € a persona entro 48 ore. Disdetta gratuita fino a 24 ore prima.\n"
        "4. Riporta tutti gli orari e le cifre richieste, senza omissioni.\n"
        "5. DIVIETO DI INVENZIONE: se l'informazione non è nel contesto, rispondi che non è presente nel menu."
    )
    if memoria:
        system_prompt += f"\n\nIl cliente ha segnalato in precedenza: '{memoria}'. Tienine conto nella risposta."
    r = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"--- CONTESTO DAL MENU ---\n{ctx}\n-------------------------\n\nDomanda: {domanda}"),
    ])
    return str(r.content).strip()


def genera(state: State, *, store: Optional[BaseStore] = None) -> dict:
    print("--- [NODO] ESECUZIONE RAG (genera) ---")
    documenti = state.get("documenti") or []
    if not documenti:
        testo = "Mi dispiace, non ho trovato informazioni rilevanti nel menu per rispondere alla tua richiesta."
    else:
        try:
            testo = risposta_dal_menu(state["domanda"], documenti, trova_telefono(state), store)
        except Exception as e:
            testo = f"Mi dispiace, c'è stato un problema nel generare la risposta ({e})."
    return {"risposta": testo, "messages": [AIMessage(content=testo)]}


def non_so(state: State) -> dict:
    testo = "Mi dispiace, non ho trovato questa informazione nel menu della trattoria."
    return {"risposta": testo, "messages": [AIMessage(content=testo)]}


def contesto_data() -> str:
    oggi = datetime.now()
    righe = [f"OGGI è {GIORNI[oggi.weekday()]} {oggi:%Y-%m-%d}. Calendario dei prossimi giorni:"]
    for i in range(0, 15):
        g = oggi + timedelta(days=i)
        righe.append(f"- {GIORNI[g.weekday()]} {g:%Y-%m-%d}" + (" (CHIUSO)" if g.weekday() == 0 else ""))
    return "\n".join(righe)


def decidi_tool(state: State, *, store: Optional[BaseStore] = None) -> dict:
    """L'LLM decide quali tool chiamare (NON li esegue: così, alla ripresa dopo
    un interrupt, l'LLM non viene richiamato e le scelte restano identiche)."""
    print("--- [NODO] DECISIONE TOOL (prenotazione) ---")
    telefono = state.get("telefono")
    memoria = leggi_allergie(store, telefono)
    turni = ", ".join(f"{t['id']} = {t['nome']} {t['orario']}" for t in elenco_turni())

    prompt = (
        "Sei l'assistente della Trattoria Nonna Rosa per le prenotazioni. Rispondi in italiano.\n\n"
        f"{contesto_data()}\n\n"
        f"TURNI (turno_id): {turni}.\n"
        f"Il telefono del cliente in questa chat è: {telefono or 'non noto'}.\n\n"
        "SCELTA DEGLI STRUMENTI:\n"
        "1. PRENOTARE: se hai nome, telefono, data (YYYY-MM-DD), turno_id e numero di persone chiama 'crea_prenotazione' "
        "(controlla da solo la disponibilità e chiede conferma al cliente). Usa i dati di TUTTA la conversazione.\n"
        "2. 'verifica_disponibilita' solo per domande informative (es. 'c'è posto per 4 domani?').\n"
        "3. 'annulla_o_consulta' per vedere o annullare prenotazioni (azione 'consulta' o 'annulla').\n"
        "4. 'cerca_menu' per dubbi su piatti, allergeni o regole.\n"
        "Converti date relative ('domani', 'sabato') usando il calendario. Il tavolo si prenota fino a 30 giorni prima, "
        "da 1 a 12 persone, lunedì chiuso. Oltre 8 persone serve un acconto di 10 € a persona.\n"
        "Se mancano dati obbligatori NON chiamare strumenti: chiedili con gentilezza (una sola domanda breve). "
        "Non inventare mai dati."
    )
    if memoria:
        prompt += f"\n\nIl cliente ha segnalato in precedenza: '{memoria}' (se celiaco imposta celiachia=True)."

    storico = [m for m in state.get("messages", []) if isinstance(m, (HumanMessage, AIMessage))][-8:]
    try:
        msg = llm.bind_tools(tools_db).invoke([SystemMessage(content=prompt)] + storico)
    except Exception as e:
        testo = f"Mi dispiace, non riesco a elaborare la richiesta in questo momento ({e})."
        return {"risposta": testo, "tool_calls": [], "messages": [AIMessage(content=testo)]}

    chiamate = []
    for call in (msg.tool_calls or []):
        args = sanifica_argomenti_tool(call["name"], call.get("args", {}), telefono)
        item = {"name": call["name"], "args": args}
        if item not in chiamate:
            chiamate.append(item)

    if chiamate:
        print(f"🛠️  Tool scelti dall'LLM: {[c['name'] for c in chiamate]}")
        return {"tool_calls": chiamate}

    testo = str(msg.content).strip() or "Potresti ripetere la richiesta con nome, data, turno e numero di persone?"
    return {"risposta": testo, "tool_calls": [], "messages": [AIMessage(content=testo)]}


def messaggio_errore_tool(e: Exception) -> str:
    righe = e.errors() if hasattr(e, "errors") else []
    mancanti = [NOMI_CAMPI.get(str(r["loc"][0]), str(r["loc"][0])) for r in righe if r.get("type") == "missing"]
    if mancanti:
        return "Per procedere mi servono ancora: " + ", ".join(mancanti) + "."
    if righe:
        return "Alcuni dati non sono validi: " + "; ".join(f"{r['loc'][0]} ({r['msg']})" for r in righe) + "."
    return f"Errore: {e}"


def formatta_esito(nome_tool: str, esito) -> str:
    esito = str(esito)
    if nome_tool in ("crea_prenotazione", "salva_prenotazione"):
        try:
            d = json.loads(esito)
        except ValueError:
            return esito
        if d.get("esito") == "ok":
            testo = (f"✅ Prenotazione confermata! Codice {d['codice']}: {d['nome']}, {d['persone']} persone "
                     f"il {d['data']}, turno {d['turno']} ({d['orario']}), tavolo {d['tavolo']} (zona {d['zona']}).")
            if d.get("celiachia"):
                testo += "\nCeliachia segnalata: pasta senza glutine con cottura separata (supplemento 2 €)."
            if d.get("acconto_euro", 0) > 0:
                testo += f"\n{d['info_acconto']}"
            testo += "\nIl tavolo viene tenuto 15 minuti dopo l'orario del turno."
            return testo + "\n\n🧾 RICEVUTA DI PRENOTAZIONE (JSON):\n```json\n" + json.dumps(d, indent=2, ensure_ascii=False) + "\n```"
        if d.get("esito") == "non_confermata":
            return "Va bene, non ho effettuato la prenotazione. Posso aiutarti in altro modo?"
    return esito


def esegui_tool(state: State, *, store: Optional[BaseStore] = None) -> dict:
    """Esegue i tool scelti. È qui che scatta l'interrupt di conferma."""
    print("--- [NODO] ESECUZIONE TOOL ---")
    chiamate = state.get("tool_calls") or []
    # una sola operazione di scrittura per volta (evita doppie prenotazioni/annullamenti)
    scritture = [c for c in chiamate if c["name"] in ("crea_prenotazione", "salva_prenotazione")] or \
                [c for c in chiamate if c["name"] == "annulla_o_consulta" and str(c["args"].get("azione", "")).lower().startswith(("annull", "cancell", "disd"))]
    selezionate = scritture[:1] if scritture else chiamate[:3]

    esiti = []
    for call in selezionate:
        nome, args = call["name"], call["args"]
        print(f"🔧 Tool: {nome}  📥 {args}")
        tool_func = tools_map.get(nome)
        if tool_func is None:
            esito = f"Errore: tool '{nome}' non riconosciuto."
        else:
            try:
                esito = tool_func.invoke(args)
            except GraphBubbleUp:      # l'interrupt NON è un errore: deve risalire al grafo
                raise
            except Exception as e:
                esito = messaggio_errore_tool(e)
        print(f"📤 {str(esito)[:300]}")

        if nome == "cerca_menu":
            try:
                esito = risposta_dal_menu(state["domanda"], [str(esito)], state.get("telefono"), store)
            except Exception as e:
                esito = f"Mi dispiace, non riesco a rispondere ora ({e})."
        if nome in ("crea_prenotazione", "salva_prenotazione") and args.get("celiachia") and store and args.get("telefono"):
            salva_allergia(store, normalizza_telefono(args["telefono"]), ["Celiachia (senza glutine)"], args.get("nome", ""))
        esiti.append(formatta_esito(nome, esito))

    testo = "\n\n".join(esiti) if esiti else "Non ho potuto eseguire l'operazione richiesta."
    return {"risposta": testo, "tool_calls": [], "messages": [AIMessage(content=testo)]}


# ==========================================================
# 5. GRAFO
# ==========================================================
def instrada_intento(state: State) -> str:
    return {"prenotazione": "decidi_tool", "menu": "recupera"}.get(state.get("intento"), "gestisci_cag")


def instrada_dopo_cag(state: State) -> str:
    return END if state.get("esito_cag") else "recupera"


def instrada_valutazione(state: State) -> str:
    if state.get("pertinente"):
        return "genera"
    if (state.get("tentativi") or 0) < 2:
        return "riscrivi"
    # dopo 2 riscritture: se ci sono documenti, genera (il prompt vieta di inventare); altrimenti non_so
    return "genera" if state.get("documenti") else "non_so"


def instrada_dopo_decisione(state: State) -> str:
    return "esegui_tool" if state.get("tool_calls") else END


builder = StateGraph(State)
for nome, nodo in [("classifica_intento", classifica_intento), ("gestisci_cag", gestisci_cag),
                   ("recupera", recupera), ("valuta", valuta), ("riscrivi", riscrivi), ("genera", genera),
                   ("non_so", non_so), ("decidi_tool", decidi_tool), ("esegui_tool", esegui_tool)]:
    builder.add_node(nome, nodo)

builder.add_edge(START, "classifica_intento")
builder.add_conditional_edges("classifica_intento", instrada_intento, ["gestisci_cag", "recupera", "decidi_tool"])
builder.add_conditional_edges("gestisci_cag", instrada_dopo_cag, [END, "recupera"])
builder.add_edge("recupera", "valuta")
builder.add_conditional_edges("valuta", instrada_valutazione, ["genera", "riscrivi", "non_so"])
builder.add_edge("riscrivi", "recupera")
builder.add_edge("genera", END)
builder.add_edge("non_so", END)
builder.add_conditional_edges("decidi_tool", instrada_dopo_decisione, ["esegui_tool", END])
builder.add_edge("esegui_tool", END)

graph = builder.compile(checkpointer=checkpointer, store=store)


# ==========================================================
# 6. STREAMING (facoltativo)
# ==========================================================
def stream_risposta(graph_instance, input_data: dict, config: dict) -> str:
    print("🌊 [STREAMING OUTPUT]: ", end="", flush=True)
    testo = ""
    for msg, _meta in graph_instance.stream(input_data, config=config, stream_mode="messages"):
        if isinstance(msg, AIMessageChunk) and msg.content:
            print(msg.content, end="", flush=True)
            testo += msg.content
    print("\n" + "-" * 50)
    return testo


# ==========================================================
# 7. CHAT DA TERMINALE
# ==========================================================
PAROLE_SI = {"si", "sì", "s", "yes", "y", "ok", "okay", "certo", "confermo", "conferma", "confermato",
             "va bene", "procedi", "esatto", "perfetto", "giusto"}
PAROLE_NO = {"no", "n", "nope", "annulla", "annullo", "stop", "lascia stare", "non confermo", "sbagliato", "errato"}


def interpreta_conferma(testo: str) -> Optional[bool]:
    """True = conferma, False = rifiuto, None = non si capisce (si richiede).
    Accetta anche frasi come 'ok conferma', 'sì confermo', 'si grazie', 'no annulla'."""
    t = re.sub(r"[^\wàèéìòù ]", " ", testo.lower()).strip()
    parole = t.split()
    if not parole:
        return None
    if t in PAROLE_NO or parole[0] in ("no", "non", "annulla", "stop") or any(p in ("no", "annulla", "annullo") for p in parole):
        return False
    if t in PAROLE_SI or parole[0] in PAROLE_SI or any(p in ("confermo", "conferma", "confermato") for p in parole):
        return True
    return None


def chiedi_conferma() -> bool:
    while True:
        esito = interpreta_conferma(input("\n👉 Scrivi 'si' per confermare o 'no' per annullare: "))
        if esito is not None:
            return esito
        print("   Non ho capito: rispondi 'si' oppure 'no'.")


def mostra_ultime_prenotazioni(n: int = 8) -> None:
    conn = sqlite3.connect(str(DB_IN_USO), timeout=20.0)
    try:
        righe = conn.execute("""
            SELECT p.id, c.nome, c.telefono, p.data, t.nome || ' ' || t.orario, p.persone, p.stato
            FROM prenotazioni p JOIN clienti c ON c.id = p.cliente_id JOIN turni t ON t.id = p.turno_id
            ORDER BY p.id DESC LIMIT ?""", (n,)).fetchall()
    finally:
        conn.close()
    print(f"\n📒 Ultime prenotazioni nel database ({DB_IN_USO}):")
    for r in righe:
        print(f"   #{r[0]} | {r[1]} | tel {r[2]} | {r[3]} {r[4]} | {r[5]} pers. | {r[6]}")
    if not righe:
        print("   (nessuna prenotazione)")
    print()


def mostra_richiesta_conferma(dati) -> None:
    print("\n" + "=" * 55)
    print("⚠️  RICHIESTA DI CONFERMA")
    print("=" * 55)
    if isinstance(dati, dict):
        for k, v in dati.items():
            if k != "domanda":
                print(f"   • {k}: {v}")
        print(f"\n{dati.get('domanda', 'Confermi?')}")
    else:
        print(dati)


def stampa_risposta(res) -> None:
    if res and res.get("risposta"):
        print(f"\n🤖 Assistente:\n{res['risposta']}\n")


def avvia_chatbot():
    print("\n" + "=" * 65)
    print(" 🍷 BENVENUTO NEL CHATBOT DI TRATTORIA NONNA ROSA")
    print("=" * 65)
    print(" Comandi: help | nuovo | utente | stato | esci")
    print("=" * 65 + "\n")

    session_count = 1
    telefono_corrente = normalizza_telefono(input("📱 Inserisci il tuo numero di telefono (es. 347557777): "))
    if len(telefono_corrente) < 6:
        telefono_corrente = "347557777"
        print(f"👉 Usato numero predefinito: {telefono_corrente}")

    thread_id = f"sessione_{telefono_corrente}_{session_count}"
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        try:
            # c'è una conferma in sospeso (anche dopo un riavvio)?
            stato = graph.get_state(config)
            if stato.next and stato.tasks and getattr(stato.tasks[0], "interrupts", None):
                mostra_richiesta_conferma(stato.tasks[0].interrupts[0].value)
                stampa_risposta(graph.invoke(Command(resume=chiedi_conferma()), config=config))
                continue

            user_input = input(f"👤 [{telefono_corrente} @ {thread_id}] > ").strip()
            if not user_input:
                continue
            cmd = user_input.lower()

            if cmd in ("esci", "exit", "quit"):
                print("\n👋 Grazie per aver contattato la Trattoria Nonna Rosa. Arrivederci!")
                break
            elif cmd in ("nuovo", "reset"):
                session_count += 1
                thread_id = f"sessione_{telefono_corrente}_{session_count}"
                config = {"configurable": {"thread_id": thread_id}}
                print(f"\n🔄 Nuova sessione: '{thread_id}' (le allergie restano nello Store)\n")
                continue
            elif cmd == "utente":
                nuovo = normalizza_telefono(input("📱 Nuovo numero di telefono: "))
                if len(nuovo) >= 6:
                    telefono_corrente, session_count = nuovo, 1
                    thread_id = f"sessione_{telefono_corrente}_{session_count}"
                    config = {"configurable": {"thread_id": thread_id}}
                    print(f"\n👤 Utente: {telefono_corrente} (thread {thread_id})\n")
                else:
                    print("⚠️ Numero non valido.\n")
                continue
            elif cmd == "stato":
                item = store.get(("clienti", "allergie"), telefono_corrente)
                print(f"\n📌 Telefono: {telefono_corrente} | Thread: {thread_id}")
                print(f"   Memoria Store: {item.value if item and item.value else 'nessuna allergia salvata'}\n")
                continue
            elif cmd in ("db", "prenotazioni"):
                mostra_ultime_prenotazioni()
                continue
            elif cmd in ("help", "aiuto"):
                print("\n  nuovo = nuova chat | utente = cambia telefono | stato = mostra memoria | esci = chiudi\n")
                continue

            stampa_risposta(graph.invoke({"domanda": user_input, "telefono": telefono_corrente}, config=config))

        except KeyboardInterrupt:
            print("\n\n👋 Chiusura del chatbot...")
            break
        except Exception as e:
            print(f"\n❌ Errore: {e}\n")

    sqlite_conn.close()
    store_conn.close()


if __name__ == "__main__":
    avvia_chatbot()
