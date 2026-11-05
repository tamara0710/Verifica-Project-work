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
PATH_CHECKPOINTS = BASE_DIR / "data" / "checkpoints.sqlite"

da_pdf_a_md()
crea_tabelle_complete()
retriever = carica_md_in_vectorstore_con_ParentDocumentRetriever()

def carica_info_cag() -> str:
    if PATH_INFO.exists():
        return PATH_INFO.read_text(encoding="utf-8")
    print("⚠️ WARNING: info_trattoria.md non trovato.")
    return ""

CONTESTO_CAG = carica_info_cag()

# Configurazione checkpointer SQLite (Facoltativo +5 pt: la chat riprende dopo il riavvio)
PATH_CHECKPOINTS.parent.mkdir(parents=True, exist_ok=True)
sqlite_conn = sqlite3.connect(str(PATH_CHECKPOINTS), check_same_thread=False)
checkpointer = SqliteSaver(sqlite_conn)
checkpointer.setup()

# Configurazione Store (Facoltativo +5 pt: allergie del cliente ricordate tra conversazioni diverse)
store = InMemoryStore()


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
            pass
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

    prompt = (
        "Sei l'assistente virtuale della Trattoria Nonna Rosa per la gestione delle prenotazioni.\n\n"
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

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("      VERIFICA COMPLETA TRACCIA B (OBBLIGATORI + FACOLTATIVI)")
    print("=" * 60)

    # ---------------------------------------------------------
    # TEST 1: PRENOTAZIONE CON INTERRUPT & SQLITESAVER (Obbligatori + Facoltativo SqliteSaver)
    # ---------------------------------------------------------
    # config_1 = {"configurable": {"thread_id": "sessione_prenotazione_1"}}
    # domanda_1 = "Ciao, mi chiamo Cristiano Ronaldo, voglio prenotare un tavolo per 5 persone per il 2026-11-12 a pranzo (turno 1), il mio numero è 347557777, tra noi c'è una persona celiaca"
    # print(f"\n[TEST 1] PRENOTAZIONE CON INTERRUPT & MEMORIZZAZIONE ALLERGIE")
    # print(f"Domanda: \"{domanda_1}\"")
    
    # r1 = graph.invoke({"domanda": domanda_1}, config=config_1)
    # stato_1 = graph.get_state(config_1)
    
    # if stato_1.next:
    #     dati_interrupt = stato_1.tasks[0].interrupts[0].value
    #     print("\n" + "="*50)
    #     print("⏸️  INTERRUPT ATTIVATO (Richiesta conferma)")
    #     print("="*50)
    #     print(f"📋 Dettagli: {dati_interrupt}")
    #     print("\n✅ Risposta utente all'interrupt: 'True' (Confermo)")
    #     print("-" * 50)
        
    #     r1 = graph.invoke(Command(resume=True), config=config_1)

    # print(f"\n💬 Risposta finale Test 1:\n{r1['risposta']}\n")

    # ---------------------------------------------------------
    # TEST 2: STORE CROSS-CONVERSAZIONE (Facoltativo +5 pt)
    # Lo stesso cliente fa una nuova domanda in un Thread/Sessione DIVERSA.
    # Il grafo ricorderà che è celiaco grazie allo Store!
    # ---------------------------------------------------------
    # config_2 = {"configurable": {"thread_id": "sessione_nuova_thread_2"}}
    # domanda_2 = "Ciao sono Cristiano Ronaldo (tel 347557777), quali primi mi consigliate?"
    # print("\n" + "=" * 60)
    # print("[TEST 2] MEMORIA STORE CROSS-CONVERSAZIONE (Nuovo Thread)")
    # print(f"Domanda: \"{domanda_2}\"")
    # r2 = graph.invoke({"domanda": domanda_2}, config=config_2)
    # print(f"\n💬 Risposta Test 2 (con memoria allergie dallo Store):\n{r2['risposta']}\n")

    # ---------------------------------------------------------
    # TEST 3: REGOLA ACCONTO GRUPPI > 8 PERSONE (Facoltativo +5 pt)
    # ---------------------------------------------------------
    # config_3 = {"configurable": {"thread_id": "sessione_acconto_gruppi"}}
    # domanda_3 = "Vorrei informazioni sulla caparra e l'acconto per i gruppi di 10 persone"
    # print("\n" + "=" * 60)
    # print("[TEST 3] REGOLA ACCONTO GRUPPI > 8 PERSONE (da PDF)")
    # print(f"Domanda: \"{domanda_3}\"")
    # r3 = graph.invoke({"domanda": domanda_3}, config=config_3)
    # print(f"\n💬 Risposta Test 3:\n{r3['risposta']}\n")

    # ---------------------------------------------------------
    # TEST : 
    # ---------------------------------------------------------
    # config_test = {"configurable": {"thread_id": "test_dolci_vegani"}}
    # domanda_test = "Quali dolci sono vegani?"

    # print(f"Domanda: {domanda_test}")
    # risultato = graph.invoke({"domanda": domanda_test}, config=config_test)
    # print(f"Risposta: {risultato['risposta']}")
    
    #  # ---------------------------------------------------------
    # # TEST : 
    # # ---------------------------------------------------------
    # config_test = {"configurable": {"thread_id": "test_primi_senza_glutine"}}
    # domanda_test = "Quali primi sono senza glutine?"

    # print(f"Domanda: {domanda_test}")
    # risultato = graph.invoke({"domanda": domanda_test}, config=config_test)
    # print(f"Risposta: {risultato['risposta']}")

    # ---------------------------------------------------------
    # TEST : 
    # ---------------------------------------------------------
    # config_test = {"configurable": {"thread_id": "test_piatti_vegetariani"}}
    # domanda_test = "Nel menu sono presenti piatti vegetariani?"

    # print(f"Domanda: {domanda_test}")
    # risultato = graph.invoke({"domanda": domanda_test}, config=config_test)
    # print(f"Risposta: {risultato['risposta']}")

    # ---------------------------------------------------------
    # TEST : 
    # ---------------------------------------------------------
    config_test = {"configurable": {"thread_id": "test_cani"}}
    domanda_test = "Possiamo portare un cane in trattoria?"

    print(f"Domanda: {domanda_test}")
    risultato = graph.invoke({"domanda": domanda_test}, config=config_test)
    print(f"Risposta: {risultato['risposta']}")

    # ---------------------------------------------------------
    # TEST 4: STREAMING DEI TOKEN IN OUTPUT (Facoltativo +5 pt)
    # ---------------------------------------------------------
    # config_4 = {"configurable": {"thread_id": "sessione_streaming"}}
    # domanda_4 = "Qual è il numero di telefono e l'indirizzo della trattoria?"
    # print("\n" + "=" * 60)
    # print("[TEST 4] STREAMING TOKEN IN TEMPO REALE")
    # print(f"Domanda: \"{domanda_4}\"")
    # stream_risposta(graph, {"domanda": domanda_4}, config=config_4)
    # print("\n🎉 TUTTI I TEST DELLA TRACCIA COMPLETATI CON SUCCESSO!")



    