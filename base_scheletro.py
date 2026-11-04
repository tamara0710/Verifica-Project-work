"""Pattern CRAG con routing a 3 vie (CAG / RAG / DB) e guardrail anti-allucinazione severo."""
from pathlib import Path
from typing import TypedDict, Annotated, Literal, Optional
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from config import get_llm, get_embeddings
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.types import Command
from langchain_core.tools import tool

from langchain_core.documents import Document
from langchain_chroma import Chroma

from src.database.connection import crea_tabelle_complete
from src.database import repositories
from src.utils import da_pdf_a_md, carica_md_in_vectorstore, carica_md_in_vectorstore_con_ParentDocumentRetriever

from src.database.repositories import (
    verifica_disponibilita,
    crea_prenotazione,
    annulla_o_consulta
)


# 1. INIZIALIZZAZIONE DB E CAG

da_pdf_a_md()
crea_tabelle_complete()
retriever = carica_md_in_vectorstore_con_ParentDocumentRetriever()

BASE_DIR = Path(__file__).parent
PATH_INFO = BASE_DIR / "data" / "info_trattoria.md"

def carica_info_cag() -> str:
    if PATH_INFO.exists():
        return PATH_INFO.read_text(encoding="utf-8")
    print("⚠️ WARNING: info_trattoria.md non trovato.")
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

# @tool
# def salva_prenotazione(nome: str, telefono: str, data: str, turno_id: int, persone: int, email: str | None = None, note: str | None = "", celiachia: bool | None = False) -> str:
#     """Registra e salva una prenotazione nel database della trattoria."""
#     if celiachia is None:
#         celiachia = False
#     if note is None:
#         note = ""
#     if email is None:
#         email = ""
#     #tavoli = repositories.verifica_disponibilita(data, turno_id, persone)
#     tavoli = repositories.verifica_disponibilita.invoke({"data": data, "turno_id": int(turno_id), "persone": int(persone)})
#     if not tavoli:
#         return "Impossibile prenotare: nessun tavolo libero."

#     if email is None:
#         email = ""
#     cliente_id = repositories.trova_o_crea_cliente(nome, telefono, email=email)
#     acconto = (persone * 10.0) if persone > 8 else 0.0
#     testo_note = note if note is not None else ""
#     prenotazione_id = repositories.inserisci_prenotazione(
#         cliente_id, tavoli[0][0], turno_id, data, persone, testo_note, 1 if celiachia else 0, acconto
#     )
#     return f"Prenotazione confermata con ID #{prenotazione_id} a nome di {nome} per {persone} persone."

@tool
def salva_prenotazione(
    nome: str, 
    telefono: str, 
    data: str, 
    turno_id: int, 
    persone: int, 
    email: str | None = None, 
    note: str | None = "", 
    celiachia: bool | None = False
) -> str:
    """Registra e salva una prenotazione nel database della trattoria."""
    
    # 1. Sanificazione e normalizzazione dei parametri opzionali per l'LLM
    if celiachia is None:
        celiachia = False
    if note is None:
        note = ""
    if email is None:
        email = ""

    # 2. Verifica disponibilità tramite sotto-tool LangChain (.invoke)
    risultato_tavoli = repositories.verifica_disponibilita.invoke({
        "data": data, 
        "turno_id": int(turno_id), 
        "persone": int(persone)
    })
    
    # Se il tool restituisce una stringa vuota, None o un messaggio di errore
    if not risultato_tavoli or "nessun" in str(risultato_tavoli).lower():
        return "Impossibile prenotare: nessun tavolo libero."

    # 3. Gestione del cliente (creazione o recupero ID)
    # Passa 'conn' come primo parametro posizionale
    cliente_id = repositories.trova_o_crea_cliente(conn, nome=nome, telefono=telefono, email=email)
    
    # 4. Calcolo acconto e logica del tavolo scelto
    acconto = (int(persone) * 10.0) if int(persone) > 8 else 0.0
    
    # Estraggo l'ID del tavolo: gestiamo sia una lista reale [[id, ...]] sia una stringa
    # Se 'risultato_tavoli' è una lista/tupla estraiamo l'id, altrimenti usiamo il valore direttamente
    if isinstance(risultato_tavoli, list) and len(risultato_tavoli) > 0:
        tavolo_id = risultato_tavoli[0][0] if isinstance(risultato_tavoli[0], (list, tuple)) else resultado_tavoli[0]
    else:
        # Fallback sicuro se restituisce un ID singolo o stringa numerica
        tavolo_id = risultato_tavoli

    # 5. Inserimento della prenotazione finale
    prenotazione_id = repositories.inserisci_prenotazione(
        cliente_id, 
        tavolo_id, 
        int(turno_id), 
        data, 
        int(persone), 
        note, 
        1 if celiachia else 0, 
        acconto
    )
    
    return f"Prenotazione confermata con ID #{prenotazione_id} a nome di {nome} per {persone} persone."


tools_db = [controlla_disponibilita_tavoli, salva_prenotazione]
tools_map = {t.name: t for t in tools_db}


# 3. CLASSIFICAZIONE INTENTO

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
            "- 'info_generali': contatti (telefono, email), indirizzo, Wi-Fi, parcheggio, accessibilità disabili.\n"
            "- 'menu': piatti, prezzi, allergeni, celiachia/glutine, vegani, ORARI, turni, CANI/ANIMALI DOMESTICI, coperto, acconti e disdette.\n"
            "- 'prenotazione': intenzione ESPLICITA di prenotare un tavolo o chiedere disponibilità posti."
        )
    )

class Giudizio(BaseModel):
    pertinente: bool = Field(description="I documenti estratti consentono di rispondere alla domanda?")

classifier = llm.with_structured_output(ClassificaIntento)
grader = get_llm().with_structured_output(Giudizio)


# 4. NODI DEL GRAFO

def classifica_intento(state: State) -> dict:
    system_prompt = (
        "Sei un classificatore di intenti rigido per l'assistente della Trattoria Nonna Rosa.\n\n"
        "REGOLE TASSATIVE PER LA CLASSIFICAZIONE:\n"
        "1. 'info_generali' (CAG): Solo per contatti (telefono, email), indirizzo, Wi-Fi, parcheggio convenzionato e accessibilità disabili.\n"
        "2. 'menu' (RAG): Per QUALSIASI domanda su piatti, ingredienti, allergeni, celiachia, piatti vegani, "
        "ORARI DI APERTURA, turni, CANI/ANIMALI DOMESTICI, costo coperto, acconti/caparra e disdette.\n"
        "3. 'prenotazione' (DB): SOLO ED ESCLUSIVAMENTE se l'utente esprime la VOLONTÀ ESPLICITA di PRENOTARE UN TAVOLO o chiedere disponibilità posti."
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
    print("\n🔍 RAG: Recupero documenti da Chroma Vectorstore...")
    print(f"📥 Query di ricerca: '{q}'")
    docs_estrattti = retriever.invoke(q)
    print(f"📄 Documenti trovati: {len(docs_estrattti)} chunk")
    print("-" * 50)
    return {"documenti": [d.page_content for d in docs_estrattti]}

def valuta(state: State) -> dict:
    prompt = (
        f"Domanda: {state['domanda']}\n"
        f"Documenti tratti dal menu: {state['documenti']}\n\n"
        "Valuta se i documenti contengono informazioni utili per rispondere alla domanda."
    )
    g = grader.invoke(prompt)

    print(f"⚖️ Valutazione Pertinenza Documenti: {g.pertinente}")
    print("-" * 50)
    return {"pertinente": g.pertinente}

def riscrivi(state: State) -> dict:
    q = llm.invoke(
        f"Riformula come query di ricerca breve focalizzata su piatti, ingredienti o allergeni del menu: {state['domanda']}"
    ).content
    print(f"🔄 Riformulazione Query RAG: '{q}'")
    return {"query": q, "tentativi": state.get("tentativi", 0) + 1} 


def genera(state: State) -> dict:
    print("--- [NODO] ESECUZIONE RAG (genera) ---")
    documenti = state.get("documenti", [])
    if not documenti:
        return {"risposta": "Mi dispiace, non ho trovato informazioni rilevanti nel menu per rispondere alla tua richiesta."}

    ctx = "\n\n".join(documenti)

    system_prompt = (
        "Sei l'assistente virtuale della Trattoria Nonna Rosa.\n"
        "Rispondi alla domanda dell'utente basandoti ESCLUSIVAMENTE sul testo del menu e del regolamento forniti nel contesto.\n\n"
        "REGOLE RIGIDE DI LETTURA E LOGICA (VALIDE PER QUALSIASI MENU):\n\n"
        "1. LETTURA RIGIDA DELLA COLONNA ALLERGENI:\n"
        "   - La colonna 'Allergeni' indica gli allergeni PRESENTI nel piatto.\n"
        "   - Se la colonna 'Allergeni' di una riga contiene una sostanza (es. 'glutine', 'latte', 'pesce'), quel piatto CONTIENE tassativamente quella sostanza.\n"
        "   - È un errore critico classificare come 'senza [allergene]' un piatto la cui riga riporta quell'allergene nella colonna Allergeni.\n\n"
        "2. IDENTIFICAZIONE OPZIONI SPECIALI (Celiachia / Vegano / Intolleranze):\n"
        "   - Considera idonei solo i piatti che riportano l'etichetta esplicita nel nome/descrizione (es. '(senza glutine)', '(vegano)') oppure quelli la cui colonna Allergeni non contiene l'allergene richiesto.\n"
        "   - Se la colonna Allergeni di un piatto contiene 'glutine', quel piatto CONTIENE GLUTINE.\n"
        "   - Gli unici piatti idonei per celiaci/senza glutine sono quelli contrassegnati come '(senza glutine)'"
        "   - Ispeziona SEMPRE sia le tabelle dei piatti sia le note/regolamento generali, in quanto le note possono contenere opzioni su richiesta o supplementi.\n\n"
        "3. COMPLETEZZA SULLE INFORMAZIONI GENERALI ED ORARI:\n"
        "   - Quando vengono chiesti orari, regole o condizioni, estrai TUTTI i dettagli presenti nel testo (giorni, turni orari specifici, limitazioni di zona/dehors) senza sintetizzare od omettere cifre e orari prescritti.\n\n"
        "4. CANI E ANIMALI DOMESTICI:\n"
        "   - Quando l'utente chiede se è possibile portare un cane/animale, spiega chiaramente la condizione del locale facendo riferimento alle informazioni prensenti nella sezione note per gli ospiti del file menu_trattoria.md.\n"
        "   - Calibra la tua domanda sulla possibilità o meno di accettare animali in base alle informazioni che trovi nel file menu_trattoria.md.\n\n"
        "5. DIVIETO ASSOLUTO DI ALLUCINAZIONE:\n"
        "   - Basa la risposta SOLO su ciò che è scritto nel testo.\n"
        "   - Se un ingrediente, piatto o regola non è presente nel testo (es. pizza, panini), dichiara chiaramente che non è presente nel menu senza inventare condizioni o alternative non scritte."
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
    
    # SYSTEM PROMPT DIRETTIVO E RIGIDO
    prompt = (
        "Sei l'assistente virtuale della Trattoria Nonna Rosa per la gestione delle prenotazioni dei tavoli.\n\n"
        "GUARDRAIL DI SICUREZZA:\n"
        "1. NON invocare tool se l'utente non chiede esplicitamente di prenotare un tavolo.\n"
        "2. Se mancano parametri obbligatori (data YYYY-MM-DD, turno 1/2, persone, nome, telefono), richiedili SENZA eseguire tool."
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
            "Fornisci un riscontro chiaro al cliente."
        ).content
        return {"risposta": risposta_finale}
    
    # Se l'LLM risponde direttamente senza usare tool
    print("\nℹ️  Nessun tool invocato per questa richiesta.")
    return {"risposta": msg.content}


# 5. ESECUZIONE GRAFO

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



# from pathlib import Path

# # Determina la cartella corrente dello script .py
# base_dir = Path(__file__).parent if "__file__" in globals() else Path.cwd()
# output_path = base_dir / "grafo_langgraph.png"

# try:
#     # Ottiene i dati binari del PNG generato da Mermaid e li scrive su file
#     png_data = graph.get_graph().draw_mermaid_png()
#     output_path.write_bytes(png_data)
#     print(f"🎉 Grafo salvato con successo in: {output_path.resolve()}")
# except Exception as e:
#     print(f"❌ Impossibile generare l'immagine: {e}")


# 6. TEST ESECUZIONE (CON GESTIONE INTERRUPT)

if __name__ == "__main__":
#####======= TEST GIANLUCA
    # test_cases = [
    #     ("TC-01", "Qual è il numero di telefono e l'indirizzo?", "CAG (FAQ)"),
    #     ("TC-02", "Quali sono gli orari di apertura del weekend?", "RAG (Orari Menu)"),
    #     ("TC-03", "Posso portare un cane in trattoria?", "RAG (Regole Cani)"),
    #     ("TC-04", "Quali primi senza glutine ci sono?", "RAG (Menu SG)"),
    #     ("TC-05", "Ci sono piatti vegani tra i dolci o gli antipasti?", "RAG (Menu Vegan)"),
    #     ("TC-06", "Le orecchiette alle cime di rapa contengono pesce o lattosio?", "RAG (Allergeni)"),
    #     ("TC-07", "Avete la pizza margherita o i panini?", "RAG (Piatto assente)"),
    #     ("TC-08", "Vorrei informazioni sulla caparra per i gruppi", "RAG (Acconti)"),
    # ]

    # print("==================================================")
    # print("      AVVIO BATTERIA DI TEST CORRETTA             ")
    # print("==================================================\n")

    # for test_id, domanda, descrizione in test_cases:
    #     config = {"configurable": {"thread_id": f"sessione_{test_id}"}}
    #     print(f"[{test_id}] TEST: {descrizione}")
    #     print(f"Domanda: \"{domanda}\"")
    #     risultato = graph.invoke({"domanda": domanda}, config=config)
    #     print(f"Risposta ottenuta:\n{risultato['risposta']}")
    #     print("-" * 60 + "\n")

#####======= TEST EDOARDO
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