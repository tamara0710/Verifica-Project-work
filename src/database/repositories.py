"""Tool del chatbot della Trattoria "Da Nonna Rosa" (Traccia B).

1. regole della trattoria (i numeri si cambiano solo qui)
2. funzioni di supporto (non sono tool)
3. i tool: verifica_disponibilita, crea_prenotazione, annulla_o_consulta
"""
import json
import re
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool
from langgraph.types import interrupt

DB_PATH = Path(__file__).resolve().parent / "trattoria.db"

# ==========================================================
# 1. REGOLE DELLA TRATTORIA (dal PDF del menù)
# ==========================================================
PERSONE_MIN = 1
PERSONE_MAX = 12
SOGLIA_ACCONTO = 8            # acconto se le persone sono PIÙ di 8
ACCONTO_A_PERSONA = 10        # euro
GIORNI_MAX_ANTICIPO = 30
ORE_DISDETTA_GRATUITA = 24
MINUTI_TOLLERANZA = 15
DEHORS_DA_MESE = 4            # dehors aperto da aprile...
DEHORS_A_MESE = 10            # ...a ottobre
TELEFONO_TRATTORIA = "+39 080 1234567"   # come in info_trattoria.md

STATO_CONFERMATA = "confermata"
STATO_ATTESA_ACCONTO = "in_attesa_acconto"
STATO_ANNULLATA = "annullata"
STATI_ATTIVI = (STATO_CONFERMATA, STATO_ATTESA_ACCONTO)

SI = {"true", "1", "si", "sì", "s", "y", "yes", "ok", "confermo", "conferma"}


# ==========================================================
# 2. FUNZIONI DI SUPPORTO
# ==========================================================
def connetti_db():
    conn = sqlite3.connect(DB_PATH, timeout=20.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def normalizza_telefono(telefono) -> str:
    """Solo cifre, senza prefisso internazionale: '+39 347 557 7777' -> '3475577777'."""
    cifre = re.sub(r"\D", "", str(telefono or ""))
    if cifre.startswith("0039"):
        cifre = cifre[4:]
    elif cifre.startswith("39") and len(cifre) > 10:
        cifre = cifre[2:]
    return cifre


def telefono_valido(cifre: str) -> bool:
    return 6 <= len(cifre) <= 13


def a_intero(valore) -> Optional[int]:
    try:
        return int(float(str(valore).strip()))
    except (ValueError, TypeError):
        return None


def a_bool(valore) -> bool:
    if isinstance(valore, bool):
        return valore
    return str(valore).strip().lower() in SI


def elenco_turni() -> list:
    """Turni presenti nel DB, ordinati per orario (usato anche nel prompt dell'agente)."""
    with closing(connetti_db()) as conn:
        return [dict(r) for r in conn.execute("SELECT id, nome, orario FROM turni ORDER BY orario")]


def controlla_richiesta(data, persone):
    """Ritorna (giorno, persone, None) se va bene, oppure (None, None, messaggio_errore)."""
    n = a_intero(persone)
    if n is None or n < PERSONE_MIN or n > PERSONE_MAX:
        return None, None, f"Errore: il numero di persone deve essere compreso tra {PERSONE_MIN} e {PERSONE_MAX}."

    giorno = None
    for formato in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            giorno = datetime.strptime(str(data).strip(), formato).date()
            break
        except ValueError:
            continue
    if giorno is None:
        return None, None, "Errore: la data deve essere nel formato YYYY-MM-DD, per esempio 2026-11-10."

    oggi = datetime.now().date()
    if giorno < oggi:
        return None, None, "Errore: la data è già passata."
    if giorno > oggi + timedelta(days=GIORNI_MAX_ANTICIPO):
        return None, None, f"Errore: si può prenotare solo fino a {GIORNI_MAX_ANTICIPO} giorni prima."
    if giorno.weekday() == 0:
        return None, None, "Errore: il lunedì la trattoria è chiusa."
    return giorno, n, None


def controlla_turno(conn, giorno, turno_id):
    """Ritorna (turno, None) oppure (None, messaggio_errore)."""
    tid = a_intero(turno_id)
    turno = conn.execute("SELECT id, nome, orario FROM turni WHERE id = ?", (tid,)).fetchone() if tid is not None else None
    if turno is None:
        elenco = conn.execute("SELECT id, nome, orario FROM turni ORDER BY orario").fetchall()
        testo = ", ".join(f"{t['id']} = {t['nome']} {t['orario']}" for t in elenco)
        return None, f"Errore: il turno {turno_id} non esiste. Turni disponibili: {testo}."

    inizio = datetime.strptime(f"{giorno.isoformat()} {turno['orario']}", "%Y-%m-%d %H:%M")
    if datetime.now() > inizio + timedelta(minutes=MINUTI_TOLLERANZA):
        return None, "Errore: questo turno è già iniziato, scegli un altro turno."
    return turno, None


def trova_tavolo_libero(conn, giorno, turno_id: int, persone: int):
    """Tavolo libero più piccolo con posti >= persone, oppure None."""
    liberi = conn.execute("""
        SELECT t.id, t.posti, t.zona
        FROM tavoli AS t
        WHERE t.posti >= ?
          AND NOT EXISTS (
              SELECT 1 FROM prenotazioni AS p
              WHERE p.tavolo_id = t.id AND p.data = ? AND p.turno_id = ?
                AND p.stato IN ('confermata', 'in_attesa_acconto'))
        ORDER BY t.posti, t.id
    """, (persone, giorno.isoformat(), turno_id)).fetchall()

    dehors_aperto = DEHORS_DA_MESE <= giorno.month <= DEHORS_A_MESE
    for tavolo in liberi:
        if tavolo["zona"].lower() == "dehors" and not dehors_aperto:
            continue
        return tavolo
    return None


def turni_alternativi(conn, giorno, persone: int, escludi_turno: int) -> str:
    """Testo con gli altri turni del giorno che hanno ancora un tavolo libero."""
    liberi = []
    for t in conn.execute("SELECT id, nome, orario FROM turni WHERE id != ? ORDER BY orario", (escludi_turno,)):
        inizio = datetime.strptime(f"{giorno.isoformat()} {t['orario']}", "%Y-%m-%d %H:%M")
        if datetime.now() > inizio + timedelta(minutes=MINUTI_TOLLERANZA):
            continue
        if trova_tavolo_libero(conn, giorno, t["id"], persone) is not None:
            liberi.append(f"{t['nome']} {t['orario']} (turno {t['id']})")
    return (" Turni ancora liberi lo stesso giorno: " + ", ".join(liberi) + ".") if liberi else ""


def acconto_richiesto(persone: int) -> int:
    return persone * ACCONTO_A_PERSONA if persone > SOGLIA_ACCONTO else 0


def frase_acconto(importo: int) -> str:
    return (
        f"Serve un acconto di {importo} euro entro 48 ore dalla prenotazione. "
        f"Le modalità di pagamento vanno concordate con la trattoria "
        f"(tel. {TELEFONO_TRATTORIA}): il chatbot non raccoglie dati di carte."
    )


def trova_o_crea_cliente(conn, nome: str, telefono: str, email: str) -> int:
    riga = conn.execute("SELECT id FROM clienti WHERE telefono = ?", (telefono,)).fetchone()
    if riga is not None:
        return riga["id"]
    cur = conn.execute("INSERT INTO clienti (nome, telefono, email) VALUES (?, ?, ?)",
                       (nome, telefono, email or None))
    return cur.lastrowid


def prenotazioni_attive(conn, telefono: str) -> list:
    """Prenotazioni attive e non ancora scadute del cliente (solo future / turno in corso)."""
    righe = conn.execute("""
        SELECT p.id, p.data, p.persone, p.stato, p.segnala_celiachia,
               t.nome AS turno, t.orario, tv.id AS tavolo, tv.zona
        FROM prenotazioni AS p
        JOIN clienti AS c ON c.id = p.cliente_id
        JOIN turni AS t ON t.id = p.turno_id
        JOIN tavoli AS tv ON tv.id = p.tavolo_id
        WHERE c.telefono = ? AND p.stato IN ('confermata', 'in_attesa_acconto')
          AND p.data >= date('now', 'localtime')
        ORDER BY p.data, t.orario
    """, (telefono,)).fetchall()
    adesso = datetime.now()
    attive = []
    for p in righe:
        inizio = datetime.strptime(f"{p['data']} {p['orario']}", "%Y-%m-%d %H:%M")
        if adesso <= inizio + timedelta(minutes=MINUTI_TOLLERANZA):
            attive.append(p)
    return attive


def descrivi_prenotazione(p) -> str:
    cel = " (celiachia segnalata)" if p["segnala_celiachia"] == 1 else ""
    testo = (f"- Codice {p['id']}: {p['data']}, turno {p['turno']} ({p['orario']}), "
             f"{p['persone']} persone, tavolo {p['tavolo']} (zona {p['zona']}), stato: {p['stato']}{cel}")
    importo = acconto_richiesto(p["persone"])
    if importo > 0:
        testo += f"\n  Acconto: {importo} euro, da pagare entro 48 ore dalla prenotazione."
    return testo


# ==========================================================
# 3. I TOOL
# ==========================================================
@tool
def verifica_disponibilita(data: str, turno_id: int, persone: int) -> str:
    """Verifica se c'è un tavolo libero per data (YYYY-MM-DD), turno e numero di persone.

    turno_id: 1 = Pranzo 12:30, 3 = Pranzo 14:00, 2 = Cena 19:30, 4 = Cena 21:30.
    Regole: chiuso il lunedì, prenotazioni fino a 30 giorni prima, da 1 a 12 persone.
    Non prenota niente: serve solo a controllare.
    """
    try:
        giorno, n, errore = controlla_richiesta(data, persone)
        if errore:
            return errore
        with closing(connetti_db()) as conn:
            turno, errore = controlla_turno(conn, giorno, turno_id)
            if errore:
                return errore
            tavolo = trova_tavolo_libero(conn, giorno, turno["id"], n)
            if tavolo is None:
                return ("Non ci sono tavoli disponibili per la data e il turno richiesti."
                        + turni_alternativi(conn, giorno, n, turno["id"]))
        risposta = (f"Tavolo {tavolo['id']} disponibile il {giorno.isoformat()}, "
                    f"turno {turno['nome']} ({turno['orario']}): "
                    f"{tavolo['posti']} posti, zona {tavolo['zona']}.")
        importo = acconto_richiesto(n)
        if importo > 0:
            risposta += " " + frase_acconto(importo)
        return risposta
    except Exception as e:
        return f"Errore durante la verifica della disponibilità: {e}"


@tool
def crea_prenotazione(
    nome: str,
    telefono: str,
    data: str,
    turno_id: int,
    persone: int,
    email: str = "",
    note: str = "",
    celiachia: bool = False,
) -> str:
    """Crea una nuova prenotazione. Prima di scrivere sul database chiede conferma al cliente.

    Richiede nome, telefono, data (YYYY-MM-DD), turno_id e numero di persone.
    turno_id: 1 = Pranzo 12:30, 3 = Pranzo 14:00, 2 = Cena 19:30, 4 = Cena 21:30.
    Email e note sono opzionali. Metti celiachia=True se un ospite è celiaco
    (pasta senza glutine con cottura separata, supplemento 2 euro).
    Regole: chiuso il lunedì, fino a 30 giorni prima, da 1 a 12 persone,
    acconto di 10 euro a persona se il gruppo ha più di 8 persone.
    Ritorna la ricevuta in formato JSON.
    """
    # --- 1. pulizia e controlli (nessuna scrittura) ---
    nome = str(nome or "").strip()
    email = str(email or "").strip()
    note = str(note or "").strip()
    tel = normalizza_telefono(telefono)
    celiachia = a_bool(celiachia)

    if not nome:
        return "Errore: inserisci il nome del cliente."
    if not telefono_valido(tel):
        return "Errore: il numero di telefono non sembra valido (servono almeno 6 cifre)."

    giorno, n, errore = controlla_richiesta(data, persone)
    if errore:
        return errore
    data = giorno.isoformat()

    if celiachia:
        note = ("CELIACHIA: pasta senza glutine con cottura separata. " + note).strip()
    importo = acconto_richiesto(n)
    stato = STATO_ATTESA_ACCONTO if importo > 0 else STATO_CONFERMATA

    try:
        with closing(connetti_db()) as conn:
            turno, errore = controlla_turno(conn, giorno, turno_id)
            if errore:
                return errore
            doppia = conn.execute("""
                SELECT p.id FROM prenotazioni p JOIN clienti c ON c.id = p.cliente_id
                WHERE c.telefono = ? AND p.data = ? AND p.turno_id = ?
                  AND p.stato IN ('confermata', 'in_attesa_acconto')
            """, (tel, data, turno["id"])).fetchone()
            if doppia is not None:
                return (f"Errore: questo numero ha già la prenotazione {doppia['id']} per quel giorno e turno. "
                        f"Se vuoi cambiarla, annullala prima.")
            tavolo = trova_tavolo_libero(conn, giorno, turno["id"], n)
            if tavolo is None:
                return ("Errore: non ci sono tavoli disponibili per la data e il turno richiesti."
                        + turni_alternativi(conn, giorno, n, turno["id"]))
    except Exception as e:
        return f"Errore durante il controllo della disponibilità: {e}"

    # --- 2. conferma del cliente PRIMA dell'INSERT (fuori da ogni try/except) ---
    conferma = interrupt({
        "domanda": "Confermi la prenotazione?",
        "nome": nome,
        "telefono": tel,
        "data": data,
        "turno": f"{turno['nome']} ({turno['orario']})",
        "persone": n,
        "zona": tavolo["zona"],
        "acconto_euro": importo,
        "celiachia": celiachia,
    })
    if not a_bool(conferma):
        return json.dumps({"esito": "non_confermata",
                           "messaggio": "Prenotazione non confermata dal cliente."}, ensure_ascii=False)

    # --- 3. scrittura in una transazione ---
    with closing(connetti_db()) as conn:
        try:
            conn.execute("BEGIN IMMEDIATE")
            tavolo = trova_tavolo_libero(conn, giorno, turno["id"], n)   # ricontrollo
            if tavolo is None:
                conn.rollback()
                return "Errore: nel frattempo il tavolo è stato prenotato da un altro cliente."
            cliente_id = trova_o_crea_cliente(conn, nome, tel, email)
            cur = conn.execute("""
                INSERT INTO prenotazioni (cliente_id, tavolo_id, turno_id, data, persone, note, stato, segnala_celiachia)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (cliente_id, tavolo["id"], turno["id"], data, n, note or None, stato, 1 if celiachia else 0))
            prenotazione_id = cur.lastrowid
            conn.commit()
        except Exception as e:
            conn.rollback()
            return f"Errore durante la creazione della prenotazione: {e}"

    # --- 4. ricevuta JSON ---
    ricevuta = {
        "esito": "ok", "codice": prenotazione_id, "nome": nome, "telefono": tel, "data": data,
        "turno": turno["nome"], "orario": turno["orario"], "tavolo": tavolo["id"], "zona": tavolo["zona"],
        "persone": n, "stato": stato, "acconto_euro": importo, "celiachia": celiachia,
    }
    if importo > 0:
        ricevuta["info_acconto"] = frase_acconto(importo)
    return json.dumps(ricevuta, ensure_ascii=False)


AZIONI_CONSULTA = {"consulta", "consultare", "mostra", "visualizza", "lista", "elenco", "vedi", "controlla"}
AZIONI_ANNULLA = {"annulla", "annullare", "cancella", "cancellare", "disdici", "disdire", "disdetta"}


@tool
def annulla_o_consulta(
    azione: str,
    telefono: str,
    prenotazione_id: Optional[int] = None,
) -> str:
    """Consulta o annulla le prenotazioni di un cliente usando il suo telefono.

    azione = "consulta": mostra le prenotazioni attive e le regole su acconto e disdetta.
    azione = "annulla": annulla la prenotazione prenotazione_id (se il cliente ne ha una sola
    si può omettere). Il tool chiede da solo la conferma al cliente prima di annullare.
    Disdetta gratuita fino a 24 ore prima; per i gruppi con acconto (oltre 8 persone),
    una disdetta più tardi comporta la perdita dell'acconto.
    """
    az = str(azione or "").strip().lower()
    if az in AZIONI_CONSULTA:
        az = "consulta"
    elif az in AZIONI_ANNULLA:
        az = "annulla"
    else:
        return "Errore: l'azione deve essere 'consulta' oppure 'annulla'."

    tel = normalizza_telefono(telefono)
    if not telefono_valido(tel):
        return "Errore: inserisci il numero di telefono della prenotazione."

    try:
        with closing(connetti_db()) as conn:
            attive = prenotazioni_attive(conn, tel)

            if az == "consulta":
                if not attive:
                    return "Non ho trovato prenotazioni attive per questo numero di telefono."
                return (
                    "Prenotazioni trovate:\n" + "\n".join(descrivi_prenotazione(p) for p in attive)
                    + "\n\nRegole: si prenota fino a "
                    f"{GIORNI_MAX_ANTICIPO} giorni prima, da {PERSONE_MIN} a {PERSONE_MAX} persone. "
                    f"Per i gruppi oltre {SOGLIA_ACCONTO} persone serve un acconto di {ACCONTO_A_PERSONA} euro "
                    f"a persona entro 48 ore. La disdetta è gratuita fino a {ORE_DISDETTA_GRATUITA} ore prima; "
                    f"dopo, i gruppi con acconto lo perdono. Il tavolo viene tenuto {MINUTI_TOLLERANZA} minuti "
                    f"dopo l'inizio del turno."
                )

            # ---- annulla ----
            pid = a_intero(prenotazione_id)
            if pid is None:
                if len(attive) == 1:
                    pid = attive[0]["id"]
                elif not attive:
                    return "Non ho trovato prenotazioni attive per questo numero di telefono."
                else:
                    return ("Ci sono più prenotazioni attive, dimmi il codice di quella da annullare:\n"
                            + "\n".join(descrivi_prenotazione(p) for p in attive))

            p = conn.execute("""
                SELECT p.id, p.data, p.persone, p.stato, t.nome AS turno, t.orario
                FROM prenotazioni AS p
                JOIN clienti AS c ON c.id = p.cliente_id
                JOIN turni AS t ON t.id = p.turno_id
                WHERE p.id = ? AND c.telefono = ?
            """, (pid, tel)).fetchone()
            if p is None:
                return "Errore: non esiste una prenotazione con questo codice per questo telefono."
            if p["stato"] not in STATI_ATTIVI:
                return f"La prenotazione {p['id']} non è attiva (stato: {p['stato']})."

            inizio = datetime.strptime(f"{p['data']} {p['orario']}", "%Y-%m-%d %H:%M")
            adesso = datetime.now()
            if adesso > inizio + timedelta(minutes=MINUTI_TOLLERANZA):
                return (f"Il turno è iniziato da più di {MINUTI_TOLLERANZA} minuti, quindi il tavolo "
                        f"non è più tenuto e non posso annullare la prenotazione.")

            importo = acconto_richiesto(p["persone"])
            in_ritardo = (inizio - adesso) < timedelta(hours=ORE_DISDETTA_GRATUITA)
            perde_acconto = in_ritardo and importo > 0
    except Exception as e:
        return f"Errore durante l'operazione: {e}"

    # conferma del cliente PRIMA di annullare (fuori da ogni try/except)
    conferma = interrupt({
        "domanda": "Confermi l'annullamento?",
        "codice": p["id"],
        "data": p["data"],
        "turno": f"{p['turno']} ({p['orario']})",
        "persone": p["persone"],
        "perdi_acconto_euro": importo if perde_acconto else 0,
    })
    if not a_bool(conferma):
        return f"Annullamento non confermato: la prenotazione {p['id']} resta attiva."

    try:
        with closing(connetti_db()) as conn:
            cur = conn.execute(
                "UPDATE prenotazioni SET stato = ? WHERE id = ? AND stato IN ('confermata', 'in_attesa_acconto')",
                (STATO_ANNULLATA, p["id"]))
            conn.commit()
            if cur.rowcount == 0:
                return f"La prenotazione {p['id']} non risulta più attiva."
    except Exception as e:
        return f"Errore durante l'annullamento: {e}"

    if perde_acconto:
        return f"Prenotazione {p['id']} annullata. Essendo meno di {ORE_DISDETTA_GRATUITA} ore al turno, l'acconto di {importo} euro non viene restituito."
    return f"Prenotazione {p['id']} annullata con successo."
