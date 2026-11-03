"""Tool del chatbot della Trattoria "Da Nonna Rosa" (Traccia B).

Il file è diviso in 3 parti:
1. impostazioni e regole della trattoria (modifichi i numeri solo qui)
2. funzioni di supporto (NON sono tool: le usano i tool qui sotto)
3. i tool: verifica_disponibilita, crea_prenotazione, annulla_o_consulta
"""
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path

from langchain_core.tools import tool
from langgraph.types import interrupt

DB_PATH = Path(__file__).resolve().parents[1] / "database" / "trattoria.db"

# ==========================================================
# 1. REGOLE DELLA TRATTORIA (prese dal PDF del menù)
# ==========================================================
PERSONE_MIN = 1
PERSONE_MAX = 12
SOGLIA_ACCONTO = 8            # l'acconto serve se le persone sono PIÙ di 8
ACCONTO_A_PERSONA = 10        # euro
GIORNI_MAX_ANTICIPO = 30
ORE_DISDETTA_GRATUITA = 24
MINUTI_TOLLERANZA = 15        # il tavolo viene tenuto 15 minuti dopo l'inizio del turno
DEHORS_DA_MESE = 4            # il dehors è aperto da aprile...
DEHORS_A_MESE = 10            # ...a ottobre
TELEFONO_TRATTORIA = "347 890 3567"

# Stati delle prenotazioni nel DB.
# Se nel tuo DB l'annullamento si chiama 'disdetta', cambia solo qui.
STATO_CONFERMATA = "confermata"
STATO_ATTESA_ACCONTO = "in_attesa_acconto"
STATO_ANNULLATA = "annullata"


# ==========================================================
# 2. FUNZIONI DI SUPPORTO (non sono tool)
# ==========================================================
def connetti_db():
    """Crea una connessione al database SQLite."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # così posso scrivere riga["id"]
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def controlla_richiesta(data: str, persone: int):
    """Controlla persone e data.
    Ritorna (giorno, None) se va bene, oppure (None, messaggio_errore)."""
    if persone < PERSONE_MIN or persone > PERSONE_MAX:
        return None, f"Errore: il numero di persone deve essere compreso tra {PERSONE_MIN} e {PERSONE_MAX}."

    try:
        giorno = datetime.strptime(data, "%Y-%m-%d").date()
    except ValueError:
        return None, "Errore: la data deve essere nel formato YYYY-MM-DD, per esempio 2026-11-10."

    oggi = datetime.now().date()

    if giorno < oggi:
        return None, "Errore: la data è già passata."
    if giorno > oggi + timedelta(days=GIORNI_MAX_ANTICIPO):
        return None, f"Errore: si può prenotare solo fino a {GIORNI_MAX_ANTICIPO} giorni prima."
    if giorno.weekday() == 0:  # in Python lunedì = 0
        return None, "Errore: il lunedì la trattoria è chiusa."

    return giorno, None


def controlla_turno(conn, giorno, turno_id: int):
    """Controlla che il turno esista e non sia già iniziato.
    Ritorna (turno, None) oppure (None, messaggio_errore)."""
    turno = conn.execute(
        "SELECT id, nome, orario FROM turni WHERE id = ?",
        (turno_id,)
    ).fetchone()

    if turno is None:
        # aiuto l'agente elencando i turni che esistono davvero
        elenco = conn.execute("SELECT id, nome, orario FROM turni ORDER BY orario").fetchall()
        testo = ", ".join(f"{t['id']} = {t['nome']} {t['orario']}" for t in elenco)
        return None, f"Errore: il turno {turno_id} non esiste. Turni disponibili: {testo}."

    # se prenoto per oggi, il turno non deve essere già iniziato
    inizio = datetime.strptime(f"{giorno.isoformat()} {turno['orario']}", "%Y-%m-%d %H:%M")
    if datetime.now() > inizio + timedelta(minutes=MINUTI_TOLLERANZA):
        return None, "Errore: questo turno è già iniziato, scegli un altro turno."

    return turno, None


def trova_tavolo_libero(conn, giorno, turno_id: int, persone: int):
    """Ritorna il tavolo libero più piccolo che basta per le persone, oppure None.
    La usano sia verifica_disponibilita sia crea_prenotazione: così scelgono sempre lo stesso tavolo."""
    tavoli_liberi = conn.execute("""
        SELECT t.id, t.posti, t.zona
        FROM tavoli AS t
        WHERE t.posti >= ?
          AND NOT EXISTS (
              SELECT 1
              FROM prenotazioni AS p
              WHERE p.tavolo_id = t.id
                AND p.data = ?
                AND p.turno_id = ?
                AND p.stato IN ('confermata', 'in_attesa_acconto')
          )
        ORDER BY t.posti, t.id
    """, (persone, giorno.isoformat(), turno_id)).fetchall()

    dehors_aperto = DEHORS_DA_MESE <= giorno.month <= DEHORS_A_MESE

    for tavolo in tavoli_liberi:
        # se il dehors è chiuso salto i tavoli esterni
        if tavolo["zona"] == "dehors" and not dehors_aperto:
            continue
        return tavolo

    return None


def acconto_richiesto(persone: int) -> int:
    """Ritorna l'acconto in euro (0 se non serve)."""
    if persone > SOGLIA_ACCONTO:
        return persone * ACCONTO_A_PERSONA
    return 0


def frase_acconto(importo: int) -> str:
    """Frase da dire al cliente quando c'è un acconto."""
    return (
        f"Serve un acconto di {importo} euro entro 48 ore dalla prenotazione. "
        f"Le modalità di pagamento vanno concordate con la trattoria "
        f"(tel. {TELEFONO_TRATTORIA}): il chatbot non raccoglie dati di carte."
    )


def trova_o_crea_cliente(conn, nome: str, telefono: str, email: str) -> int:
    """Cerca il cliente dal telefono, se non c'è lo crea. Ritorna l'id del cliente."""
    riga = conn.execute(
        "SELECT id FROM clienti WHERE telefono = ?",
        (telefono,)
    ).fetchone()

    if riga is not None:
        return riga["id"]

    cursor = conn.execute(
        "INSERT INTO clienti (nome, telefono, email) VALUES (?, ?, ?)",
        (nome, telefono, email if email else None),
    )
    return cursor.lastrowid


# ==========================================================
# 3. I TOOL
# ==========================================================
@tool
def verifica_disponibilita(data: str, turno_id: int, persone: int) -> str:
    """Verifica se c'è un tavolo libero per data (YYYY-MM-DD), turno e numero di persone.

    Regole: trattoria chiusa il lunedì, prenotazioni fino a 30 giorni prima,
    da 1 a 12 persone. Se il turno_id non esiste, l'errore elenca i turni validi.
    Non prenota niente: serve solo a controllare.
    """
    giorno, errore = controlla_richiesta(data, persone)
    if errore:
        return errore

    with closing(connetti_db()) as conn:
        turno, errore = controlla_turno(conn, giorno, turno_id)
        if errore:
            return errore

        tavolo = trova_tavolo_libero(conn, giorno, turno_id, persone)

    if tavolo is None:
        return "Non ci sono tavoli disponibili per la data e il turno richiesti."

    risposta = (
        f"Tavolo {tavolo['id']} disponibile il {giorno.isoformat()}, "
        f"turno {turno['nome']} ({turno['orario']}): "
        f"{tavolo['posti']} posti, zona {tavolo['zona']}."
    )

    importo = acconto_richiesto(persone)
    if importo > 0:
        risposta += " " + frase_acconto(importo)

    return risposta


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
    Email e note sono opzionali. Metti celiachia=True se un ospite è celiaco
    (pasta senza glutine con cottura separata, supplemento 2 euro).
    Regole: trattoria chiusa il lunedì, prenotazioni fino a 30 giorni prima,
    da 1 a 12 persone, acconto di 10 euro a persona se il gruppo ha più di 8 persone.
    Ritorna la ricevuta in formato JSON.
    """
    # --- 1. pulizia e controlli (nessuna scrittura sul DB) ---
    nome = nome.strip()
    telefono = telefono.strip()
    email = email.strip()
    note = note.strip()

    if not nome or not telefono:
        return "Errore: inserisci il nome e il telefono del cliente."

    giorno, errore = controlla_richiesta(data, persone)
    if errore:
        return errore

    data = giorno.isoformat()  # formato sempre uguale nel DB (es. 2026-11-05)

    if celiachia:
        note = ("CELIACHIA: pasta senza glutine con cottura separata. " + note).strip()

    importo = acconto_richiesto(persone)
    if importo > 0:
        stato = STATO_ATTESA_ACCONTO
    else:
        stato = STATO_CONFERMATA

    segnala_celiachia = 1 if celiachia else 0

    with closing(connetti_db()) as conn:
        turno, errore = controlla_turno(conn, giorno, turno_id)
        if errore:
            return errore

        tavolo = trova_tavolo_libero(conn, giorno, turno_id, persone)

    if tavolo is None:
        return "Errore: non ci sono tavoli disponibili per la data e il turno richiesti."

    # --- 2. conferma del cliente PRIMA dell'INSERT ---
    # L'interrupt sta fuori dal try/except, altrimenti verrebbe catturato come errore.
    conferma = interrupt({
        "domanda": "Confermi la prenotazione?",
        "nome": nome,
        "telefono": telefono,
        "data": data,
        "turno": f"{turno['nome']} ({turno['orario']})",
        "persone": persone,
        "zona": tavolo["zona"],
        "acconto_euro": importo,
        "celiachia": celiachia,
    })

    if conferma is not True:
        return json.dumps(
            {"esito": "non_confermata", "messaggio": "Prenotazione non confermata dal cliente."},
            ensure_ascii=False,
        )

    # --- 3. scrivo nel DB (tutto in una transazione) ---
    with closing(connetti_db()) as conn:
        try:
            conn.execute("BEGIN IMMEDIATE")

            # ricontrollo il tavolo: nel frattempo un altro cliente potrebbe averlo preso
            tavolo = trova_tavolo_libero(conn, giorno, turno_id, persone)
            if tavolo is None:
                conn.rollback()
                return "Errore: nel frattempo il tavolo è stato prenotato da un altro cliente."

            cliente_id = trova_o_crea_cliente(conn, nome, telefono, email)

            cursor = conn.execute("""
                INSERT INTO prenotazioni (cliente_id, tavolo_id, turno_id, data, persone, note, stato, segnala_celiachia)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                cliente_id,
                tavolo["id"],
                turno_id,
                data,
                persone,
                note if note else None,
                stato,
                segnala_celiachia,
            ))
            prenotazione_id = cursor.lastrowid

            conn.commit()

        except Exception as e:
            conn.rollback()
            return f"Errore durante la creazione della prenotazione: {str(e)}"

    # --- 4. ricevuta in JSON ---
    ricevuta = {
        "esito": "ok",
        "codice": prenotazione_id,
        "nome": nome,
        "data": data,
        "turno": turno["nome"],
        "orario": turno["orario"],
        "tavolo": tavolo["id"],
        "zona": tavolo["zona"],
        "persone": persone,
        "stato": stato,
        "acconto_euro": importo,
        "celiachia": celiachia,
    }
    if importo > 0:
        ricevuta["info_acconto"] = frase_acconto(importo)

    return json.dumps(ricevuta, ensure_ascii=False)


@tool
def annulla_o_consulta(
    azione: str,
    telefono: str,
    prenotazione_id: int = 0,
    conferma_perdita_acconto: bool = False,
) -> str:
    """Consulta o annulla le prenotazioni di un cliente usando il suo telefono.

    azione = "consulta": mostra le prenotazioni attive future e le regole dell'acconto
    (10 euro a persona se il gruppo ha più di 8 persone, da pagare entro 48 ore).

    azione = "annulla": annulla la prenotazione con prenotazione_id.
    La disdetta è gratuita fino a 24 ore prima del turno. Per i gruppi con
    acconto, se si disdice dopo, l'acconto si perde: PRIMA di usare
    conferma_perdita_acconto=True bisogna chiedere conferma al cliente.
    """
    telefono = telefono.strip()
    if telefono == "":
        return "Errore: inserisci il numero di telefono della prenotazione."

    if azione != "consulta" and azione != "annulla":
        return "Errore: l'azione deve essere 'consulta' oppure 'annulla'."

    with closing(connetti_db()) as conn:
        try:
            # ==============================
            # PARTE 1: CONSULTA
            # ==============================
            if azione == "consulta":
                prenotazioni = conn.execute("""
                    SELECT p.id, p.data, p.persone, p.stato, p.segnala_celiachia,
                           t.nome AS turno, t.orario,
                           tv.id AS tavolo, tv.zona
                    FROM prenotazioni AS p
                    JOIN clienti AS c ON c.id = p.cliente_id
                    JOIN turni AS t ON t.id = p.turno_id
                    JOIN tavoli AS tv ON tv.id = p.tavolo_id
                    WHERE c.telefono = ?
                      AND p.stato IN ('confermata', 'in_attesa_acconto')
                      AND p.data >= date('now', 'localtime')
                    ORDER BY p.data, t.orario
                """, (telefono,)).fetchall()

                adesso = datetime.now()
                righe = []

                for p in prenotazioni:
                    # salto i turni di oggi già passati
                    inizio = datetime.strptime(p["data"] + " " + p["orario"], "%Y-%m-%d %H:%M")
                    if adesso > inizio + timedelta(minutes=MINUTI_TOLLERANZA):
                        continue

                    celiachia_str = " (Celiachia segnalata)" if p["segnala_celiachia"] == 1 else ""
                    testo = (
                        f"- Codice {p['id']}: {p['data']}, turno {p['turno']} ({p['orario']}), "
                        f"{p['persone']} persone, tavolo {p['tavolo']} (zona {p['zona']}), "
                        f"stato: {p['stato']}{celiachia_str}"
                    )

                    importo = acconto_richiesto(p["persone"])
                    if importo > 0:
                        testo += f"\n  Acconto: {importo} euro, da pagare entro 48 ore dalla prenotazione."

                    righe.append(testo)

                if len(righe) == 0:
                    return "Non ho trovato prenotazioni attive per questo numero di telefono."

                return (
                    "Prenotazioni trovate:\n" + "\n".join(righe)
                    + f"\n\nRegole: si prenota fino a {GIORNI_MAX_ANTICIPO} giorni prima, "
                    f"da {PERSONE_MIN} a {PERSONE_MAX} persone."
                )

            # ==============================
            # PARTE 2: ANNULLA
            # ==============================
            if prenotazione_id == 0:
                return "Errore: dimmi il codice della prenotazione da annullare."

            # cerco la prenotazione (deve essere di quel telefono)
            p = conn.execute("""
                SELECT p.id, p.data, p.persone, p.stato, t.orario
                FROM prenotazioni AS p
                JOIN clienti AS c ON c.id = p.cliente_id
                JOIN turni AS t ON t.id = p.turno_id
                WHERE p.id = ? AND c.telefono = ?
            """, (prenotazione_id, telefono)).fetchone()

            if p is None:
                return "Errore: non esiste una prenotazione con questo codice per questo telefono."

            if p["stato"] not in (STATO_CONFERMATA, STATO_ATTESA_ACCONTO):
                return f"La prenotazione {p['id']} non è attiva (stato: {p['stato']})."

            inizio_turno = datetime.strptime(p["data"] + " " + p["orario"], "%Y-%m-%d %H:%M")
            adesso = datetime.now()

            # il tavolo viene tenuto solo 15 minuti dopo l'inizio del turno
            if adesso > inizio_turno + timedelta(minutes=MINUTI_TOLLERANZA):
                return (
                    f"Il turno è iniziato da più di {MINUTI_TOLLERANZA} minuti, quindi il tavolo "
                    f"non è più tenuto e non posso annullare la prenotazione."
                )

            # disdetta gratuita solo fino a 24 ore prima
            disdetta_in_ritardo = (inizio_turno - adesso) < timedelta(hours=ORE_DISDETTA_GRATUITA)

            # l'acconto si perde solo per i gruppi con acconto
            importo = acconto_richiesto(p["persone"])
            perde_acconto = disdetta_in_ritardo and importo > 0

            if perde_acconto and not conferma_perdita_acconto:
                return (
                    f"Attenzione: mancano meno di {ORE_DISDETTA_GRATUITA} ore al turno. "
                    f"Se annulli ora perdi l'acconto di {importo} euro. "
                    f"Chiedi al cliente se vuole continuare e poi richiama il tool "
                    f"con conferma_perdita_acconto=True."
                )

            conn.execute(
                "UPDATE prenotazioni SET stato = ? WHERE id = ?",
                (STATO_ANNULLATA, p["id"])
            )
            conn.commit()

            if perde_acconto:
                return f"Prenotazione {p['id']} annullata. L'acconto non viene restituito."
            return f"Prenotazione {p['id']} annullata con successo."

        except Exception as e:
            conn.rollback()
            return f"Errore durante l'operazione: {str(e)}"


# Per provare i tool senza l'agente (crea_prenotazione ha l'interrupt,
# quindi si prova solo dentro il grafo con il checkpointer).
if __name__ == "__main__":
    print(DB_PATH, DB_PATH.exists())
    print(verifica_disponibilita.invoke({"data": "2026-11-10", "turno_id": 1, "persone": 4}))
    print(annulla_o_consulta.invoke({"azione": "consulta", "telefono": "3331234567"}))