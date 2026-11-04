import sqlite3
from pathlib import Path
from langchain_core.tools import tool
from datetime import datetime, timedelta

DB_PATH = Path(__file__).resolve().parents[1] / "database" / "trattoria.db"

def connetti_db():
    """Crea una connessione al database SQLite."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Per ottenere risultati come dizionari
    conn.execute("PRAGMA foreign_keys = ON")  # Abilita le chiavi esterne
    return conn

@tool
def verifica_disponibilita(data: str, turno_id: int, persone: int) -> str:
    """Verifica se c'è un tavolo libero per data (YYYY-MM-DD), turno e numero di persone.

    Regole: trattoria chiusa il lunedì, prenotazioni fino a 30 giorni prima,
    da 1 a 12 persone.
    """
    # controllo il numero di persone
    if persone < 1 or persone > 12:
        return "Errore: il numero di persone deve essere compreso tra 1 e 12."

    # controllo che la data sia scritta bene
    try:
        giorno = datetime.strptime(data, "%Y-%m-%d").date()
    except ValueError:
        return "Errore: la data deve essere nel formato YYYY-MM-DD, per esempio 2026-11-10."

    oggi = datetime.now().date()
    limite = oggi + timedelta(days=30)

    # la data non deve essere passata
    if giorno < oggi:
        return "Errore: la data è già passata."

    # non si può prenotare oltre 30 giorni
    if giorno > limite:
        return "Errore: si può prenotare solo fino a 30 giorni prima."

    # il lunedì è chiuso (in Python lunedì = 0)
    if giorno.weekday() == 0:
        return "Errore: il lunedì la trattoria è chiusa."

    # il dehors è aperto solo da aprile (4) a ottobre (10)
    dehors_aperto = giorno.month >= 4 and giorno.month <= 10

    with connetti_db() as conn:
        # controllo che il turno esista
        turno = conn.execute(
            "SELECT id, nome, orario FROM turni WHERE id = ?",
            (turno_id,)
        ).fetchone()

        if turno is None:
            return f"Errore: il turno con ID {turno_id} non esiste."

        # prendo tutti i tavoli liberi con abbastanza posti, dal più piccolo al più grande
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
            ORDER BY t.posti
        """, (persone, data, turno_id)).fetchall()

    # scelgo il primo tavolo che si può usare
    tavolo = None
    for t in tavoli_liberi:
        # se il dehors è chiuso salto i tavoli esterni
        if t["zona"] == "dehors" and not dehors_aperto:
            continue
        tavolo = t
        break

    if tavolo is None:
        return "Non ci sono tavoli disponibili per la data e il turno richiesti."

    risposta = (
        f"Tavolo {tavolo['id']} disponibile il {data}, "
        f"turno {turno['nome']} ({turno['orario']}): "
        f"{tavolo['posti']} posti, zona {tavolo['zona']}."
    )

    # per i gruppi con più di 8 persone c'è l'acconto
    if persone > 8:
        acconto = persone * 10
        risposta += f" Per più di 8 persone serve un acconto di {acconto} euro entro 48 ore dalla prenotazione."

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
) -> str:
    """Crea una nuova prenotazione nel database.

    Richiede nome, telefono, data (YYYY-MM-DD), ID del turno e numero di persone.
    Email e note sono opzionali.
    Regole: trattoria chiusa il lunedì, prenotazioni fino a 30 giorni prima,
    da 1 a 12 persone, acconto di 10 euro a persona se il gruppo ha più di 8 persone.
    """
    # 1. Pulizia dei dati
    nome = nome.strip()
    telefono = telefono.strip()
    email = email.strip()
    note = note.strip()

    if not nome or not telefono:
        return "Errore: inserisci il nome e il telefono del cliente."

    # 2. Validazione della data
    try:
        data_obj = datetime.strptime(data, "%Y-%m-%d")
    except ValueError:
        return "Errore: la data deve essere nel formato YYYY-MM-DD, per esempio 2026-11-10."

    oggi = datetime.now().date()
    giorno = data_obj.date()

    # data non passata
    if giorno < oggi:
        return "Errore: la data è già passata."

    # massimo 30 giorni in anticipo
    if giorno > oggi + timedelta(days=30):
        return "Errore: si può prenotare solo fino a 30 giorni prima."

    # lunedì chiuso (in Python lunedì = 0)
    if giorno.weekday() == 0:
        return "Errore: il lunedì la trattoria è chiusa."

    # 3. Numero di persone
    if not (1 <= persone <= 12):
        return "Errore: il numero di persone deve essere compreso tra 1 e 12."

    # 4. Il dehors è aperto solo da aprile a ottobre
    if 4 <= giorno.month <= 10:
        filtro_zona = ""
    else:
        filtro_zona = "AND t.zona != 'dehors'"

    # 5. Stato e acconto
    if persone > 8:
        stato = "in_attesa_acconto"
        acconto = persone * 10
    else:
        stato = "confermata"
        acconto = 0

    # 6. Operazioni sul database
    with connetti_db() as conn:
        try:
            conn.execute("BEGIN IMMEDIATE")

            # il turno esiste?
            turno = conn.execute(
                "SELECT id, nome, orario FROM turni WHERE id = ?",
                (turno_id,)
            ).fetchone()

            if turno is None:
                conn.rollback()
                return f"Errore: il turno con ID {turno_id} non esiste."

            # primo tavolo libero con posti sufficienti
            tavolo_query = f"""
                SELECT t.id, t.posti, t.zona
                FROM tavoli AS t
                WHERE t.posti >= ?
                  {filtro_zona}
                  AND NOT EXISTS (
                      SELECT 1
                      FROM prenotazioni AS p
                      WHERE p.tavolo_id = t.id
                        AND p.data = ?
                        AND p.turno_id = ?
                        AND p.stato IN ('confermata', 'in_attesa_acconto')
                  )
                ORDER BY t.posti ASC
                LIMIT 1
            """
            tavolo = conn.execute(tavolo_query, (persone, data, turno_id)).fetchone()

            if tavolo is None:
                conn.rollback()
                return "Errore: non ci sono tavoli disponibili per la data e il turno richiesti."

            # cliente già esistente?
            cliente = conn.execute(
                "SELECT id FROM clienti WHERE telefono = ? LIMIT 1",
                (telefono,)
            ).fetchone()

            if cliente is None:
                cursor = conn.execute(
                    "INSERT INTO clienti (nome, telefono, email) VALUES (?, ?, ?)",
                    (nome, telefono, email if email else None),
                )
                cliente_id = cursor.lastrowid
            else:
                cliente_id = cliente["id"]

            # inserisco la prenotazione (ora con lo stato)
            cursor = conn.execute("""
                INSERT INTO prenotazioni (cliente_id, tavolo_id, turno_id, data, persone, note, stato)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                cliente_id,
                tavolo["id"],
                turno_id,
                data,
                persone,
                note if note else None,
                stato,
            ))
            prenotazione_id = cursor.lastrowid

            conn.commit()

            risposta = (
                f"Prenotazione creata con successo! "
                f"Codice: {prenotazione_id}; "
                f"Data: {data}, turno {turno['nome']} ({turno['orario']}); "
                f"Tavolo: {tavolo['id']} (zona {tavolo['zona']}); "
                f"Persone: {persone}."
            )

            # messaggio extra per i gruppi grandi
            if acconto > 0:
                risposta += (
                    f" Per i gruppi con più di 8 persone serve un acconto di {acconto} euro, "
                    f"da versare entro 48 ore dalla prenotazione."
                )

            return risposta

        except Exception as e:
            conn.rollback()
            return f"Errore durante la creazione della prenotazione: {str(e)}"



@tool
def annulla_o_consulta(
    azione: str,
    telefono: str,
    prenotazione_id: int = 0,
    conferma_perdita_acconto: bool = False,
) -> str:
    """Consulta o annulla le prenotazioni di un cliente usando il suo telefono.

    azione = "consulta": mostra le prenotazioni attive e le regole dell'acconto
    (10 euro a persona se il gruppo ha più di 8 persone, da pagare entro 48 ore).

    azione = "annulla": annulla la prenotazione con prenotazione_id.
    La disdetta è gratuita fino a 24 ore prima del turno. Per i gruppi con
    acconto, se si disdice dopo, l'acconto si perde (serve conferma_perdita_acconto=True).
    """

    # pulisco il telefono e controllo che ci sia
    telefono = telefono.strip()
    if telefono == "":
        return "Errore: inserisci il numero di telefono della prenotazione."

    # controllo che l'azione sia una di quelle giuste
    if azione != "consulta" and azione != "annulla":
        return "Errore: l'azione deve essere 'consulta' oppure 'annulla'."

    with connetti_db() as conn:
        try:
            # ==============================
            # PARTE 1: CONSULTA
            # ==============================
            if azione == "consulta":
                prenotazioni = conn.execute("""
                    SELECT p.id, p.data, p.persone, p.stato,
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

                if len(prenotazioni) == 0:
                    return "Non ho trovato prenotazioni attive per questo numero di telefono."

                risposta = "Prenotazioni trovate:\n"

                for p in prenotazioni:
                    risposta += (
                        f"- Codice {p['id']}: {p['data']}, turno {p['turno']} ({p['orario']}), "
                        f"{p['persone']} persone, tavolo {p['tavolo']} (zona {p['zona']}), "
                        f"stato: {p['stato']}\n"
                    )

                    # se il gruppo è più grande di 8 c'è l'acconto
                    if p["persone"] > 8:
                        acconto = p["persone"] * 10
                        risposta += f"  Acconto: {acconto} euro, da pagare entro 48 ore dalla prenotazione.\n"

                risposta += "\nRegole: si prenota fino a 30 giorni prima, da 1 a 12 persone."
                return risposta

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

            if p["stato"] not in ("confermata", "in_attesa_acconto"):
                return f"La prenotazione {p['id']} non è attiva (stato: {p['stato']})."

            # calcolo quando inizia il turno e che ora è adesso
            inizio_turno = datetime.strptime(p["data"] + " " + p["orario"], "%Y-%m-%d %H:%M")
            adesso = datetime.now()

            # il tavolo viene tenuto solo 15 minuti dopo l'inizio del turno
            if adesso > inizio_turno + timedelta(minutes=15):
                return (
                    "Il turno è iniziato da più di 15 minuti, quindi il tavolo non è più tenuto "
                    "e non posso annullare la prenotazione."
                )

            # controllo se mancano meno di 24 ore
            ore_mancanti = (inizio_turno - adesso).total_seconds() / 3600
            disdetta_in_ritardo = ore_mancanti < 24

            # l'acconto si perde solo per i gruppi con più di 8 persone
            perde_acconto = disdetta_in_ritardo and p["persone"] > 8

            if perde_acconto and not conferma_perdita_acconto:
                acconto = p["persone"] * 10
                return (
                    f"Attenzione: mancano meno di 24 ore al turno. Se annulli ora "
                    f"perdi l'acconto di {acconto} euro. Chiedi al cliente se vuole continuare "
                    f"e poi richiama il tool con conferma_perdita_acconto=True."
                )

            # aggiorno lo stato nel database
            conn.execute(
                "UPDATE prenotazioni SET stato = 'annullata' WHERE id = ?",
                (p["id"],)
            )
            conn.commit()

            if perde_acconto:
                return f"Prenotazione {p['id']} annullata. L'acconto non viene restituito."
            return f"Prenotazione {p['id']} annullata con successo."

        except Exception as e:
            conn.rollback()
            return f"Errore durante l'operazione: {str(e)}"