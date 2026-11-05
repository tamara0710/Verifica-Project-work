"""Database SQLite della trattoria (tavoli, turni, clienti, menù, prenotazioni).

crea_tabelle_complete() si può chiamare a ogni avvio: è idempotente e aggiorna
anche i database creati con versioni precedenti (colonne e dati mancanti).
"""
import sqlite3
from contextlib import closing
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "trattoria.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=20.0, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = DELETE;")   # niente file -wal: il .db contiene sempre tutte le prenotazioni
    return conn


def _aggiungi_colonna_se_manca(cur, tabella: str, colonna: str, definizione: str) -> None:
    colonne = [r[1] for r in cur.execute(f"PRAGMA table_info({tabella})").fetchall()]
    if colonna not in colonne:
        cur.execute(f"ALTER TABLE {tabella} ADD COLUMN {colonna} {definizione}")


def crea_tabelle_complete() -> None:
    # closing() chiude davvero la connessione (il solo "with conn" fa solo commit)
    with closing(get_connection()) as conn:
        cur = conn.cursor()

        cur.execute("""CREATE TABLE IF NOT EXISTS clienti (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nome TEXT NOT NULL,
            telefono TEXT NOT NULL,
            email TEXT)""")
        cur.execute("""CREATE TABLE IF NOT EXISTS tavoli (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            posti INTEGER NOT NULL CHECK (posti > 0),
            zona TEXT NOT NULL CHECK (zona IN ('Interno', 'Dehors')))""")
        cur.execute("""CREATE TABLE IF NOT EXISTS turni (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nome TEXT NOT NULL,
            orario TEXT NOT NULL)""")

        for tabella in ("antipasti", "primi", "secondi", "dolci"):
            cur.execute(f"""CREATE TABLE IF NOT EXISTS {tabella} (
                codice TEXT PRIMARY KEY,
                nome TEXT NOT NULL,
                prezzo REAL NOT NULL CHECK (prezzo >= 0),
                allergeni TEXT NOT NULL)""")

        cur.execute("""CREATE TABLE IF NOT EXISTS prenotazioni (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cliente_id INTEGER NOT NULL,
            tavolo_id INTEGER NOT NULL,
            turno_id INTEGER NOT NULL,
            data TEXT NOT NULL,
            persone INTEGER NOT NULL CHECK (persone BETWEEN 1 AND 12),
            note TEXT,
            segnala_celiachia INTEGER NOT NULL DEFAULT 0,
            acconto_versato REAL DEFAULT 0.0,
            stato TEXT NOT NULL DEFAULT 'confermata',
            FOREIGN KEY (cliente_id) REFERENCES clienti(id) ON DELETE CASCADE,
            FOREIGN KEY (tavolo_id) REFERENCES tavoli(id) ON DELETE RESTRICT,
            FOREIGN KEY (turno_id) REFERENCES turni(id) ON DELETE RESTRICT)""")

        # --- migrazione di database creati con schemi più vecchi ---
        _aggiungi_colonna_se_manca(cur, "clienti", "email", "TEXT")
        _aggiungi_colonna_se_manca(cur, "prenotazioni", "segnala_celiachia", "INTEGER NOT NULL DEFAULT 0")
        _aggiungi_colonna_se_manca(cur, "prenotazioni", "acconto_versato", "REAL DEFAULT 0.0")

        # --- menù (dal PDF) ---
        cur.executemany("INSERT OR IGNORE INTO antipasti VALUES (?,?,?,?)", [
            ("A01", "Tagliere di salumi e formaggi pugliesi", 14.0, "latte"),
            ("A02", "Burrata con pomodorini e crostini", 10.0, "latte, glutine"),
            ("A03", "Polpo arrosto su crema di fave", 13.0, "molluschi"),
            ("A04", "Verdure grigliate (vegano)", 8.0, "-")])
        cur.executemany("INSERT OR IGNORE INTO primi VALUES (?,?,?,?)", [
            ("P01", "Orecchiette alle cime di rapa", 11.0, "glutine, pesce (acciughe)"),
            ("P02", "Spaghetti all'assassina", 12.0, "glutine"),
            ("P03", "Cavatelli con cozze e fagioli", 14.0, "glutine, molluschi"),
            ("P04", "Risotto ai funghi cardoncelli (senza glutine)", 13.0, "latte")])
        cur.executemany("INSERT OR IGNORE INTO secondi VALUES (?,?,?,?)", [
            ("S01", "Bombette di Cisternino", 15.0, "latte"),
            ("S02", "Orata al forno con patate", 18.0, "pesce"),
            ("S03", "Polpette al sugo della nonna", 13.0, "uova, glutine, latte"),
            ("S04", "Parmigiana di melanzane (vegetariano)", 12.0, "latte, uova, glutine")])
        cur.executemany("INSERT OR IGNORE INTO dolci VALUES (?,?,?,?)", [
            ("D01", "Pasticciotto leccese", 5.0, "glutine, uova, latte"),
            ("D02", "Tiramisù", 6.0, "uova, latte, glutine"),
            ("D03", "Sorbetto al limone (vegano)", 4.0, "-")])

        # --- turni: il PDF ne prevede 4 (gli id 1 e 2 restano compatibili col DB vecchio) ---
        cur.executemany("INSERT OR IGNORE INTO turni (id, nome, orario) VALUES (?,?,?)", [
            (1, "Pranzo", "12:30"),
            (2, "Cena", "19:30"),
            (3, "Pranzo", "14:00"),
            (4, "Cena", "21:30")])

        # --- tavoli: sala interna 40 coperti, dehors 20 coperti (include tavoli da 12 e 8
        #     posti, altrimenti i gruppi da 9 a 12 persone non potrebbero mai prenotare) ---
        cur.executemany("INSERT OR IGNORE INTO tavoli (id, posti, zona) VALUES (?,?,?)", [
            (1, 4, "Interno"), (2, 2, "Interno"), (3, 6, "Interno"), (5, 8, "Interno"),
            (6, 12, "Interno"), (7, 4, "Interno"), (8, 4, "Interno"),      # interno = 40
            (4, 4, "Dehors"), (9, 2, "Dehors"), (10, 4, "Dehors"), (11, 4, "Dehors"),
            (12, 6, "Dehors")])                                            # dehors = 20

        cur.execute("CREATE INDEX IF NOT EXISTS idx_prenotazioni_data_turno ON prenotazioni(data, turno_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_prenotazioni_tavolo_data ON prenotazioni(tavolo_id, data)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_clienti_telefono ON clienti(telefono)")
        conn.commit()


if __name__ == "__main__":
    crea_tabelle_complete()
    print(f"Database pronto: {DB_PATH}")
