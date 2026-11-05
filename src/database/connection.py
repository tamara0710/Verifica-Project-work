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

        cur.execute("""
            INSERT OR IGNORE INTO clienti (id, nome, telefono, email) VALUES
            (1, 'Mario Rossi', '3331234567', 'mario.rossi@email.com'),
            (2, 'Giulia Bianchi', '3382345678', 'giulia.bianchi@email.com'),
            (3, 'Luca Verdi', '3403456789', 'luca.verdi@email.com'),
            (4, 'Francesca Russo', '3474567890', 'francesca.russo@email.com'),
            (5, 'Alessandro Ferrari', '3205678901', 'alessandro.ferrari@email.com'),
            (6, 'Elena Esposito', '3396789012', 'elena.esposito@email.com'),
            (7, 'Marco Romano', '3287890123', 'marco.romano@email.com'),
            (8, 'Sara Gallo', '3318901234', 'sara.gallo@email.com'),
            (9, 'Davide Costa', '3459012345', 'davide.costa@email.com'),
            (10, 'Chiara Fontana', '3660123456', 'chiara.fontana@email.com');        
            """)


        # Popolamento completo: 12 tavoli x 4 turni = 48 prenotazioni complessive
        cur.execute("""
            INSERT OR IGNORE INTO prenotazioni (id, cliente_id, tavolo_id, turno_id, data, persone, note, segnala_celiachia, acconto_versato, stato) VALUES
            -- ============================================================
            -- TURN 1: Pranzo 12:30 (id_turno = 1)
            -- ============================================================
            (1, 1, 1, 1, '2026-11-10', 4, 'Preferenza vicino alla finestra', 0, 0.00, 'confermata'),
            (2, 2, 2, 1, '2026-11-10', 2, NULL, 0, 0.00, 'confermata'),
            (3, 7, 3, 1, '2026-11-10', 5, 'Pranzo di lavoro', 0, 0.00, 'confermata'),
            (4, 3, 4, 1, '2026-11-10', 3, 'Tavolo fuori', 1, 10.00, 'confermata'),
            (5, 4, 5, 1, '2026-11-10', 6, 'Seggiolone per bambino', 0, 20.00, 'confermata'),
            (6, 6, 6, 1, '2026-11-10', 10, 'Evento aziendale', 0, 50.00, 'confermata'),
            (7, 5, 7, 1, '2026-11-10', 4, NULL, 0, 0.00, 'in attesa'),
            (8, 8, 8, 1, '2026-11-10', 4, 'Cliente abituale', 0, 0.00, 'confermata'),
            (9, 9, 9, 1, '2026-11-10', 2, 'Dehors soleggiato', 0, 0.00, 'confermata'),
            (10, 10, 10, 1, '2026-11-10', 4, NULL, 0, 0.00, 'confermata'),
            (11, 1, 11, 1, '2026-11-10', 3, NULL, 1, 0.00, 'confermata'),
            (12, 2, 12, 1, '2026-11-10', 6, 'Famiglia con passeggino', 0, 15.00, 'confermata'),

            -- ============================================================
            -- TURN 2: Pranzo 14:00 (id_turno = 2)
            -- ============================================================
            (13, 6, 1, 2, '2026-11-10', 3, NULL, 0, 0.00, 'confermata'),
            (14, 3, 2, 2, '2026-11-10', 2, 'Pranzo veloce', 0, 0.00, 'confermata'),
            (15, 7, 3, 2, '2026-11-10', 5, 'Compleanno in famiglia', 1, 25.00, 'confermata'),
            (16, 4, 4, 2, '2026-11-10', 4, 'Vicino all ingresso dehors', 0, 0.00, 'confermata'),
            (17, 5, 5, 2, '2026-11-10', 7, 'Riunione team', 0, 30.00, 'confermata'),
            (18, 1, 6, 2, '2026-11-10', 11, 'Tavolata festeggiamenti', 0, 60.00, 'confermata'),
            (19, 2, 7, 2, '2026-11-10', 4, NULL, 0, 0.00, 'in attesa'),
            (20, 8, 8, 2, '2026-11-10', 4, NULL, 0, 0.00, 'confermata'),
            (21, 9, 9, 2, '2026-11-10', 2, 'Coppia all aperto', 0, 0.00, 'confermata'),
            (22, 10, 10, 2, '2026-11-10', 4, NULL, 0, 0.00, 'annullata'),
            (23, 7, 11, 2, '2026-11-10', 4, 'Zona Dehors tranquilla', 0, 0.00, 'confermata'),
            (24, 8, 12, 2, '2026-11-10', 5, 'Menu degustazione', 1, 20.00, 'confermata'),

            -- ============================================================
            -- TURN 3: Cena 19:30 (id_turno = 3)
            -- ============================================================
            (25, 1, 1, 3, '2026-11-10', 4, 'Tavolo tranquillo', 0, 0.00, 'confermata'),
            (26, 2, 2, 3, '2026-11-10', 2, NULL, 1, 10.00, 'confermata'),
            (27, 4, 3, 3, '2026-11-10', 6, 'Cena di famiglia', 0, 0.00, 'confermata'),
            (28, 8, 4, 3, '2026-11-10', 4, 'Dehors riscaldato', 0, 0.00, 'confermata'),
            (29, 3, 5, 3, '2026-11-10', 8, 'Gruppo amici - Max capienza', 0, 40.00, 'confermata'),
            (30, 9, 6, 3, '2026-11-10', 9, 'Compleanno 30 anni', 1, 50.00, 'confermata'),
            (31, 4, 7, 3, '2026-11-10', 4, NULL, 0, 0.00, 'confermata'),
            (32, 10, 8, 3, '2026-11-10', 3, NULL, 0, 0.00, 'confermata'),
            (33, 7, 9, 3, '2026-11-10', 2, 'Anniversario', 0, 15.00, 'confermata'),
            (34, 6, 10, 3, '2026-11-10', 4, NULL, 0, 0.00, 'in attesa'),
            (35, 5, 11, 3, '2026-11-10', 4, 'Zona Dehors', 0, 0.00, 'confermata'),
            (36, 6, 12, 3, '2026-11-10', 6, 'Richiesta torta', 1, 30.00, 'confermata'),

            -- ============================================================
            -- TURN 4: Cena 21:30 (id_turno = 4)
            -- ============================================================
            (37, 1, 1, 4, '2026-11-10', 4, 'Cena tarda post-cinema', 0, 0.00, 'confermata'),
            (38, 5, 2, 4, '2026-11-10', 2, NULL, 0, 0.00, 'confermata'),
            (39, 7, 3, 4, '2026-11-10', 6, NULL, 0, 0.00, 'confermata'),
            (40, 8, 4, 4, '2026-11-10', 4, 'Dehors riscaldato', 0, 0.00, 'confermata'),
            (41, 2, 5, 4, '2026-11-10', 7, 'Cena di lavoro', 0, 35.00, 'confermata'),
            (42, 6, 6, 4, '2026-11-10', 12, 'Tavolata laurea', 1, 60.00, 'confermata'),
            (43, 3, 7, 4, '2026-11-10', 4, NULL, 0, 0.00, 'confermata'),
            (44, 9, 8, 4, '2026-11-10', 4, NULL, 1, 20.00, 'confermata'),
            (45, 4, 9, 4, '2026-11-10', 2, NULL, 0, 0.00, 'confermata'),
            (46, 10, 10, 4, '2026-11-10', 3, 'Cliente in ritardo di 15 min', 0, 0.00, 'confermata'),
            (47, 1, 11, 4, '2026-11-10', 4, NULL, 0, 0.00, 'annullata'),
            (48, 2, 12, 4, '2026-11-10', 5, 'Richiesta amaro finale', 0, 0.00, 'confermata');
        """)

            # --- turni: il PDF ne prevede 4 (gli id 1 e 2 restano compatibili col DB vecchio) ---
        cur.executemany("INSERT OR IGNORE INTO turni (id, nome, orario) VALUES (?,?,?)", [
            (1, "Pranzo", "12:30"),
            (2, "Pranzo", "14:00"),
            (3, "Cena", "19:30"),
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
