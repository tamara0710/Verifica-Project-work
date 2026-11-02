import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "trattoria.db"

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def crea_tabelle_complete():
    conn = get_connection()
    cursor = conn.cursor()

    # --- TABELLE GIÀ ESISTENTI (Estese se necessario) ---
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS clienti (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nome TEXT NOT NULL,
        telefono TEXT NOT NULL,
        email TEXT
    )""")

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS tavoli (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        posti INTEGER NOT NULL CHECK (posti > 0),
        zona TEXT NOT NULL CHECK (zona IN ('Interno', 'Dehors')) -- Interno (40), Dehors (20)
    )""")

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS turni (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nome TEXT NOT NULL,
        orario TEXT NOT NULL
    )""")

    # --- NUOVE TABELLE PER IL MENÙ ---
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS categorie_menu (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nome TEXT NOT NULL UNIQUE -- 'Antipasti', 'Primi', 'Secondi', 'Dolci'
    )""")

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS piatti (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        categoria_id INTEGER NOT NULL,
        nome TEXT NOT NULL,
        prezzo REAL NOT NULL CHECK (prezzo >= 0),
        allergeni TEXT, -- Es: 'latte, glutine'
        is_vegano INTEGER NOT NULL DEFAULT 0, -- 1 se vegano, 0 altrimenti
        is_senza_glutine INTEGER NOT NULL DEFAULT 0, -- 1 se senza glutine
        FOREIGN KEY (categoria_id) REFERENCES categorie_menu(id) ON DELETE RESTRICT
    )""")

    # --- PRENOTAZIONI AGGIORNATE (con Acconto e Celiachia) ---
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS prenotazioni (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        cliente_id INTEGER NOT NULL,
        tavolo_id INTEGER NOT NULL,
        turno_id INTEGER NOT NULL,
        data TEXT NOT NULL,
        persone INTEGER NOT NULL CHECK (persone BETWEEN 1 AND 12), -- Max 12 come da regolamento
        note TEXT,
        segnala_celiachia INTEGER NOT NULL DEFAULT 0, -- 1 se celiaco (supplemento 2€ per pasta SG)
        acconto_versato REAL DEFAULT 0.0, -- Richiesto 10€/persona per > 8 persone
        stato TEXT NOT NULL DEFAULT 'confermata', -- 'confermata', 'disdetta', 'in_attesa_acconto'
        FOREIGN KEY (cliente_id) REFERENCES clienti(id) ON DELETE CASCADE,
        FOREIGN KEY (tavolo_id) REFERENCES tavoli(id) ON DELETE RESTRICT,
        FOREIGN KEY (turno_id) REFERENCES turni(id) ON DELETE RESTRICT
    )""")

    # Indici
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_prenotazioni_data_turno ON prenotazioni(data, turno_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_piatti_categoria ON piatti(categoria_id)")

    conn.commit()
    conn.close()
    print("Schema esteso con successo!")

crea_tabelle_complete()