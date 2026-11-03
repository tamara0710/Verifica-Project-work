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

    # --- TABELLE PER LE CATEGORIE DEL MENÙ ---
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS antipasti (
        codice TEXT PRIMARY KEY,
        nome TEXT NOT NULL,
        prezzo REAL NOT NULL CHECK (prezzo >= 0),
        allergeni TEXT NOT NULL
    )""")

    cursor.execute("""
    INSERT OR IGNORE INTO antipasti (codice, nome, prezzo, allergeni)
    VALUES
        ('A01', 'Tagliere di salumi e formaggi pugliesi', 14.00, 'latte'),
        ('A02', 'Burrata con pomodorini e crostini', 10.00, 'latte, glutine'),
        ('A03', 'Polpo arrosto su crema di fave', 13.00, 'molluschi'),
        ('A04', 'Verdure grigliate (vegano)', 8.00, '-')
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS primi (
        codice TEXT PRIMARY KEY,
        nome TEXT NOT NULL,
        prezzo REAL NOT NULL CHECK (prezzo >= 0),
        allergeni TEXT NOT NULL
    )""")

    cursor.execute("""
    INSERT OR IGNORE INTO primi (codice, nome, prezzo, allergeni)
    VALUES
        ('P01', 'Orecchiette alle cime di rapa', 11.00, 'glutine, pesce (acciughe)'),
        ('P02', 'Spaghetti all''assassina', 12.00, 'glutine'),
        ('P03', 'Cavatelli con cozze e fagioli', 14.00, 'glutine, molluschi'),
        ('P04', 'Risotto ai funghi cardoncelli (senza glutine)', 13.00, 'latte')
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS secondi (
        codice TEXT PRIMARY KEY,
        nome TEXT NOT NULL,
        prezzo REAL NOT NULL CHECK (prezzo >= 0),
        allergeni TEXT NOT NULL
    )
""")

    cursor.execute("""
    INSERT OR IGNORE INTO secondi (codice, nome, prezzo, allergeni)
    VALUES
        ('S01', 'Bombette di Cisternino', 15.00, 'latte'),
        ('S02', 'Orata al forno con patate', 18.00, 'pesce'),
        ('S03', 'Polpette al sugo della nonna', 13.00, 'uova, glutine, latte'),
        ('S04', 'Parmigiana di melanzane (vegetariano)', 12.00, 'latte, uova, glutine')
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS dolci (
        codice TEXT PRIMARY KEY,
        nome TEXT NOT NULL,
        prezzo REAL NOT NULL CHECK (prezzo >= 0),
        allergeni TEXT NOT NULL
    )""")

    cursor.execute("""
    INSERT OR IGNORE INTO dolci (codice, nome, prezzo, allergeni)
    VALUES
        ('D01', 'Pasticciotto leccese', 5.00, 'glutine, uova, latte'),
        ('D02', 'Tiramisù', 6.00, 'uova, latte, glutine'),
        ('D03', 'Sorbetto al limone (vegano)', 4.00, '-')
    """)

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

    # Inserimento dei turni validi
    cursor.execute("""
    INSERT OR IGNORE INTO turni (id, nome, orario)
    VALUES 
        (1, 'Pranzo', '12:30'),
        (2, 'Cena', '19:30')
    """)

    # Inserimento di un set di tavoli di esempio che rispettino la capienza (es. 40 interni, 20 dehors)
    cursor.execute("""
    INSERT OR IGNORE INTO tavoli (id, posti, zona)
    VALUES 
        (1, 4, 'Interno'),
        (2, 2, 'Interno'),
        (3, 6, 'Interno'),
        (4, 4, 'Dehors'),
        (5, 8, 'Interno')
    """)

    # Indici
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_prenotazioni_data_turno ON prenotazioni(data, turno_id)")

    conn.commit()
    conn.close()
    print("Schema esteso con successo!")

crea_tabelle_complete()