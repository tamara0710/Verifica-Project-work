PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS tavoli (
    id INTEGER PRIMARY KEY,
    posti INTEGER NOT NULL CHECK (posti > 0),
    zona TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS turni (
    id INTEGER PRIMARY KEY,
    nome TEXT NOT NULL,
    orario TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS clienti (
    id INTEGER PRIMARY KEY,
    nome TEXT NOT NULL,
    telefono TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS prenotazioni (
    id INTEGER PRIMARY KEY,
    cliente_id INTEGER NOT NULL,
    tavolo_id INTEGER NOT NULL,
    turno_id INTEGER NOT NULL,
    data TEXT NOT NULL,
    persone INTEGER NOT NULL CHECK (persone BETWEEN 1 AND 12),
    note TEXT,
    stato TEXT NOT NULL DEFAULT 'confermata',
    FOREIGN KEY (cliente_id) REFERENCES clienti(id),
    FOREIGN KEY (tavolo_id) REFERENCES tavoli(id),
    FOREIGN KEY (turno_id) REFERENCES turni(id)
);

CREATE INDEX IF NOT EXISTS idx_prenotazioni_data_turno
    ON prenotazioni(data, turno_id);

CREATE INDEX IF NOT EXISTS idx_prenotazioni_tavolo_data
    ON prenotazioni(tavolo_id, data);