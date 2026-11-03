from src.database.connection import get_connection

def verifica_disponibilita(data: str, turno_id: int, persone: int):
    conn = get_connection()
    try:
        cursor = conn.execute("""
        SELECT t.id, t.posti, t.zona
        FROM tavoli as t
        WHERE t.posti >= ?
            AND NOT EXISTS(
            SELECT 1
            FROM prenotazioni as p
            WHERE p.tavolo_id = t.id
            AND p.data = ?
            AND p.turno_id = ?
            AND p.stato <> 'disdetta'
            )
        ORDER BY t.posti, t.id
        """, (persone, data, turno_id))

        return cursor.fetchall()
    finally:
        conn.close()

def trova_o_crea_cliente(nome: str, telefono: str, email: str = None) -> int:
    """Cerca un cliente tramite numero di telefono o ne crea uno nuovo."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM clienti WHERE telefono = ?", (telefono,))
        row = cursor.fetchone()
        if row:
            return row[0]
        
        cursor.execute(
            "INSERT INTO clienti (nome, telefono, email) VALUES (?, ?, ?)",
            (nome, telefono, email)
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()

def inserisci_prenotazione(cliente_id: int, tavolo_id: int, turno_id: int, data: str, persone: int, note: str = "", segnala_celiachia: int = 0, acconto_versato: float = 0.0) -> int:
    """Inserisce una nuova prenotazione nel database."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO prenotazioni (cliente_id, tavolo_id, turno_id, data, persone, note, segnala_celiachia, acconto_versato, stato)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'confermata')
        """, (cliente_id, tavolo_id, turno_id, data, persone, note, segnala_celiachia, acconto_versato))
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()

from langchain_core.tools import tool

@tool
def controlla_disponibilita_tavoli(data: str, turno_id: int, persone: int) -> str:
    """Verifica la disponibilità di tavoli per una specifica data (YYYY-MM-DD), turno (1=Pranzo, 2=Cena) e numero di persone (da 1 a 12)."""
    tavoli = verifica_disponibilita(data, turno_id, persone)
    if not tavoli:
        return f"Nessun tavolo disponibile per {persone} persone in data {data} per il turno {turno_id}."
    tavolo_scelto = tavoli[0]
    return f"Disponibilità confermata. Tavolo idoneo ID: {tavolo_scelto[0]} in zona {tavolo_scelto[2]} ({tavolo_scelto[1]} posti)."

@tool
def registra_prenotazione(nome: str, telefono: str, data: str, turno_id: int, persone: int, email: str = None, note: str = "", celiachia: bool = False) -> str:
    """Registra una nuova prenotazione nel DB. Richiede nome, telefono, data (YYYY-MM-DD), turno_id (1=Pranzo, 2=Cena) e numero persone."""
    tavoli = verifica_disponibilita(data, turno_id, persone)
    if not tavoli:
        return "Impossibile completare la prenotazione: non ci sono tavoli liberi per i dati specificati."
    
    tavolo_id = tavoli[0][0]
    cliente_id = trova_o_crea_cliente(nome, telefono, email)
    
    # Calcolo dell'acconto per gruppi > 8 persone secondo il regolamento memorizzato nello schema
    acconto = (persone * 10.0) if persone > 8 else 0.0
    segnala_celiachia = 1 if celiachia else 0
    
    prenotazione_id = inserisci_prenotazione(
        cliente_id=cliente_id,
        tavolo_id=tavolo_id,
        turno_id=turno_id,
        data=data,
        persone=persone,
        note=note,
        segnala_celiachia=segnala_celiachia,
        acconto_versato=acconto
    )
    
    info_acconto = f" È richiesto un acconto di {acconto}€ (gruppo > 8 persone)." if acconto > 0 else ""
    return f"Prenotazione #{prenotazione_id} registrata con successo per {nome}! Tavolo assegnato #{tavolo_id}.{info_acconto}"