from .connection import get_connection

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
        conn.close