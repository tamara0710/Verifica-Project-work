from .connection import get_connection

def cerca_tavoli_quattro_posti():
    conn = get_connection()

    try:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, posti, zona
            FROM tavoli
            WHERE posti = 4
        """)

        return cursor.fetchall()
    finally:
        conn.close()

def verifica_disponibilita(tavolo_id, persone, data, turno_id, note):
    conn = get_connection()

    try:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, posti, zona
            FROM tavoli
            WHERE id = ?
              AND posti >= ?
              AND NOT EXISTS (
                  SELECT 1
                  FROM prenotazioni
                  WHERE prenotazioni.tavolo_id = tavoli.id
                    AND prenotazioni.data = ?
                    AND prenotazioni.turno_id = ?
                    AND prenotazioni.stato != 'disdetta'
              )
        """, (tavolo_id, persone, data, turno_id))

        tavolo_disponibile = cursor.fetchone()
    finally:
        conn.close()

    if tavolo_disponibile is None:
        print("Tavolo non disponibile per questa richiesta")
    else:
        print("Tavolo disponibile:", tavolo_disponibile)

    print("Preferenza richiesta:", note)

verifica_disponibilita(
    1,
    2,
    "2026-11-10",
    1,
    "Preferisce un tavolo vicino alla finestra"
)

def cerca_tavolo_disponibile(persone, data, turno_id):
    conn = get_connection()

    try:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, posti, zona
            FROM tavoli
            WHERE posti >= ?
              AND NOT EXISTS (
                  SELECT 1
                  FROM prenotazioni
                  WHERE prenotazioni.tavolo_id = tavoli.id
                    AND prenotazioni.data = ?
                    AND prenotazioni.turno_id = ?
                    AND prenotazioni.stato != 'disdetta'
              )
            ORDER BY posti ASC, id ASC
            LIMIT 1
        """, (persone, data, turno_id))

        return cursor.fetchone()
    finally:
        conn.close()

if __name__ == "__main__":
    tavolo = cerca_tavolo_disponibile(2, "2026-11-10", 1)

    if tavolo is None:
        print("Nessun tavolo disponibile.")
    else:
        print(
            f"Tavolo {tavolo[0]} disponibile: "
            f"{tavolo[1]} posti, zona {tavolo[2]}."
        )