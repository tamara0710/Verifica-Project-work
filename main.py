"""
Entry point per l'interfaccia utente da Terminale (CLI).
"""
import re
import sqlite3
from typing import Optional

from langgraph.types import Command

from graph import DB_IN_USO, chiudi_connessioni, graph, store
from src.database.repositories import normalizza_telefono

# ==========================================================
# GESTIONE INTERFACCIA TERMINALE
# ==========================================================
PAROLE_SI = {"si", "sì", "s", "yes", "y", "ok", "okay", "certo", "confermo", "conferma", "confermato",
             "va bene", "procedi", "esatto", "perfetto", "giusto"}
PAROLE_NO = {"no", "n", "nope", "annulla", "annullo", "stop", "lascia stare", "non confermo", "sbagliato", "errato"}


def interpreta_conferma(testo: str) -> Optional[bool]:
    t = re.sub(r"[^\wàèéìòù ]", " ", testo.lower()).strip()
    parole = t.split()
    if not parole:
        return None
    if t in PAROLE_NO or parole[0] in ("no", "non", "annulla", "stop") or any(p in ("no", "annulla", "annullo") for p in parole):
        return False
    if t in PAROLE_SI or parole[0] in PAROLE_SI or any(p in ("confermo", "conferma", "confermato") for p in parole):
        return True
    return None


def chiedi_conferma() -> bool:
    while True:
        esito = interpreta_conferma(input("\n👉 Scrivi 'si' per confermare o 'no' per annullare: "))
        if esito is not None:
            return esito
        print("   Non ho capito: rispondi 'si' oppure 'no'.")


def mostra_ultime_prenotazioni(n: int = 8) -> None:
    conn = sqlite3.connect(str(DB_IN_USO), timeout=20.0)
    try:
        righe = conn.execute("""
            SELECT p.id, c.nome, c.telefono, p.data, t.nome || ' ' || t.orario, p.persone, p.stato
            FROM prenotazioni p JOIN clienti c ON c.id = p.cliente_id JOIN turni t ON t.id = p.turno_id
            ORDER BY p.id DESC LIMIT ?""", (n,)).fetchall()
    finally:
        conn.close()
    print(f"\n📒 Ultime prenotazioni nel database ({DB_IN_USO}):")
    for r in righe:
        print(f"   #{r[0]} | {r[1]} | tel {r[2]} | {r[3]} {r[4]} | {r[5]} pers. | {r[6]}")
    if not righe:
        print("   (nessuna prenotazione)")
    print()


def mostra_richiesta_conferma(dati) -> None:
    print("\n" + "=" * 55)
    print("⚠️  RICHIESTA DI CONFERMA")
    print("=" * 55)
    if isinstance(dati, dict):
        for k, v in dati.items():
            if k != "domanda":
                print(f"   • {k}: {v}")
        print(f"\n{dati.get('domanda', 'Confermi?')}")
    else:
        print(dati)


def stampa_risposta(res) -> None:
    if res and res.get("risposta"):
        print(f"\n🤖 Assistente:\n{res['risposta']}\n")


def avvia_chatbot():
    print("\n" + "=" * 65)
    print(" 🍷 BENVENUTO NEL CHATBOT DI TRATTORIA NONNA ROSA")
    print("=" * 65)
    print(" Comandi: help | nuovo | utente | stato | esci")
    print("=" * 65 + "\n")

    session_count = 1
    telefono_corrente = normalizza_telefono(input("📱 Inserisci il tuo numero di telefono (es. 347557777): "))
    if len(telefono_corrente) < 6:
        telefono_corrente = "347557777"
        print(f"👉 Usato numero predefinito: {telefono_corrente}")

    thread_id = f"sessione_{telefono_corrente}_{session_count}"
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        try:
            stato = graph.get_state(config)
            if stato.next and stato.tasks and getattr(stato.tasks[0], "interrupts", None):
                mostra_richiesta_conferma(stato.tasks[0].interrupts[0].value)
                stampa_risposta(graph.invoke(Command(resume=chiedi_conferma()), config=config))
                continue

            user_input = input(f"👤 [{telefono_corrente} @ {thread_id}] > ").strip()
            if not user_input:
                continue
            cmd = user_input.lower()

            if cmd in ("esci", "exit", "quit"):
                print("\n👋 Grazie per aver contattato la Trattoria Nonna Rosa. Arrivederci!")
                break
            elif cmd in ("nuovo", "reset"):
                session_count += 1
                thread_id = f"sessione_{telefono_corrente}_{session_count}"
                config = {"configurable": {"thread_id": thread_id}}
                print(f"\n🔄 Nuova sessione: '{thread_id}' (le allergie restano nello Store)\n")
                continue
            elif cmd == "utente":
                nuovo = normalizza_telefono(input("📱 Nuovo numero di telefono: "))
                if len(nuovo) >= 6:
                    telefono_corrente, session_count = nuovo, 1
                    thread_id = f"sessione_{telefono_corrente}_{session_count}"
                    config = {"configurable": {"thread_id": thread_id}}
                    print(f"\n👤 Utente: {telefono_corrente} (thread {thread_id})\n")
                else:
                    print("⚠️ Numero non valido.\n")
                continue
            elif cmd == "stato":
                item = store.get(("clienti", "allergie"), telefono_corrente)
                print(f"\n📌 Telefono: {telefono_corrente} | Thread: {thread_id}")
                print(f"   Memoria Store: {item.value if item and item.value else 'nessuna allergia salvata'}\n")
                continue
            elif cmd in ("db", "prenotazioni"):
                mostra_ultime_prenotazioni()
                continue
            elif cmd in ("help", "aiuto"):
                print("\n  nuovo = nuova chat | utente = cambia telefono | stato = mostra memoria | esci = chiudi\n")
                continue

            stampa_risposta(graph.invoke({"domanda": user_input, "telefono": telefono_corrente}, config=config))

        except KeyboardInterrupt:
            print("\n\n👋 Chiusura del chatbot...")
            break
        except Exception as e:
            print(f"\n❌ Errore: {e}\n")

    chiudi_connessioni()


if __name__ == "__main__":
    avvia_chatbot()