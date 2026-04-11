#!/usr/bin/env python
# coding: utf-8

# In[3]:


#!/usr/bin/env python
# coding: utf-8

# ─────────────────────────────────────────────────────────────
# PACKAGES
# ─────────────────────────────────────────────────────────────
import os
import json
import requests
import pandas as pd


# ─────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/chat"
MODELLO = "phi3:mini"
PAROLA_MAGICA = "yes"
OUTPUT_JSON = "context_memory.json"


# ─────────────────────────────────────────────────────────────
# 2. CSV READER
# ─────────────────────────────────────────────────────────────

def leggi_csv(filepath: str) -> str:
    if not os.path.exists(filepath):
        print(f"  [WARN] File not found: {filepath} — skipped.")
        return ""

    df = pd.read_csv(filepath, sep=";")
    return df.to_string(index=False)


# ─────────────────────────────────────────────────────────────
# 3. OLLAMA HTTP CALL
# ─────────────────────────────────────────────────────────────

def chiedi_a_ollama(cronologia: list, llm: str, ollama_url: str) -> str:
    payload = {
        "model": llm,
        "messages": cronologia,
        "stream": False,
        "options": {
            "temperature": 0.5,
            "num_predict": 800
        }
    }

    try:
        risposta = requests.post(ollama_url, json=payload, timeout=5000)
        risposta.raise_for_status()
        dati = risposta.json()
        return dati["message"]["content"]

    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            "\n[ERROR] Ollama is not running!\n"
            "  Start it with: ollama serve\n"
        )
    except requests.exceptions.Timeout:
        raise TimeoutError(
            "\n[ERROR] Timeout: the model took too long to respond.\n"
        )


# ─────────────────────────────────────────────────────────────
# 4. INITIAL PROMPT BUILDER
# ─────────────────────────────────────────────────────────────

def costruisci_prompt_iniziale(
    general_ctx: str,
    business_ctx: str,
    features_ctx: str
) -> str:

    prompt = f"""
You are an expert agent specialized in human context analysis and data interpretation.
You are provided with three levels of contextual information extracted from CSV files.
Your task is to deeply understand these inputs and produce a structured, comprehensive summary that highlights the key insights, 
relationships, and implications for the business environment.

---
## CONTESTO GENERALE
{general_ctx if general_ctx else "Non fornito."}

---
## CONTESTO DI BUSINESS
{business_ctx if business_ctx else "Non fornito."}

---
## CONTESTO DELLE FEATURES E CONCETTI LATENTI
{features_ctx if features_ctx else "Non fornito."}

---
Based on all provided information, deliver the following:

Agent Role
Define the role the agent should assume to provide maximum value.
Focus on analytical, interpretive, and decision‑support capabilities.

General Context
Summarize the overall domain and environment.
Highlight the macro‑level dynamics, actors, and constraints.

Business Context
Describe key objectives, processes, KPIs, and business logic relevant to the scenario.
Identify strategic priorities and operational drivers.

Purpose
Specify what the agent/system must concretely accomplish.
Don't create profile/cluster names (next agent will do it)
Clarify expected outputs, decisions, or transformations.
Included Features
Explain what each variable measures and what latent concepts they represent.
Highlight how these features relate to the business problem.

Be precise, concise, and structured. Use bullet points where appropriate.
"""
    return prompt.strip()



# ─────────────────────────────────────────────────────────────
# 5. JSON STRUCTURE EXTRACTOR
# ─────────────────────────────────────────────────────────────

def estrai_struttura_json(risposta_finale: str, llm: str, ollama_url: str) -> dict:

    prompt_json = f"""
From the following analysis, extract a JSON object with the exact structure shown below.
Respond ONLY with valid JSON. And all insights and description should be prompt-designed for the next LLM agent.
No markdown, no comments, no additional text.

{{
  "role": "stringa",
  "general_context": "stringa",
  "business_context": "stringa",
  "scope": "stringa",
  "features_and_concepts": ["stringa", "stringa"]
  "other": "stringa",
}}

Analisi da cui estrarre:
{risposta_finale}
"""

    messaggi = [{"role": "user", "content": prompt_json}]
    testo = chiedi_a_ollama(messaggi, llm=llm, ollama_url=ollama_url)

    # Pulizia: rimuove eventuali backtick markdown che il modello aggiunge
    testo_pulito = testo.strip().strip("```json").strip("```").strip()

    try:
        struttura = json.loads(testo_pulito)
    except json.JSONDecodeError:
        # Fallback sicuro: salva la risposta grezza senza crashare
        print("  [WARN] Il modello non ha restituito JSON valido. Salvo risposta grezza.")
        struttura = {
            "role": "Non estratto automaticamente",
            "general_context": "Non estratto automaticamente",
            "business_context": "Non estratto automaticamente",
            "scope": "Non estratto automaticamente",
            "features_and_concepts": [],
            "other": "Non estratto automaticamente",
            "risposta_completa": risposta_finale
        }

    return struttura


# ─────────────────────────────────────────────────────────────
# 6. MEMORY PERSISTENCE
# ─────────────────────────────────────────────────────────────


def salva_memoria(struttura: dict, filepath: str = OUTPUT_JSON):
    """
    Scrive la struttura appresa su disco in formato JSON leggibile.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(struttura, f, ensure_ascii=False, indent=2)
    print(f"\n Memoria salvata in: {filepath}")

# ─────────────────────────────────────────────────────────────
# 7. MAIN AGENT LOOP
# ─────────────────────────────────────────────────────────────

def execute_agent_context(
    work_directory: str,
    general_csv: str = "general_context.csv",
    business_csv: str = "business_context.csv",
    features_csv: str = "features_context.csv",
    llm: str = MODELLO,
    ollama_url: str = OLLAMA_URL
) -> dict:

    os.chdir(work_directory)

    print(f"Model: {llm}")

    # — STEP 1: Leggi i CSV —
    print("\n Lettura dei file CSV...")
    general_ctx = leggi_csv(general_csv)
    business_ctx = leggi_csv(business_csv)
    features_ctx = leggi_csv(features_csv)

     # — STEP 2: Primo prompt —
    print(" Invio contesto al modello (potrebbe richiedere qualche secondo)...\n")
    prompt_iniziale = costruisci_prompt_iniziale(
        general_ctx, business_ctx, features_ctx
    )

    cronologia = [{"role": "user", "content": prompt_iniziale}]
    # — STEP 3: Prima risposta —
    risposta = chiedi_a_ollama(cronologia, llm=llm, ollama_url=ollama_url)
    cronologia.append({"role": "assistant", "content": risposta})

    # — STEP 4: Loop di feedback interattivo —

    while True:
        print("\n" + "─" * 60)
        print(" COMPRENSIONE CORRENTE DELL'AGENTE:\n")
        print(risposta)
        print("\n" + "─" * 60)
        print(f"\n Inserisci feedback per affinare (o '{PAROLA_MAGICA}' per confermare):\n")

        feedback = input("You -> ").strip()

                # Parola magica: esce dal loop
        if feedback.lower() == PAROLA_MAGICA:
            print("\n Comprensione confermata! Salvataggio in corso...")
            break

        if not feedback:
            print("  [INFO] Nessun testo inserito, riprova.")
            continue

        cronologia.append({"role": "user", "content": feedback})

        risposta = chiedi_a_ollama(cronologia[-3:], llm=llm, ollama_url=ollama_url)
        cronologia.append({"role": "assistant", "content": risposta})

    struttura = estrai_struttura_json(risposta, llm=llm, ollama_url=ollama_url)
    salva_memoria(struttura)

    return struttura


# In[ ]:





# In[ ]:




