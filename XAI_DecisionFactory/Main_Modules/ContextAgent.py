#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

# Local Ollama endpoint (do not change unless you moved the server)
OLLAMA_URL = "http://localhost:11434/api/chat"

# Model to use — must already be pulled with: ollama pull <name>
MODELLO = "phi3:mini"   # swap with "llama3", "phi3", etc. if preferred

# Magic word that exits the interactive feedback loop
PAROLA_MAGICA = "yes"

# Output JSON file that stores the agent's extracted memory
OUTPUT_JSON = "context_memory.json"


# ─────────────────────────────────────────────────────────────
# 2. CSV READER
# ─────────────────────────────────────────────────────────────

def leggi_csv(filepath: str) -> str:
    """
    Reads a semicolon-delimited CSV file and converts it to a
    plain-text table suitable for injection into an LLM prompt.
    Returns an empty string if the file does not exist.
    """
    if not os.path.exists(filepath):
        print(f"  [WARN] File not found: {filepath} — skipped.")
        return ""

    df = pd.read_csv(filepath, sep=";")
    return df.to_string(index=False)


# ─────────────────────────────────────────────────────────────
# 3. OLLAMA HTTP CALL
# ─────────────────────────────────────────────────────────────

def chiedi_a_ollama(cronologia: list) -> str:
    """
    Sends the full conversation history to the local Ollama server
    and returns the model's plain-text reply.

    Ollama accepts the same message format as OpenAI:
      [{"role": "user" | "assistant" | "system", "content": "..."}]

    Parameters
    ----------
    cronologia : list of dicts
        Full conversation history accumulated so far.

    Returns
    -------
    str
        The text content of the model's response.

    Raises
    ------
    ConnectionError  if Ollama is not running.
    TimeoutError     if the model takes too long to respond.
    """
    payload = {
        "model": MODELLO,
        "messages": cronologia,   # BUG FIX: was referencing undefined 'messages'
        "stream": False,
        "options": {
            "temperature": 0.6,   # balanced: creative but precise
            "num_predict": 1000   # max response length in tokens
        }
    }

    try:
        risposta = requests.post(OLLAMA_URL, json=payload, timeout=5000)
        risposta.raise_for_status()
        dati = risposta.json()
        return dati["message"]["content"]

    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            "\n[ERROR] Ollama is not running!\n"
            "  Start it with: ollama serve\n"
            "  Or open the Ollama app from the system tray."
        )
    except requests.exceptions.Timeout:
        raise TimeoutError(
            "\n[ERROR] Timeout: the model took too long to respond.\n"
            "  Try a lighter model such as 'phi3:mini'."
        )


# ─────────────────────────────────────────────────────────────
# 4. INITIAL PROMPT BUILDER
# ─────────────────────────────────────────────────────────────

def costruisci_prompt_iniziale(
    general_ctx: str,
    business_ctx: str,
    features_ctx: str
) -> str:
    """
    Assembles the first user prompt by injecting all three CSV
    context layers into a structured analytical request.
    The prompt instructs the model to produce a comprehensive
    summary covering role, context, purpose, and feature semantics.

    Parameters
    ----------
    general_ctx  : str  — content from the general context CSV
    business_ctx : str  — content from the business context CSV
    features_ctx : str  — content from the features context CSV

    Returns
    -------
    str
        The fully assembled prompt string, ready to be sent.
    """
    prompt = f"""
You are an expert agent specialized in human context analysis and data interpretation.
You are provided with three levels of contextual information extracted from CSV files.
Your task is to deeply understand these inputs and produce a structured, comprehensive summary that highlights the key insights, 
relationships, and implications for the business environment.

---
## GENERAL CONTEXT
{general_ctx if general_ctx else "Not provided."}

---
## BUSINESS CONTEXT
{business_ctx if business_ctx else "Not provided."}

---
## FEATURES AND LATENT CONCEPTS
{features_ctx if features_ctx else "Not provided."}

---
Based on all provided information, deliver the following:

Agent Role
Define the role the agent should assume to provide maximum value.
Focus on analytical, interpretive, and decision-support capabilities.

General Context
Summarize the overall domain and environment.
Highlight the macro-level dynamics, actors, and constraints.

Business Context
Describe key objectives, processes, KPIs, and business logic relevant to the scenario.
Identify strategic priorities and operational drivers.

Purpose
Specify what the agent/system must concretely accomplish.
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

def estrai_struttura_json(risposta_finale: str) -> dict:
    """
    Asks the model to distill the final free-text analysis into a
    clean, well-defined JSON object that will be persisted as the
    agent's context memory for downstream LLM agents.

    The JSON schema is fixed; the model is instructed to return
    only valid JSON with no extra text or markdown fences.

    If the model returns malformed JSON, a safe fallback dict is
    returned so the pipeline never crashes.

    Parameters
    ----------
    risposta_finale : str
        The last free-text analysis produced by the agent.

    Returns
    -------
    dict
        Parsed JSON structure (or fallback dict on parse failure).
    """
    prompt_json = f"""
From the following analysis, extract a JSON object with the exact structure shown below.
Respond ONLY with valid JSON. All insights and descriptions should be prompt-designed for the next LLM agent.
No markdown, no comments, no additional text.

{{
  "role": "string",
  "general_context": "string",
  "business_context": "string",
  "scope": "string",
  "features_and_concepts": ["string", "string"],
  "other": "string"
}}

Analysis to extract from:
{risposta_finale}
"""
    # BUG FIX: was using undefined 'messaggi'; now correctly built inline
    messaggi = [{"role": "user", "content": prompt_json}]
    testo = chiedi_a_ollama(messaggi)

    # Clean up markdown code fences that some models add around JSON
    testo_pulito = testo.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        struttura = json.loads(testo_pulito)
    except json.JSONDecodeError:
        # Safe fallback: preserve the raw reply without crashing
        print("  [WARN] Model did not return valid JSON. Saving raw response.")
        struttura = {
            "role": "Not automatically extracted",
            "general_context": "Not automatically extracted",
            "business_context": "Not automatically extracted",
            "scope": "Not automatically extracted",
            "features_and_concepts": [],
            "other": "Not automatically extracted",
            "raw_response": risposta_finale
        }

    return struttura


# ─────────────────────────────────────────────────────────────
# 6. MEMORY PERSISTENCE
# ─────────────────────────────────────────────────────────────

def salva_memoria(struttura: dict, filepath: str = OUTPUT_JSON):
    """
    Writes the extracted context structure to disk as a
    human-readable JSON file.

    Parameters
    ----------
    struttura : dict   — the structured context memory to persist
    filepath  : str    — destination file path (default: OUTPUT_JSON)
    """
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(struttura, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] Memory saved to: {filepath}")


# ─────────────────────────────────────────────────────────────
# 7. MAIN AGENT LOOP
# ─────────────────────────────────────────────────────────────

def execute_agent_context(
    work_directory: str,
    general_csv:  str = "general_context.csv",
    business_csv: str = "business_context.csv",
    features_csv: str = "features_context.csv"
) -> dict:
    """
    Full execution of the context-extraction agent:

      1. Change working directory and read the three CSV files.
      2. Build the initial analysis prompt and send it to Ollama.
      3. Display the model's understanding to the user.
      4. Collect iterative feedback in a loop until the user
         types the magic word to confirm the output.
      5. Extract a structured JSON memory and persist it to disk.

    Parameters
    ----------
    work_directory : str  — absolute path to the working folder
    general_csv    : str  — relative path to the general context CSV
    business_csv   : str  — relative path to the business context CSV
    features_csv   : str  — relative path to the features context CSV

    Returns
    -------
    dict
        The final structured context memory extracted by the model.
    """
    os.chdir(work_directory)

    print("=" * 60)
    print("  CONTEXT AGENT — powered by Ollama (local)")
    print(f"  Model: {MODELLO}")
    print("=" * 60)

    # ── STEP 1: Read the three CSV context files ──────────────
    print("\n[INFO] Reading CSV context files...")
    general_ctx  = leggi_csv(general_csv)
    business_ctx = leggi_csv(business_csv)
    features_ctx = leggi_csv(features_csv)

    # ── STEP 2: Build and send the initial prompt ─────────────
    print("[INFO] Sending context to the model (this may take a few seconds)...\n")
    prompt_iniziale = costruisci_prompt_iniziale(
        general_ctx, business_ctx, features_ctx
    )

    # Conversation history — accumulates the full dialogue
    cronologia = [{"role": "user", "content": prompt_iniziale}]

    # ── STEP 3: First model response ─────────────────────────
    risposta = chiedi_a_ollama(cronologia)
    cronologia.append({"role": "assistant", "content": risposta})

    # ── STEP 4: Interactive feedback loop ────────────────────
    while True:
        print("\n" + "─" * 60)
        print("[AGENT UNDERSTANDING]\n")
        print(risposta)
        print("\n" + "─" * 60)
        print(f"\nEnter feedback to refine the analysis "
              f"(or '{PAROLA_MAGICA}' to confirm and exit):\n")

        feedback = input("You -> ").strip()

        # Magic word: break out of the loop and proceed to saving
        if feedback.lower() == PAROLA_MAGICA:
            print("\n[INFO] Understanding confirmed. Saving memory...")
            break

        if not feedback:
            print("  [INFO] No input received — please try again.")
            continue

        # Append user feedback and request an updated analysis
        cronologia.append({"role": "user", "content": feedback})
        print("\n[INFO] Updating understanding...")
        risposta = chiedi_a_ollama(cronologia)
        cronologia.append({"role": "assistant", "content": risposta})

    # ── STEP 5: Extract structured JSON and persist to disk ───
    print("\n[INFO] Extracting final structured memory...")
    struttura = estrai_struttura_json(risposta)
    salva_memoria(struttura)

    print("\n[FINAL MEMORY]")
    print(json.dumps(struttura, ensure_ascii=False, indent=2))

    return struttura




