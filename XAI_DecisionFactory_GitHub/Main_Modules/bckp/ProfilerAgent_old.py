#!/usr/bin/env python
# coding: utf-8

# In[17]:


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
MODELLO = "phi3:mini"

# Magic word that exits the interactive feedback loop
MAGIC_WORD = "yes"

# Input: context memory file produced by the first context agent
CONTEXT_MEMORY_FILE = "context_memory.json"

# Output: JSON file with all generated profile descriptions
OUTPUT_JSON = "profile_descriptions.json"

# HTTP timeout for Ollama calls (seconds)
TIMEOUT = 5000

# All possible column names that may identify a profile
# (case-sensitive — extend this list if needed)
PROFILE_COLUMNS = [
    "profile_ID", "Profile_ID", "profile_id", "profile", "Profile",
    "cluster", "Cluster", "segment", "Segment", "group", "Group"
]


# ─────────────────────────────────────────────────────────────
# 2. LOAD CONTEXT MEMORY  (output of the first context agent)
# ─────────────────────────────────────────────────────────────

def load_context_memory(filepath: str = CONTEXT_MEMORY_FILE) -> str:
    """
    Loads the JSON memory file produced by the context agent and
    formats it as readable text for injection into the LLM prompt.

    The JSON keys are those written by the first agent
    (English names as per the rewritten context_agent.py).

    Parameters
    ----------
    filepath : str  — path to the context_memory.json file

    Returns
    -------
    str
        Formatted multi-line text summarising role, context,
        purpose, and features.

    Raises
    ------
    FileNotFoundError  if the file does not exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"\n[ERROR] '{filepath}' not found!\n"
            "   Run the context agent first to generate it."
        )

    with open(filepath, "r", encoding="utf-8") as f:
        memory = json.load(f)

    # BUG FIX: original code used Italian key names ("ruolo", "contesto_generale", etc.)
    # which do not match the English keys written by the rewritten context agent.
    # Now reading the correct English keys with safe .get() fallbacks.
    features_raw = memory.get("features_and_concepts", [])
    # features_and_concepts may be a list of strings OR a list of dicts
    # (the LLM sometimes returns structured objects instead of plain strings).
    # Normalise every item to a string before joining.
    if isinstance(features_raw, list):
        features_str = ", ".join(
            json.dumps(item, ensure_ascii=False) if isinstance(item, dict) else str(item)
            for item in features_raw
        )
    else:
        features_str = str(features_raw)

    text = (
        f"ROLE: {memory.get('role', 'N/A')}\n"
        f"GENERAL CONTEXT: {memory.get('general_context', 'N/A')}\n"
        f"BUSINESS CONTEXT: {memory.get('business_context', 'N/A')}\n"
        f"PURPOSE: {memory.get('scope', 'N/A')}\n"
        f"FEATURES AND CONCEPTS: {features_str}\n"
        f"OTHER: {memory.get('other', 'N/A')}"
    )

    print(f"  [OK] Context memory loaded from '{filepath}'")
    return text


# ─────────────────────────────────────────────────────────────
# 3. CSV READER
# ─────────────────────────────────────────────────────────────

def read_csv(filepath: str, sep: str = ";") -> pd.DataFrame:
    """
    Reads a CSV file with configurable separator.
    Attempts UTF-8 encoding first, then falls back to latin-1.
    Returns an empty DataFrame if the file is not found or unreadable.

    Parameters
    ----------
    filepath : str  — path to the CSV file
    sep      : str  — column separator (default: ";")

    Returns
    -------
    pd.DataFrame
    """
    if not os.path.exists(filepath):
        print(f"  [WARN] File not found: {filepath} — skipped.")
        return pd.DataFrame()

    for enc in ["utf-8", "latin-1"]:
        try:
            df = pd.read_csv(filepath, sep=sep, encoding=enc)
            print(f"  [OK] {filepath} — {df.shape[0]} rows, {df.shape[1]} cols")
            return df
        except Exception:
            continue

    print(f"  [ERR] Cannot read: {filepath}")
    return pd.DataFrame()


# ─────────────────────────────────────────────────────────────
# 4. SHAP JSON READER
# ─────────────────────────────────────────────────────────────

def read_shap_json(filepath: str) -> str:
    """
    Reads the SHAP importance JSON file produced by the analytical
    pipeline and returns its content as a formatted text string
    ready for prompt injection.

    Returns 'Not available.' if the file is missing or unparseable.

    Parameters
    ----------
    filepath : str  — path to shap_importance.json

    Returns
    -------
    str
    """
    if not os.path.exists(filepath):
        print(f"  [WARN] SHAP file not found: {filepath} — skipped.")
        return "Not available."

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"  [OK] {filepath} — SHAP data loaded.")
        return json.dumps(data, ensure_ascii=False, indent=2)
    except json.JSONDecodeError as e:
        print(f"  [ERR] Cannot parse {filepath}: {e}")
        return "Not available."


# ─────────────────────────────────────────────────────────────
# 5. PROFILE UTILITIES
# ─────────────────────────────────────────────────────────────

def find_profile_column(df: pd.DataFrame) -> str | None:
    """
    Scans the DataFrame columns for the first name that matches
    any entry in PROFILE_COLUMNS (case-sensitive).

    Returns the matching column name, or None if none is found.
    """
    for col in PROFILE_COLUMNS:
        if col in df.columns:
            return col
    return None


def extract_profiles(df_stats: pd.DataFrame) -> list:
    """
    Returns the sorted list of unique profile IDs found in the
    profile_stats DataFrame.

    Uses the first matching profile column from PROFILE_COLUMNS.
    Falls back to the first DataFrame column if none matches.

    Parameters
    ----------
    df_stats : pd.DataFrame  — the profile statistics table

    Returns
    -------
    list of profile ID values
    """
    col = find_profile_column(df_stats)
    if col:
        profiles = sorted(df_stats[col].unique().tolist(), key=str)
        print(f"  [OK] Profile column: '{col}' — {len(profiles)} unique profiles")
        return profiles

    fallback_col = df_stats.columns[0]
    print(f"  [WARN] No profile column found; using first column: '{fallback_col}'")
    return sorted(df_stats[fallback_col].unique().tolist(), key=str)


def filter_by_profile(df: pd.DataFrame, profile_id: str) -> str:
    """
    Filters a DataFrame to the rows matching profile_id and
    returns the data as a 'feature: value' text block — a format
    that is much easier for an LLM to read than a raw table.

    If the DataFrame has no recognised profile column (e.g. global
    data like SHAP values), the full table is returned as-is.

    If the DataFrame is empty, returns 'Not available.'.

    Parameters
    ----------
    df         : pd.DataFrame  — source data table
    profile_id : str           — the profile ID to filter on

    Returns
    -------
    str
    """
    if df.empty:
        return "Not available."

    col = find_profile_column(df)
    if col:
        subset = df[df[col].astype(str) == str(profile_id)]
        if not subset.empty:
            lines = []
            for _, row in subset.iterrows():
                for colname, value in row.items():
                    if colname != col:   # skip the profile_id column itself
                        lines.append(f"  {colname}: {value}")
                lines.append("")         # blank line between rows
            return "\n".join(lines).strip()

    # No profile column found — return full table (used for global sources)
    return df.to_string(index=False)


# ─────────────────────────────────────────────────────────────
# 6. OLLAMA HTTP CALL
# ─────────────────────────────────────────────────────────────

def ask_ollama(messages: list) -> str:
    """
    Sends a conversation history to the local Ollama server and
    returns the model's plain-text reply.

    Parameters
    ----------
    messages : list of dicts
        Full conversation history in OpenAI-compatible format:
        [{"role": "user" | "assistant" | "system", "content": "..."}]

    Returns
    -------
    str  — the text content of the model's response

    Raises
    ------
    ConnectionError  if Ollama is not running.
    TimeoutError     if the model does not respond within TIMEOUT.
    """
    payload = {
        "model": MODELLO,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.35,  # balanced: creative but precise
            "num_predict": 800   # max response length in tokens
        }
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
        response.raise_for_status()
        return response.json()["message"]["content"]

    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            "\n[ERROR] Ollama is not running!\n"
            "   Start it with: ollama serve"
        )
    except requests.exceptions.Timeout:
        raise TimeoutError(
            f"\n[ERROR] Timeout after {TIMEOUT}s.\n"
            "   Try reducing the CSV data size or increase TIMEOUT."
        )


# ─────────────────────────────────────────────────────────────
# 7. INTERACTIVE FEEDBACK LOOP
# ─────────────────────────────────────────────────────────────

def feedback_loop(initial_response: str, history: list, label: str) -> str:
    """
    Displays the current model response and collects user feedback
    in a loop until the magic word is entered.

      - Free text  → appended to history, new Ollama call, loop continues
      - MAGIC_WORD → confirms the current response and exits the loop

    Parameters
    ----------
    initial_response : str   — first model response to show
    history          : list  — conversation history (mutated in place)
    label            : str   — section label shown in the console header

    Returns
    -------
    str  — the final confirmed response text
    """
    response = initial_response

    while True:
        print(f"\n{'─' * 60}")
        print(f"[{label}]\n")
        print(response)
        print(f"\n{'─' * 60}")
        print(f"\nEnter feedback to refine, or '{MAGIC_WORD}' to confirm:\n")

        user_input = input("You -> ").strip()

        if user_input.lower() == MAGIC_WORD:
            print("[OK] Confirmed.")
            return response

        if not user_input:
            print("  [INFO] Empty input — please try again.")
            continue

        history.append({"role": "user", "content": user_input})
        print("\n[INFO] Updating response...")
        response = ask_ollama(history[-2:]) #gestisco la poca memoria
        history.append({"role": "assistant", "content": response})


# ─────────────────────────────────────────────────────────────
# 8. PROFILE PROMPT BUILDER
# ─────────────────────────────────────────────────────────────

def build_profile_prompt(
    context:    str,
    profile_id: str,
    stats:      str,
    shap:       str,
    features:   str,
    oddratio:   str,
    rules:      str
) -> str:
    """
    Assembles the full analytical prompt for a single profile.
    The profile_id is injected explicitly at the top so the model
    never has to guess or infer it from tabular data.

    Parameters
    ----------
    context    : str  — business context from the first agent
    profile_id : str  — the profile being described
    stats      : str  — filtered profile statistics text
    shap       : str  — global SHAP importance text
    features   : str  — filtered decision-tree feature importances
    oddratio   : str  — filtered odds-ratio text
    rules      : str  — filtered decision-tree assignment rules

    Returns
    -------
    str  — the fully assembled prompt
    """
    return f"""
You are an expert data analyst and storyteller specialized in machine learning model output profiling.
You analyze and make actionable profiles based on ML model score.
You write compelling, data-driven, actionable profile descriptions for stakeholders.

---
## BUSINESS CONTEXT
{context}

---
## YOU ARE ANALYZING EXACTLY THIS PROFILE:
Profile ID = {profile_id}
This ID comes directly from the profile_stats file.
Use it as-is. Do not change it, do not invent others.

---
## DATA FOR PROFILE ID: {profile_id}

### Statistical Summary (mean and mode per feature and Profile_ID)
{stats}

### SHAP Mean Values (average feature impact on model prediction)
{shap}

### Feature Importance (decision tree — top drivers for this profile)
{features}

### Odds Ratios (logistic regression — likelihood vs average population)
{oddratio}

### Assignment Rules (conditions that assign a customer to this profile)
{rules}

---
## YOUR TASK

Write a complete description of the profile with ID = {profile_id}.
Use ONLY the numbers and values listed above. Do NOT invent data. Not use metrics just numbers

Structure your answer as follows:

---
### Profile Name and Tagline and Propensity Score | ID: {profile_id}
- **Name**: A short memorable name (e.g. "The Loyal High-Spender")
- **Tagline**: One sentence capturing who they are
- **Propensity Score**: HP,you find this information in the profile_stats

---
### Who Are They? (Storytelling)
2 sentences. Tell a vivid human story using the actual feature values above.
Ground every sentence in real data — for example: "On average, they spend X per month and visit Y times."
Add also 1 sentence about the main difference between the Profile_ID vs all the others 

---
### Key Data Insights
2 bullet points in this format:
- **[feature name]** = [value from data] -> [what this means for the business] -> [difference with all other prorfiles]
Only use values explicitly present in the data above.

---
### Actionable Recommendations
Provide 2 concrete, specific actions. Each must:
- Start with a verb (for example: "Launch", "Target", "Reduce", "Increase")
- Reference specific data values to justify the action
- Be directly executable by a business team

---
### Watch Out For
List 1 risks or pitfalls specific to this profile based on the data.

###CONSTRAINTS:
- Every profile MUST have a unique Name, Tagline, and Recommendations. Never reuse the same words across profiles.
- If Profile_ID contains "Generic": write shorter, vaguer descriptions. These are low-propensity row, do not over-analyze them.
- If Profile_ID does NOT contain "Generic": be specific, detailed, and data-driven.
---
### Be specific. Be data-driven. Avoid generic statements like "this segment is important".
    Every claim must trace back to a number or pattern in the data provided.
---
""".strip()


# ─────────────────────────────────────────────────────────────
# 9. SINGLE PROFILE ANALYSIS
# ─────────────────────────────────────────────────────────────

def analyze_profile(
    context:     str,
    profile_id:  str,
    shap_txt:    str,
    df_stats:    pd.DataFrame,
    df_features: pd.DataFrame,
    df_oddratio: pd.DataFrame,
    df_rules:    pd.DataFrame
) -> dict:
    """
    Generates the full description for a single profile:
      1. Filters each DataFrame to the rows for this profile_id
      2. Builds the analytical prompt with all data injected
      3. Calls Ollama for the first response
      4. Runs the interactive feedback loop until the user confirms
      5. Returns the confirmed description as a dict

    Parameters
    ----------
    context     : str           — business context text
    profile_id  : str           — profile being analyzed
    shap_txt    : str           — global SHAP importance text
    df_stats    : pd.DataFrame  — profile statistics table
    df_features : pd.DataFrame  — decision-tree feature importances
    df_oddratio : pd.DataFrame  — logistic regression odds ratios
    df_rules    : pd.DataFrame  — decision-tree assignment rules

    Returns
    -------
    dict with keys "profile_id" and "description"
    """
    print(f"\n  [INFO] Analyzing profile: '{profile_id}'...")

    # Filter each data source to this specific profile
    stats_txt    = filter_by_profile(df_stats,    profile_id)
    features_txt = filter_by_profile(df_features, profile_id)
    oddratio_txt = filter_by_profile(df_oddratio, profile_id)
    rules_txt    = filter_by_profile(df_rules,    profile_id)

    # Assemble the prompt with profile_id explicitly injected
    prompt = build_profile_prompt(
        context, profile_id,
        stats_txt, shap_txt, features_txt, oddratio_txt, rules_txt
    )

    # First model call
    history  = [{"role": "user", "content": prompt}]
    response = ask_ollama(history)
    history.append({"role": "assistant", "content": response})

    # Interactive feedback loop — user refines or confirms with "yes"
    final_description = feedback_loop(
        response, history,
        label=f"PROFILE DESCRIPTION — ID: {profile_id}"
    )

    return {
        "profile_id":  str(profile_id),
        "description": final_description
    }


# ─────────────────────────────────────────────────────────────
# 10. SAVE OUTPUT
# ─────────────────────────────────────────────────────────────

def save_output(results: list, filepath: str = OUTPUT_JSON):
    """
    Saves all generated profile descriptions to a structured JSON file.

    Parameters
    ----------
    results  : list  — list of dicts produced by analyze_profile()
    filepath : str   — destination file path
    """
    output = {
        "total_profiles": len(results),
        "profiles": results
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] All profile descriptions saved to: {filepath}")


# ─────────────────────────────────────────────────────────────
# 11. MAIN FUNCTION
# ─────────────────────────────────────────────────────────────

def run_profile_agent(
    work_directory:       str,
    oddratio_csv:         str = "oddsratio.csv",
    profile_features_csv: str = "profile_features.csv",
    profile_stats_csv:    str = "profile_stats.csv",
    profile_rules_csv:    str = "profile_rules.csv",
    shap_importance_json: str = "shap_importance.json",
    context_memory:       str = CONTEXT_MEMORY_FILE,
    sep:                  str = ";"
) -> list:
    """
    Main entry point of the profile analyst agent.

    Steps
    -----
    1. Change working directory.
    2. Load context memory from the first (context) agent.
    3. Read the four profile CSV files and the SHAP JSON.
    4. Extract unique profile IDs from profile_stats.csv.
    5. For each profile: generate description -> feedback loop -> confirm.
    6. Save all descriptions to profile_descriptions.json.

    Parameters
    ----------
    work_directory        : str  — absolute path to the working folder
    oddratio_csv          : str  — relative path to odds-ratio CSV
    profile_features_csv  : str  — relative path to feature importances CSV
    profile_stats_csv     : str  — relative path to profile statistics CSV
    profile_rules_csv     : str  — relative path to decision-tree rules CSV
    shap_importance_json  : str  — relative path to SHAP importance JSON
    context_memory        : str  — relative path to context memory JSON
    sep                   : str  — CSV column separator (default: ";")

    Returns
    -------
    list of dicts, one per profile, each with keys
    "profile_id" and "description".
    """
    os.chdir(work_directory)

    print("=" * 60)
    print("  PROFILE ANALYST AGENT — powered by Ollama")
    print(f"  Model : {MODELLO}")
    print(f"  Folder: {work_directory}")
    print("=" * 60)

    # ── STEP 1: Load context memory from the first agent ─────
    print("\n[INFO] Loading business context memory...")
    context = load_context_memory(context_memory)

    # ── STEP 2: Read profile CSV files ───────────────────────
    print("\n[INFO] Loading profile CSV files...")
    df_oddratio  = read_csv(oddratio_csv,         sep)
    df_features  = read_csv(profile_features_csv, sep)
    df_stats     = read_csv(profile_stats_csv,    sep)
    df_rules     = read_csv(profile_rules_csv,    sep)

    # ── STEP 3: Read SHAP JSON (global — not filtered per profile) ─
    print("\n[INFO] Loading SHAP importance JSON...")
    shap_txt = read_shap_json(shap_importance_json)

    # ── STEP 4: Extract profile list from stats table ────────
    if df_stats.empty:
        raise ValueError(
            "[ERROR] profile_stats.csv is empty or not found.\n"
            "   This file is required to extract the profile list."
        )
    profiles = extract_profiles(df_stats)
    print(f"\n[INFO] Profiles to analyze: {profiles}")
    print(f"       Total: {len(profiles)}")

    # ── STEP 5: Analyze each profile with interactive feedback ─
    results = []
    for i, profile_id in enumerate(profiles, 1):
        print(f"\n{'=' * 60}")
        print(f"  PROFILE {i} of {len(profiles)}: '{profile_id}'")
        print(f"{'=' * 60}")

        result = analyze_profile(
            context, profile_id, shap_txt,
            df_stats, df_features, df_oddratio, df_rules
        )
        results.append(result)
        print(f"\n[OK] Profile '{profile_id}' confirmed — moving to next.")

    # ── STEP 6: Persist all results ───────────────────────────
    save_output(results)

    print(f"\n[DONE] {len(results)} profiles analyzed.")
    print(f"       Output: {OUTPUT_JSON}")

    return results

