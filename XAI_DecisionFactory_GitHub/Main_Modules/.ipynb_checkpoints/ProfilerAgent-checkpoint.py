#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
MAGIC_WORD = "yes"
CONTEXT_MEMORY_FILE = "context_memory.json"
OUTPUT_JSON = "profile_descriptions.json"
TIMEOUT = 5000
PROFILE_COLUMNS = [
    "profile_ID", "Profile_ID", "profile_id", "profile", "Profile",
    "cluster", "Cluster", "segment", "Segment", "group", "Group"
]


# ─────────────────────────────────────────────────────────────
# 2. LOAD CONTEXT MEMORY
# ─────────────────────────────────────────────────────────────

def load_context_memory(filepath: str = CONTEXT_MEMORY_FILE) -> str:
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"\n[ERROR] '{filepath}' not found!\n"
            "   Run the context agent first to generate it."
        )

    with open(filepath, "r", encoding="utf-8") as f:
        memory = json.load(f)

    features_raw = memory.get("features_and_concepts", [])
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
    for col in PROFILE_COLUMNS:
        if col in df.columns:
            return col
    return None


def extract_profiles(df_stats: pd.DataFrame) -> list:
    col = find_profile_column(df_stats)
    if col:
        profiles = sorted(df_stats[col].unique().tolist(), key=str)
        print(f"  [OK] Profile column: '{col}' — {len(profiles)} unique profiles")
        return profiles

    fallback_col = df_stats.columns[0]
    print(f"using first column as Profile: '{fallback_col}'")
    return sorted(df_stats[fallback_col].unique().tolist(), key=str)


def filter_by_profile(df: pd.DataFrame, profile_id: str) -> str:
    if df.empty:
        return "Not available."

    col = find_profile_column(df)
    if col:
        subset = df[df[col].astype(str) == str(profile_id)]
        if not subset.empty:
            lines = []
            for _, row in subset.iterrows():
                for colname, value in row.items():
                    if colname != col:
                        lines.append(f"  {colname}: {value}")
                lines.append("")
            return "\n".join(lines).strip()

    return df.to_string(index=False)


# ─────────────────────────────────────────────────────────────
# 6. OLLAMA HTTP CALL
# ─────────────────────────────────────────────────────────────

def ask_ollama(messages: list, llm: str = MODELLO, ollama_url: str = OLLAMA_URL) -> str:  # FIX: aggiunto ollama_url
    payload = {
        "model": llm,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 810
        }
    }

    try:
        response = requests.post(ollama_url, json=payload, timeout=TIMEOUT)  # FIX: ollama_url invece di OLLAMA_URL
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

def feedback_loop(initial_response: str, history: list, label: str,
                  llm: str = MODELLO, ollama_url: str = OLLAMA_URL) -> str:  # FIX: aggiunti llm e ollama_url
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
        response = ask_ollama(history[-3:], llm=llm, ollama_url=ollama_url)  # FIX: -3 invece di -2, + llm e ollama_url
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
    return f"""
You are an expert data analyst and storyteller specialized in machine learning model output profiling.
You analyze and make actionable profiles based on ML model score.
You write compelling, data-driven, actionable profile descriptions for stakeholders.
You don't use units like ("$", "%" or similar), just look at the numbers.

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
- **Name**: A short memorable unique name (e.g. "The Loyal High-Spender")
- **Tagline**: One sentence capturing who they are
- **Propensity Score**: HP,you find this information in the profile_stats

---
### Who Are They? (Storytelling)
2 sentences. Tell a vivid human story using the actual feature values above.
Ground every sentence in real data — for example: "On average, they spend X per month and visit Y times."
Add also 1 sentence about the main difference between the Profile_ID vs all the others 

---
### Key Data Insights
1 bullet points in this format:
- **[feature name]** = [value from data] -> [what this means for the business] -> [difference with all other prorfiles]
Only use values and metrics explicitly present in the data above. Don't create metrics by yourslef

---
### Actionable Recommendations
Provide 1 concrete, specific actions.  must:
- Start with a verb (for example: "Launch", "Target", "Reduce", "Increase")
- Reference specific data values to justify the action
- Be directly executable by a business team

---
### Watch Out For
List 1 risks or pitfalls specific to this profile based on the data.
---
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
    df_rules:    pd.DataFrame,
    llm:         str = MODELLO,
    ollama_url:  str = OLLAMA_URL  # FIX: aggiunto ollama_url
) -> dict:
    print(f"\n  [INFO] Analyzing profile: '{profile_id}'...")

    stats_txt    = filter_by_profile(df_stats,    profile_id)
    features_txt = filter_by_profile(df_features, profile_id)
    oddratio_txt = filter_by_profile(df_oddratio, profile_id)
    rules_txt    = filter_by_profile(df_rules,    profile_id)

    prompt = build_profile_prompt(
        context, profile_id,
        stats_txt, shap_txt, features_txt, oddratio_txt, rules_txt
    )

    history  = [{"role": "user", "content": prompt}]
    response = ask_ollama(history, llm=llm, ollama_url=ollama_url)  # FIX: ollama_url propagato
    history.append({"role": "assistant", "content": response})

    final_description = feedback_loop(
        response, history,
        label=f"PROFILE DESCRIPTION — ID: {profile_id}",
        llm=llm, ollama_url=ollama_url  # FIX: llm e ollama_url propagati
    )

    return {
        "profile_id":  str(profile_id),
        "description": final_description
    }


# ─────────────────────────────────────────────────────────────
# 10. SAVE OUTPUT
# ─────────────────────────────────────────────────────────────

def save_output(results: list, filepath: str = OUTPUT_JSON):
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
    sep:                  str = ";",
    llm:                  str = MODELLO,
    ollama_url:           str = OLLAMA_URL  # FIX: aggiunto ollama_url
) -> list:

    os.chdir(work_directory)

    print("=" * 60)
    print("  PROFILE ANALYST AGENT — powered by Ollama")
    print(f"  Model : {llm}")
    print(f"  Folder: {work_directory}")
    print("=" * 60)

    print("\n[INFO] Loading business context memory...")
    context = load_context_memory(context_memory)

    print("\n[INFO] Loading profile CSV files...")
    df_oddratio  = read_csv(oddratio_csv,         sep)
    df_features  = read_csv(profile_features_csv, sep)
    df_stats     = read_csv(profile_stats_csv,    sep)
    df_rules     = read_csv(profile_rules_csv,    sep)

    print("\n[INFO] Loading SHAP importance JSON...")
    shap_txt = read_shap_json(shap_importance_json)

    if df_stats.empty:
        raise ValueError(
            "[ERROR] profile_stats.csv is empty or not found.\n"
            "   This file is required to extract the profile list."
        )

    profiles = extract_profiles(df_stats)
    print(f"Total Profiles to analyze: {len(profiles)}")

    results = []
    for i, profile_id in enumerate(profiles, 1):
        print(f"\n{'=' * 60}")
        print(f"  PROFILE {i} of {len(profiles)}: '{profile_id}'")
        print(f"{'=' * 60}")

        result = analyze_profile(
            context, profile_id, shap_txt,
            df_stats, df_features, df_oddratio, df_rules,
            llm=llm, ollama_url=ollama_url  # FIX: ollama_url propagato
        )
        results.append(result)
        print(f"\n[OK] Profile '{profile_id}' confirmed — moving to next.")

    save_output(results)

    print(f"\n[DONE] {len(results)} profiles analyzed.")
    print(f"       Output: {OUTPUT_JSON}")

    return results


# In[ ]:




