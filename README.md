# XAI_DecisionFactory
**Turn ML outputs into actionable insights with semi-global explainability and dual-agent HITL intelligence.**

XAI_DecisionFactory is an **Explainable AI framework** for regression and classification models that combines two complementary layers:

- **Statistical Layer** – analyzes model outputs to uncover **semi-global patterns**, robust segments, and key feature contributions.  

- **Agentic Layer with Human-in-the-Loop (HITL)** – powered by **two specialized agents**:  
  - **Context Agent**: interprets **model context, purpose, and objectives** to align insights with business or research goals.  
  - **Analytical Agent**: performs **analytical reasoning** on statistical outputs, generating **interpretable profiles, actionable insights, and recommendations**. Human feedback ensures **hallucination control** and **refinement of results**.

This **dual-agent, two-step architecture** ensures your models are not only **technically transparent** but also **decision-ready**, bridging the gap between data science outputs and real-world action.

### Key Features
- **Semi-global explainability** – focuses on meaningful segments rather than single predictions or entire models  
- **Actionable insights** – translates model patterns into concrete recommendations  
- **Dual-agent HITL output** – produces **human-validated, context-aware profiles and guidance** informed by model purpose and objectives

---

## Phase 1 — Analytical Profiling  
**Extracting statistically grounded insights**  

This phase transforms raw model outputs into **structured, interpretable profiles**.  

High-propensity (HP) observations are defined as the top *x%* of model scores (threshold configurable), aiming to identify subpopulations with distinct behavioral and predictive patterns.

**Methodological Steps**:  
- **SHAP-based Modeling** – train a secondary model (e.g., Random Forest or LightGBM) using HP as target to compute SHAP values.  
- **Dimensionality Reduction (PCA)** – reduce feature space complexity while preserving variance.  
- **Clustering (DBSCAN)** – identify dense groups of similar observations; retain clusters with significantly higher HP rates than baseline.  
- **Decision Tree Profiling** – extract interpretable rules describing each cluster; final leaves are called **profiles**.  
- **Logistic Regression (Odds Ratios)** – quantify feature contributions within each profile.  

**Outputs**:  
- Profile-level statistical summaries  
- Interpretable rules  
- Feature importance and Odds Ratios  
- Structured `.csv` / `.json` datasets  
- Profile assignment per observation  
- PDF report with plots  

This hybrid approach ensures **robustness and interpretability**.

---

## Phase 2 — Agentic AI · Context Understanding  
**Structuring business knowledge**  

An **LLM-based Context Agent** ingests:  
- Business domain and objectives  
- Model purpose  
- Feature definitions  
- Constraints and success criteria  

It produces a **formalized contextual representation**, aligning technical outputs with business meaning.

---

## Phase 3 — Agentic AI · Analytical Reasoning  
**From statistics to decisions**  

The **Analytical Agent** combines:  
- Context (Phase 2)  
- Statistical profiles (Phase 1)  

It generates:  
- Human-readable profile descriptions  
- Data-grounded explanations  
- Targeted recommendations  

Each profile becomes a **decision unit**.

**LLM Implementation**:  
- Uses **phi3-mini via Ollama**  
- Runs efficiently on **basic CPU**
