#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import json
import numpy as np
import pandas as pd
import shap
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from statsmodels.stats.proportion import proportions_ztest

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm as rl_cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    PageBreak, Table, TableStyle
)


def ds_profile_generator(
    work_directory,
    file_path,
    sep=";",
    decimal=",",
    id_col="ID",
    target_score_col="score",
    threshold=10,
    min_profile_size=0.05,
    eps=0.25,
    output_path="IF_analytical_output"
):
    """
    Full analytical pipeline: SHAP importance, DBSCAN clustering,
    decision-tree profiling with statistical validation, elastic-net
    multinomial odds-ratio, and PDF visual report.

    Parameters
    ----------
    file_path         : path to the input CSV file
    sep               : column separator in the CSV
    decimal           : decimal separator in the CSV
    id_col            : name of the unique identifier column
    target_score_col  : name of the model score column
    threshold         : percentile used to define HP (top) and LP (bottom)
    min_profile_size  : minimum profile size as a fraction of total rows (0-1);
                        used as min_samples in DBSCAN and as min_samples_leaf
                        lower-bound in the decision tree search
    eps               : DBSCAN neighbourhood radius
    output_path       : folder where all output files are written

    Output files
    ------------
    shap_importance.json    global mean |SHAP| per feature, sorted descending
    PROFILE_RULES.csv       decision-tree path (rule) for each leaf / Profile_ID
    PROFILE_STATS.csv       mean of all numeric columns aggregated by Profile_ID
    ID_W_PROFILE.csv        row-level mapping: id, Profile_ID, HP, LP
    PROFILE_FEATURES.csv    decision-tree feature importances (non-zero only)
    oddsratio.csv           odds-ratio from elastic-net multinomial logistic
                            regression; columns = feature names, rows = classes
    InsightForge_Report.pdf visual report: SHAP, PCA scatter, numeric bars,
                            categorical bars, OR diverging bar charts per profile
    Main Steps
    1. Data Preprocessing

    Input data are prepared through a standardized preprocessing pipeline.
    This includes:

    Handling missing values via statistical imputation (mean for numerical features, explicit category for categorical ones)
    Encoding categorical variables using one-hot encoding to ensure model compatibility
    Feature scaling (Min-Max normalization) to make variables comparable and suitable for distance-based algorithms (e.g., DBSCAN, PCA, logistic regression)
    2. SHAP-Based Feature Importance

    Two ensemble models, Random Forest and LightGBM, are trained using the binary target variable HP (High Propensity).

    Model selection is performed via cross-validation using ROC-AUC as the evaluation metric
    The best-performing model is selected
    SHAP (SHapley Additive exPlanations) values are computed to quantify the marginal contribution of each feature
    Global feature importance is derived as the mean absolute SHAP value across observations

    ⚠️ If all candidate models achieve ROC-AUC < 0.65, a warning is raised to indicate weak predictive performance and potential instability of downstream insights

    3. PCA for Feature Embedding

    To mitigate the curse of dimensionality and improve clustering stability:

    Feature space is reduced using Principal Component Analysis (PCA)
    The number of components is selected to retain at least ~70% of total variance
    This transformation preserves the most informative structure of the data while reducing noise and redundancy
    The transformation is applied after encoding and scaling to ensure consistency
    4. DBSCAN Clustering

    Customer segmentation is performed using DBSCAN, a density-based clustering algorithm:

    Clusters group observations with similar feature patterns in the transformed space
    A minimum cluster size constraint is enforced via min_samples
    For each cluster, the HP rate (high propensity proportion) is computed

    Cluster validation:

    Only clusters with HP rate at least 20% higher than the global baseline are considered
    A two-proportion z-test is performed comparing cluster vs population
    Clusters failing statistical significance (p > 0.05 or |z| < 1.96) are discarded
    Invalid clusters are reassigned to a fallback segment labeled "Generic"
    5. Decision Tree for Profile Extraction

    An interpretable segmentation layer is built using a Decision Tree classifier:

    Target variable: validated clusters from DBSCAN
    The tree learns explicit rules mapping features to cluster membership
    Each leaf defines a candidate Profile_ID

    Profile validation:

    Profiles must exceed minimum size constraints
    Profiles must show HP rate at least +20% vs baseline
    Statistical validation is performed via two-proportion z-test
    Profiles failing significance are reassigned to "Generic"

    The final output is a set of interpretable, statistically validated customer profiles
    ⚠️ If all candidate models achieve ROC-AUC < 0.65, a warning is raised to indicate weak predictive performance and potential instability of downstream insights

    6. Logistic Regression for Odds Impact Analysis

    A multinomial logistic regression with Elastic Net regularization is trained:

    Target: Profile_ID
    Inputs: full feature space (encoded and scaled)
    Model is optimized via RandomizedSearchCV

    Outputs:

    Coefficients are exponentiated to obtain odds ratios (OR)
    Each row represents a profile, each column a feature
    OR > 1 → positive association with the profile
    OR < 1 → negative association

    Only statistically and practically relevant effects are emphasized in reporting and visualization.
    If all candidate models achieve ROC-AUC < 0.65, a warning is raised to indicate weak predictive performance and potential instability of downstream insights
    """
    os.chdir (work_directory)
    os.makedirs(output_path, exist_ok=True)

    # =========================================================
    # 1. LOAD DATA
    # =========================================================
    df = pd.read_csv(file_path, sep=sep, decimal=decimal)

    # =========================================================
    # 2. HP / LP / TRICT LABELS
    #    HP = 1 for scores in the top-threshold percentile
    #    LP = 1 for scores in the bottom-threshold percentile
    #    TRICT: 0 = LP band, 1 = middle band, 2 = HP band
    # =========================================================
    p_low, p_high = np.percentile(df[target_score_col], [threshold, 100 - threshold])

    df["HP"]    = np.where(df[target_score_col] >= p_high, 1, 0)
    df["LP"]    = np.where(df[target_score_col] <= p_low,  1, 0)
    df["TRICT"] = np.where(
        df[target_score_col] <= p_low,  0,
        np.where(df[target_score_col] > p_high, 2, 1)
    )

    # Global HP rate: baseline for all propensity comparisons
    base_propensity = df["HP"].mean()

    # =========================================================
    # 3. FEATURE MATRIX
    #    Built once and reused across SHAP, PCA/DBSCAN, the
    #    decision tree, and the logistic regression.
    # =========================================================
    _internal_cols = {id_col, target_score_col, "HP", "LP", "TRICT"}

    raw_num_cols = [
        c for c in df.select_dtypes(exclude=["object", "category"]).columns
        if c not in _internal_cols
    ]
    raw_cat_cols = [
        c for c in df.select_dtypes(include=["object", "category"]).columns
        if c not in _internal_cols
    ]

    # Impute numeric columns with their column mean
    X_num = df[raw_num_cols].fillna(df[raw_num_cols].mean())

    # One-hot encode categorical columns (drop first level to avoid collinearity)
    ohe_main = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
    if raw_cat_cols:
        X_cat_arr     = ohe_main.fit_transform(df[raw_cat_cols])
        ohe_cat_names = ohe_main.get_feature_names_out(raw_cat_cols).tolist()
        X_cat_df      = pd.DataFrame(X_cat_arr, columns=ohe_cat_names, index=df.index)
    else:
        X_cat_df      = pd.DataFrame(index=df.index)
        ohe_cat_names = []

    X                   = pd.concat([X_num, X_cat_df], axis=1)
    feature_names_model = list(X.columns)

    # Scale to [0, 1] — required by DBSCAN cosine metric and logistic regression
    scaler   = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # =========================================================
    # 4. SHAP MODEL
    #    Compare LightGBM and RandomForest via 5-fold CV ROC-AUC.
    #    The winner is refitted on the full dataset and used for
    #    global SHAP importance computation.
    # =========================================================
    Y_hp = df["HP"].values

    search_gb = RandomizedSearchCV(
        lgb.LGBMClassifier(verbose=-1),
        {"n_estimators": [50, 100]},
        cv=5, scoring="roc_auc", n_jobs=-1, random_state=1926
    )
    search_rf = RandomizedSearchCV(
        RandomForestClassifier(),
        {"n_estimators": [50, 100]},
        cv=5, scoring="roc_auc", n_jobs=-1, random_state=1926
    )
    search_gb.fit(X_scaled, Y_hp)
    search_rf.fit(X_scaled, Y_hp)

    # ROC-AUC reliability check: warn if either model underperforms
    _ROC_THRESHOLD = 0.65
    if search_gb.best_score_ < _ROC_THRESHOLD:
        print(f"[WARNING] Model LightGBMClassifier - not reliable results "
              f"(ROC-AUC = {search_gb.best_score_:.3f} < {_ROC_THRESHOLD})")
    if search_rf.best_score_ < _ROC_THRESHOLD:
        print(f"[WARNING] Model RandomForestClassifier - not reliable results "
              f"(ROC-AUC = {search_rf.best_score_:.3f} < {_ROC_THRESHOLD})")

    best_model = (
        search_gb if search_gb.best_score_ >= search_rf.best_score_
        else search_rf
    ).best_estimator_
    best_model.fit(X_scaled, Y_hp)

    explainer   = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_scaled)

    # Robust extraction: handle list output (RandomForest) and 3-D arrays
    if isinstance(shap_values, list):
        shap_array = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        shap_array = shap_values
    if shap_array.ndim == 3:
        shap_array = shap_array[:, :, 1]

    mean_impact = np.abs(shap_array).mean(axis=0).reshape(-1)

    shap_output = sorted(
        [{"feature": feat, "mean_impact": float(mean_impact[i])}
         for i, feat in enumerate(feature_names_model)],
        key=lambda x: x["mean_impact"], reverse=True
    )
    with open(f"{output_path}/shap_importance.json", "w") as f:
        json.dump(shap_output, f, indent=2)

    # =========================================================
    # 5. PCA  (dimensionality reduction before DBSCAN)
    #    Retain components that together explain at least 71%
    #    of the total variance.
    # =========================================================
    pca_cluster = PCA(n_components=0.71, random_state=1926)
    X_pca       = pca_cluster.fit_transform(X_scaled)

    # =========================================================
    # 6. DBSCAN  ->  initial cluster assignment
    #    min_samples is derived from min_profile_size so that
    #    clusters smaller than the minimum profile threshold are
    #    treated as noise.  Noise rows (label = -1) fall into
    #    "Generic" downstream.
    # =========================================================
    min_samples   = max(3, int(min_profile_size * len(X_pca)))
    db            = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    df["cluster"] = db.fit_predict(X_pca)

    df["cluster_propensity"] = df.groupby("cluster")["HP"].transform("mean")

    # Keep clusters whose HP rate is at least 1.2x the global baseline
    df["final_cluster"] = np.where(
        df["cluster_propensity"] / base_propensity > 1.2,
        df["cluster"].astype(str),
        "Generic"
    )

    # =========================================================
    # 7. Z-TEST ON CLUSTERS
    #    Each candidate cluster is validated with a two-proportion
    #    z-test comparing its HP rate against the rest of the data.
    #    Clusters that fail (p > 0.05 or |z| < 1.96) become "Generic".
    # =========================================================
    cluster_ztest_rows = []
    for cl in df["final_cluster"].unique():
        mask_cl = df["final_cluster"] == cl
        z_stat, p_val = proportions_ztest(
            count=[df.loc[mask_cl,  "HP"].sum(), df.loc[~mask_cl, "HP"].sum()],
            nobs =[int(mask_cl.sum()), int((~mask_cl).sum())]
        )
        cluster_ztest_rows.append({
            "final_cluster":   cl,
            "cluster_p_value": p_val,
            "cluster_z_stat":  z_stat
        })

    df = df.merge(pd.DataFrame(cluster_ztest_rows), on="final_cluster", how="left")

    df["final_cluster"] = np.where(
        (df["cluster_p_value"] <= 0.05) & (np.abs(df["cluster_z_stat"]) >= 1.96),
        df["final_cluster"],
        "Generic"
    )

    # =========================================================
    # 8. DECISION TREE  ->  leaf-level segmentation
    #    The tree is trained to predict the validated DBSCAN
    #    cluster labels, learning interpretable feature-based
    #    boundaries that approximate the cluster structure.
    #
    #    Hyperparameters are selected with RandomizedSearchCV:
    #      - min_samples_leaf: derived from min_profile_size
    #        (same proportional logic as DBSCAN min_samples)
    #      - max_depth: [None, 6, 10, 15]
    #      - criterion: ["gini", "entropy"]
    # =========================================================
    _drop_dt = {
        id_col, target_score_col,
        "HP", "LP", "TRICT",
        "cluster", "cluster_propensity",
        "final_cluster", "cluster_p_value", "cluster_z_stat"
    }
    X_dt_df = df.drop(columns=list(_drop_dt), errors="ignore")

    dt_num_cols = X_dt_df.select_dtypes(exclude=["object", "category"]).columns.tolist()
    dt_cat_cols = X_dt_df.select_dtypes(include=["object", "category"]).columns.tolist()

    ohe_dt = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
    if dt_cat_cols:
        X_dt_cat_arr = ohe_dt.fit_transform(X_dt_df[dt_cat_cols])
        dt_cat_names = ohe_dt.get_feature_names_out(dt_cat_cols).tolist()
        X_dt_cat_df  = pd.DataFrame(X_dt_cat_arr, columns=dt_cat_names, index=df.index)
    else:
        X_dt_cat_df  = pd.DataFrame(index=df.index)
        dt_cat_names = []

    X_dt          = pd.concat([X_dt_df[dt_num_cols], X_dt_cat_df], axis=1)
    dt_feat_names = list(X_dt.columns)
    X_dt_arr      = X_dt.fillna(X_dt.mean()).values
    Y_dt          = df["final_cluster"].values

    # min_samples_leaf search space mirrors the DBSCAN logic:
    # absolute row counts corresponding to the min_profile_size fraction
    msl_base = max(10, int(min_profile_size * len(df)))
    dt_param_grid = {
        "max_depth":        [None, 6, 10, 15],
        "criterion":        ["gini", "entropy"],
        "min_samples_leaf": [msl_base, msl_base * 2, msl_base * 3],
    }

    dt_search = RandomizedSearchCV(
        DecisionTreeClassifier(random_state=1926),
        dt_param_grid,
        n_iter=15, cv=3, scoring="f1_weighted",
        n_jobs=-1, random_state=1926
    )
    dt_search.fit(X_dt_arr, Y_dt)
    dt = dt_search.best_estimator_

    # ROC-AUC reliability check for the decision tree
    # CV scoring was f1_weighted; re-evaluate as ROC-AUC (OvR) on the same folds
    from sklearn.model_selection import cross_val_score
    _dt_roc = cross_val_score(
        dt, X_dt_arr, Y_dt,
        cv=3, scoring='roc_auc_ovr_weighted', n_jobs=-1
    ).mean()
    if _dt_roc < 0.65:
        print(f"[WARNING] Model DecisionTreeClassifier - not reliable results "
              f"(ROC-AUC OvR weighted = {_dt_roc:.3f} < 0.65)")

    # Compute leaf-level HP propensity for every row
    df["leaf_id"]         = dt.apply(X_dt_arr)
    df["leaf_propensity"] = df.groupby("leaf_id")["HP"].transform("mean")

    # First pass: candidate profiles require propensity > 1.2x baseline
    df["Profile_ID"] = np.where(
        df["leaf_propensity"] / base_propensity > 1.2,
        df["leaf_id"].astype(str),
        "Generic"
    )

    # =========================================================
    # 9. Z-TEST ON PROFILES  ->  Profile_ID final validation
    #    A candidate profile is confirmed only when ALL three
    #    conditions hold simultaneously:
    #      (a) size >= min_profile_size * total rows
    #      (b) p-value <= 0.05
    #      (c) |z-stat| >= 1.96
    #    Profiles that fail any condition are relabelled "Generic".
    # =========================================================
    min_leaf_size = int(min_profile_size * len(df))

    profile_ztest_rows = []
    for pid in df["Profile_ID"].unique():
        mask_pid   = df["Profile_ID"] == pid
        leaf_count = int(mask_pid.sum())
        z_stat, p_val = proportions_ztest(
            count=[df.loc[mask_pid,  "HP"].sum(), df.loc[~mask_pid, "HP"].sum()],
            nobs =[leaf_count, int((~mask_pid).sum())]
        )
        profile_ztest_rows.append({
            "Profile_ID":      pid,
            "profile_count":   leaf_count,
            "profile_p_value": p_val,
            "profile_z_stat":  z_stat
        })

    df = df.merge(pd.DataFrame(profile_ztest_rows), on="Profile_ID", how="left")

    df["Profile_ID"] = np.where(
        (df["profile_count"]   >= min_leaf_size) &
        (df["profile_p_value"] <= 0.05) &
        (np.abs(df["profile_z_stat"]) >= 1.96),
        df["Profile_ID"],
        "Generic"
    )

    # =========================================================
    # 10. RULE EXTRACTION
    #     Traverse the fitted decision tree and collect the full
    #     conjunction of split conditions for every leaf node.
    # =========================================================
    def extract_rules(tree_clf, feat_names):
        cl = tree_clf.tree_.children_left
        cr = tree_clf.tree_.children_right
        ft = tree_clf.tree_.feature
        th = tree_clf.tree_.threshold
        rules = {}

        def recurse(node, path):
            if cl[node] == cr[node]:   # leaf node reached
                rules[node] = " AND ".join(path) if path else "ROOT"
            else:
                fname = feat_names[ft[node]]
                recurse(cl[node], path + [f"{fname} <= {th[node]:.3f}"])
                recurse(cr[node], path + [f"{fname} > {th[node]:.3f}"])

        recurse(0, [])
        return rules

    rules_dict = extract_rules(dt, dt_feat_names)

    df_rules = (
        df[["leaf_id", "Profile_ID"]]
        .drop_duplicates()
        .assign(rule=lambda d: d["leaf_id"].map(rules_dict))
        [["leaf_id", "Profile_ID", "rule"]]
        .sort_values("Profile_ID")
        .reset_index(drop=True)
    )
    df_rules.to_csv(f"{output_path}/PROFILE_RULES.csv", index=False)

    # =========================================================
    # 11. FEATURE IMPORTANCES  (decision tree, non-zero only)
    # =========================================================
    df_importances = (
        pd.DataFrame({"feature": dt_feat_names,
                      "importance": dt.feature_importances_})
        .query("importance > 0")
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    df_importances.to_csv(f"{output_path}/PROFILE_FEATURES.csv", index=False)

    # =========================================================
    # 12. PROFILE STATS  (mean of all numerics per Profile_ID)
    # =========================================================
    df["profile_volume"] = df.groupby("Profile_ID")[id_col].transform("count")
    df["profile_pct"]    = df["profile_volume"] / len(df)

    _exclude_agg = {
        "Profile_ID", id_col,
        "cluster", "leaf_id", "final_cluster",
        "cluster_p_value", "cluster_z_stat",
        "profile_p_value", "profile_z_stat", "profile_count"
    }
    agg_num_cols = [
        c for c in df.select_dtypes(include="number").columns
        if c not in _exclude_agg
    ]

    df_stats = (
        df.groupby("Profile_ID")[agg_num_cols]
        .mean()
        .reset_index()
    )
    df_stats.to_csv(f"{output_path}/PROFILE_STATS.csv", index=False)

    # =========================================================
    # 13. ID -> PROFILE TABLE
    # =========================================================
    df[[id_col, "Profile_ID", "HP", "LP"]].to_csv(
        f"{output_path}/ID_W_PROFILE.csv", index=False
    )

    # =========================================================
    # 14. ELASTIC-NET MULTINOMIAL LOGISTIC REGRESSION
    #     Predicts Profile_ID from original features.
    #     Hyperparameters (C, l1_ratio) are selected via
    #     RandomizedSearchCV.  The resulting coefficients are
    #     exponentiated to produce odds ratios, saved with
    #     readable feature names as column headers.
    #
    #     Binary case: sklearn returns coef_ with shape (1, p)
    #     instead of (2, p); we expand it symmetrically.
    # =========================================================
    _drop_lr = _drop_dt | {
        "Profile_ID",
        "cluster_propensity", "leaf_propensity",
        "profile_p_value", "profile_z_stat", "profile_count",
        "cluster_p_value", "cluster_z_stat",
        "profile_volume", "profile_pct"
    }
    X_lr_df = df.drop(columns=list(_drop_lr), errors="ignore")

    lr_num_cols = X_lr_df.select_dtypes(include=["number"]).columns.tolist()
    lr_cat_cols = X_lr_df.select_dtypes(exclude=["number"]).columns.tolist()

    X_lr_num = (
        SimpleImputer(strategy="mean").fit_transform(X_lr_df[lr_num_cols])
        if lr_num_cols else np.empty((len(X_lr_df), 0))
    )
    if lr_cat_cols:
        ohe_lr       = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
        X_lr_cat     = ohe_lr.fit_transform(X_lr_df[lr_cat_cols].fillna("missing"))
        lr_cat_names = ohe_lr.get_feature_names_out(lr_cat_cols).tolist()
    else:
        X_lr_cat     = np.empty((len(X_lr_df), 0))
        lr_cat_names = []

    lr_all_names = lr_num_cols + lr_cat_names
    X_lr_final   = MinMaxScaler().fit_transform(np.hstack([X_lr_num, X_lr_cat]))
    Y_lr         = df["Profile_ID"].values

    # Elastic-net multinomial logistic regression with RandomizedSearchCV
    lr_param_grid = {
        "C":        [0.01, 0.1, 1.0, 5.0],
        "l1_ratio":  [0.15, 0.5, 1.0],
    }
    lr_search = RandomizedSearchCV(
        LogisticRegression(
            penalty="elasticnet", solver="saga",
            multi_class="multinomial",
            max_iter=5000, random_state=1926
        ),
        lr_param_grid,
        n_iter=15, cv=3, scoring="f1_weighted",
        n_jobs=-1, random_state=1926
    )
    lr_search.fit(X_lr_final, Y_lr)
    log = lr_search.best_estimator_

    # ROC-AUC reliability check for the logistic regression
    _lr_roc = cross_val_score(
        log, X_lr_final, Y_lr,
        cv=3, scoring='roc_auc_ovr_weighted', n_jobs=-1
    ).mean()
    if _lr_roc < 0.65:
        print(f"[WARNING] Model LogisticRegression (elastic-net multinomial) - "
              f"not reliable results (ROC-AUC OvR weighted = {_lr_roc:.3f} < 0.65)")

    # Handle binary case: coef_ shape is (1, p) instead of (2, p)
    if len(log.classes_) == 2 and log.coef_.shape[0] == 1:
        coef_matrix = np.vstack([-log.coef_, log.coef_])
    else:
        coef_matrix = log.coef_

    df_or = pd.DataFrame(
        np.exp(coef_matrix),
        columns=lr_all_names,
        index=log.classes_
    )
    df_or.index.name = "Profile_ID"
    df_or.reset_index(inplace=True)
    df_or.to_csv(f"{output_path}/oddsratio.csv", index=False)

    # =========================================================
    # 15. CHARTS  (temporary PNGs, all embedded in the PDF)
    # =========================================================
    plot_paths    = []
    profile_order = (
        sorted([p for p in df["Profile_ID"].unique() if p != "Generic"]) + ["Generic"]
    )

    # --- 15a. Global SHAP feature importance (top 20, horizontal bar) ---
    shap_df = pd.DataFrame(shap_output).head(20)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(shap_df["feature"][::-1], shap_df["mean_impact"][::-1], color="#4C72B0")
    ax.set_xlabel("Mean |SHAP| value")
    ax.set_title("Global SHAP Feature Importance (top 20)")
    plt.tight_layout()
    p_shap = f"{output_path}/_shap_bar.png"
    fig.savefig(p_shap, dpi=150)
    plt.close(fig)
    plot_paths.append(("Global SHAP Feature Importance", p_shap))

    # --- 15b. 2D PCA scatter coloured by Profile_ID ---
    # Stratified sample capped at 10,000 rows to keep rendering fast
    PCA_SCATTER_CAP = 10_000
    if len(df) > PCA_SCATTER_CAP:
        sample_idx = (
            df[["Profile_ID"]]
            .reset_index()
            .groupby("Profile_ID", group_keys=False)
            .apply(lambda g: g.sample(
                min(len(g), max(1, int(PCA_SCATTER_CAP * len(g) / len(df)))),
                random_state=1926
            ))["index"]
            .values
        )
        X_scaled_plot  = X_scaled[sample_idx]
        profiles_plot  = df["Profile_ID"].values[sample_idx]
    else:
        X_scaled_plot = X_scaled
        profiles_plot = df["Profile_ID"].values

    pca2d   = PCA(n_components=2, random_state=1926)
    X_pca2d = pca2d.fit_transform(X_scaled_plot)
    var_exp = pca2d.explained_variance_ratio_ * 100

    unique_profiles = sorted(set(df["Profile_ID"].values),
                             key=lambda x: (x == "Generic", x))
    cmap_scatter  = cm.get_cmap("tab10", len(unique_profiles))
    profile_color = {p: cmap_scatter(i) for i, p in enumerate(unique_profiles)}

    fig, ax = plt.subplots(figsize=(9, 6))
    for pid in unique_profiles:
        mask = profiles_plot == pid
        ax.scatter(
            X_pca2d[mask, 0], X_pca2d[mask, 1],
            c=[profile_color[pid]], label=str(pid),
            alpha=0.55, s=18, edgecolors="none"
        )
    ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}% var)")
    ax.set_title("2D PCA Projection - coloured by Profile_ID "
                 f"(n={len(X_scaled_plot):,})")
    ax.legend(title="Profile_ID", bbox_to_anchor=(1.02, 1),
              loc="upper left", fontsize=7, markerscale=1.5)
    plt.tight_layout()
    p_pca2d = f"{output_path}/_pca2d_profiles.png"
    fig.savefig(p_pca2d, dpi=150)
    plt.close(fig)
    plot_paths.append(("2D PCA Projection by Profile_ID", p_pca2d))

    # --- 15c. Mean of each numeric variable per profile (bar chart) ---
    plot_num_cols = [
        c for c in raw_num_cols
        if c in df_stats.columns and c not in {"HP", "LP", "TRICT"}
    ]
    cmap_bar = cm.get_cmap("tab10", len(profile_order))

    for col in plot_num_cols[:10]:   # cap at 10 variables
        vals = [
            df_stats.loc[df_stats["Profile_ID"] == pid, col].values[0]
            if pid in df_stats["Profile_ID"].values else np.nan
            for pid in profile_order
        ]
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(profile_order, vals,
               color=[cmap_bar(i) for i in range(len(profile_order))])
        ax.set_title(f"Avg '{col}' by Profile")
        ax.set_xlabel("Profile_ID")
        ax.set_ylabel(f"Mean {col}")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        p_num = f"{output_path}/_num_{col}.png"
        fig.savefig(p_num, dpi=150)
        plt.close(fig)
        plot_paths.append((f"Numeric: {col} by Profile", p_num))

    # --- 15d. Categorical distribution per profile (stacked bar, row %) ---
    for col in raw_cat_cols[:5]:     # cap at 5 categorical variables
        ct = pd.crosstab(df["Profile_ID"], df[col], normalize="index")
        fig, ax = plt.subplots(figsize=(9, 4))
        ct.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
        ax.set_title(f"'{col}' distribution by Profile (row %)")
        ax.set_xlabel("Profile_ID")
        ax.set_ylabel("Proportion")
        ax.legend(title=col, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        p_cat = f"{output_path}/_cat_{col}.png"
        fig.savefig(p_cat, dpi=150)
        plt.close(fig)
        plot_paths.append((f"Categorical: {col} by Profile", p_cat))

    # --- 15e. Odds-ratio diverging bar charts (one chart per named profile) ---
    # For each named profile (excluding "Generic") we show only the features
    # whose OR is significantly different from 1, defined as:
    #   OR > 1.1  (green bar extending right)
    #   OR < 0.9  (red bar extending left)
    # Bars represent log(OR) so that symmetric scale is preserved.
    or_feat_cols = [c for c in df_or.columns if c != "Profile_ID"]

    for pid in profile_order:
        if pid == "Generic":
            continue

        row = df_or[df_or["Profile_ID"] == pid]
        if row.empty:
            continue

        or_vals  = row[or_feat_cols].values.flatten().astype(float)
        log_or   = np.log(or_vals)  # log scale for diverging axis

        # Keep only features with OR meaningfully different from 1
        sig_mask = np.abs(log_or) > np.log(1.1)
        if sig_mask.sum() == 0:
            continue

        sig_features = np.array(or_feat_cols)[sig_mask]
        sig_log_or   = log_or[sig_mask]

        # Sort by log(OR) descending for readability
        order      = np.argsort(sig_log_or)[::-1]
        sig_features = sig_features[order]
        sig_log_or   = sig_log_or[order]

        bar_colors = ["#2ca02c" if v > 0 else "#d62728" for v in sig_log_or]

        fig, ax = plt.subplots(figsize=(10, max(4, len(sig_features) * 0.35)))
        ax.barh(sig_features, sig_log_or, color=bar_colors)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("log(Odds Ratio)  [green = OR > 1, red = OR < 1]")
        ax.set_title(f"Significant Odds Ratios — Profile {pid}")
        plt.tight_layout()
        p_or = f"{output_path}/_or_profile_{pid}.png"
        fig.savefig(p_or, dpi=150)
        plt.close(fig)
        plot_paths.append((f"Odds Ratios: Profile {pid}", p_or))

    # =========================================================
    # 16. PDF REPORT
    # =========================================================
    pdf_path = f"{output_path}/InsightForge_Report.pdf"
    doc      = SimpleDocTemplate(
        pdf_path, pagesize=A4,
        leftMargin=2*rl_cm, rightMargin=2*rl_cm,
        topMargin=2*rl_cm,  bottomMargin=2*rl_cm
    )
    styles        = getSampleStyleSheet()
    h1, h2, body  = styles["Heading1"], styles["Heading2"], styles["Normal"]
    story         = []

    # Cover page
    story.append(Spacer(1, 2*rl_cm))
    story.append(Paragraph("InsightForge - Analytical Report", h1))
    story.append(Spacer(1, 0.5*rl_cm))
    story.append(Paragraph(
        f"File: <b>{os.path.basename(file_path)}</b> &nbsp;|&nbsp; "
        f"Score column: <b>{target_score_col}</b> &nbsp;|&nbsp; "
        f"Threshold: <b>{threshold}th pct</b> &nbsp;|&nbsp; "
        f"DBSCAN eps: <b>{eps}</b>",
        body
    ))
    story.append(Spacer(1, 0.4*rl_cm))

    # Profile summary table (cover page)
    summary_data = [["Profile_ID", "N", "% Total", "HP Rate"]]
    for pid in profile_order:
        sub = df[df["Profile_ID"] == pid]
        summary_data.append([
            str(pid),
            str(len(sub)),
            f"{len(sub) / len(df) * 100:.1f}%",
            f"{sub['HP'].mean() * 100:.1f}%"
        ])

    tbl = Table(summary_data, hAlign="LEFT")
    tbl.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0),  colors.HexColor("#4C72B0")),
        ("TEXTCOLOR",      (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",       (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#EEF2F8"), colors.white]),
        ("GRID",           (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE",       (0, 0), (-1, -1), 9),
        ("PADDING",        (0, 0), (-1, -1), 5),
    ]))
    story.append(tbl)
    story.append(PageBreak())

    # Insert all charts with adaptive height:
    # OR charts may be taller; others use a fixed aspect ratio
    page_w = A4[0] - 4*rl_cm
    for title_txt, img_path in plot_paths:
        story.append(Paragraph(title_txt, h2))
        story.append(Spacer(1, 0.2*rl_cm))
        # OR charts are rendered taller to fit many features
        aspect = 0.80 if title_txt.startswith("Odds Ratios") else 0.50
        story.append(RLImage(img_path, width=page_w, height=page_w * aspect))
        story.append(Spacer(1, 0.6*rl_cm))

    doc.build(story)

    # Remove all temporary PNG files
    for _, img_path in plot_paths:
        try:
            os.remove(img_path)
        except OSError:
            pass

    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("PROFILES GENERATED: "+str(df_stats["Profile_ID"].nunique()))
    print(f"  -> Outputs saved in: {output_path}/")

    return {
        "shap_importance":  shap_output,
        "profile_stats":    df_stats,
        "profile_rules":    df_rules,
        "profile_features": df_importances,
        "oddsratio":        df_or,
        "id_with_profile":  df[[id_col, "Profile_ID", "HP", "LP"]],
        "pdf_report":       pdf_path,
    }

