#!/usr/bin/env python3
"""
=============================================================================
Facial Redness & Emotion Perception — Replication Analysis Script
=============================================================================
Conceptual replication of Wolf, Leder, Röseler & Schütz (2021),
"Does facial redness really affect emotion perception?"
Cognition and Emotion, 35(8), 1607-1617.

Requirements:
    pip install pandas numpy statsmodels bambi arviz matplotlib

This script is designed to run on both the current (incomplete) dataset
and the final complete dataset without modification.
=============================================================================
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
import numpy as np
import os

OUTPUT_DIR = "results" 
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Statistical packages — imported with graceful fallback
try:
    import statsmodels.formula.api as smf
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("[WARNING] statsmodels not installed. Frequentist pipeline will be skipped.")
    print("  Install with: pip install statsmodels\n")

try:
    import bambi as bmb
    import arviz as az
    HAS_BAYESIAN = True
except ImportError:
    HAS_BAYESIAN = False
    print("[WARNING] bambi/arviz not installed. Bayesian pipeline will be skipped.")
    print("  Install with: pip install bambi arviz\n")

# ============================================================================
# 1. DATA IMPORT & QUALTRICS CLEANING
# ============================================================================

DATA_PATH = "data.csv"  # <-- Adjust path if needed

print("=" * 72)
print("STEP 1: DATA IMPORT & CLEANING")
print("=" * 72)

# --- Step 1: Load CSV; Qualtrics row architecture ---
# Row 0 = real column headers (preserved automatically by pandas).
# Rows 0-1 of the DATA (i.e., indices 0 & 1 after read) contain
# Qualtrics question text and internal JSON Import IDs. Discard them.
df = pd.read_csv(DATA_PATH)
print(f"  Raw import shape: {df.shape}")
df = df.iloc[2:].reset_index(drop=True)
print(f"  After removing Qualtrics metadata rows: {df.shape}")

# --- Step 2: Type conversion (CRITICAL) ---
# Because the discarded rows contained text, pandas defaults to object dtype
# for numeric columns. Coerce them now.
df["Progress"] = pd.to_numeric(df["Progress"], errors="coerce")
df["Finished"] = pd.to_numeric(df["Finished"], errors="coerce")
df["consent_form"] = pd.to_numeric(df["consent_form"], errors="coerce")

# Convert ALL rating columns (start with 'WF-' or 'WM-') to numeric
rating_cols = [c for c in df.columns if c.startswith("WF-") or c.startswith("WM-")]
for col in rating_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
print(f"  Converted {len(rating_cols)} rating columns to numeric.")

# --- Step 3: Date filter (>= March 19, 2026) ---
df["StartDate"] = pd.to_datetime(df["StartDate"], errors="coerce")
df = df[df["StartDate"] >= "2026-03-19"].reset_index(drop=True)
print(f"  After date filter (>= 2026-03-19): {df.shape[0]} rows")

# --- Step 4: Standardise participant IDs (known typos) ---
df["participantID"] = df["participantID"].replace({"ZP009": "P009", "Po27": "P027"})
print(f"  Participant ID typos corrected (ZP009→P009, Po27→P027).")

# --- Step 5: Consent filter ---
# Qualtrics recodes: 1.0 = consent given; 2.0 = not given; NaN = missing
df = df[df["consent_form"] == 1.0].reset_index(drop=True)
print(f"  After consent filter: {df.shape[0]} rows")

# --- Step 6: Completion filter ---
df = df[(df["Finished"] == 1.0) | (df["Progress"] == 100.0)].reset_index(drop=True)
print(f"  After completion filter: {df.shape[0]} rows")

# --- Step 7: Deduplication (keep most recent valid attempt) ---
df = df.sort_values("StartDate", ascending=False)
df = df.drop_duplicates(subset="participantID", keep="first").reset_index(drop=True)
print(f"  After deduplication: {df.shape[0]} unique participants")

# --- Step 8: Exclude P046 (colorblindness exclusion) ---
n_before = df.shape[0]
df = df[df["participantID"] != "P046"].reset_index(drop=True)
n_excluded = n_before - df.shape[0]
print(f"  Excluded P046: {n_excluded} row(s) removed. Remaining: {df.shape[0]}")

# --- Step 9: Column pruning ---
# Drop ishihara columns, Qualtrics metadata, and target-assignment columns
ishihara_cols = [c for c in df.columns if "ishihara" in c.lower()]
qualtrics_meta = [
    "EndDate", "Status", "Duration (in seconds)", "RecordedDate",
    "ResponseId", "DistributionChannel", "UserLanguage",
    "consent_form", "gender", "age", "nationality", "occupation_1",
    "english_fluency_1", "Q95", "Finished", "Progress", "StartDate",
]
target_assignment_cols = [c for c in df.columns if c.startswith("target")]
cols_to_drop = [c for c in ishihara_cols + qualtrics_meta + target_assignment_cols if c in df.columns]
df = df.drop(columns=cols_to_drop)
print(f"  After column pruning: {df.shape} (participantID + {len(rating_cols)} rating cols)")
print(f"  Final N = {df['participantID'].nunique()} participants")
print()

# ============================================================================
# 2. DATA RESHAPING & VARIABLE ENGINEERING
# ============================================================================

print("=" * 72)
print("STEP 2: RESHAPE TO LONG FORMAT & ENGINEER VARIABLES")
print("=" * 72)

# Identify all rating columns present in the cleaned data
value_cols = [c for c in df.columns if c.startswith("WF-") or c.startswith("WM-")]

# Pivot to long format
df_long = df.melt(
    id_vars=["participantID"],
    value_vars=value_cols,
    var_name="item",
    value_name="rating",
)

# Drop NaN ratings (images not shown to this participant due to randomisation)
df_long = df_long.dropna(subset=["rating"]).reset_index(drop=True)
print(f"  Long format (after dropping NaN): {df_long.shape[0]} observations")

# Extract experimental variables from column names using regex
# Format: targetType-targetNumber-facialColoration-shownEmotion_emotionChoiceID
# e.g.  WF-001-NR-N_1
pattern = r"^(W[FM])-(\d+)-(NR|R)-([A-Z]+)_(\d)$"
extracted = df_long["item"].str.extract(pattern)
extracted.columns = [
    "targetType", "targetNumber", "facialColoration", "shownEmotion", "emotionChoiceID"
]
df_long = pd.concat([df_long, extracted], axis=1)
df_long = df_long.dropna(subset=["targetType"]).reset_index(drop=True)
print(f"  After regex extraction: {df_long.shape[0]} observations")

# Map target gender from targetType
df_long["target_gender"] = df_long["targetType"].map({"WF": "Female", "WM": "Male"})

# Map emotionChoiceID to readable labels
emotion_choice_labels = {
    "1": "Happy", "2": "Sad", "3": "Surprised",
    "4": "Angry", "5": "Disgusted", "6": "Scared",
}
df_long["ratedEmotion"] = df_long["emotionChoiceID"].map(emotion_choice_labels)

# Ensure rating is numeric integer
df_long["rating"] = df_long["rating"].astype(int)

# Print summary
print(f"  Target types present: {sorted(df_long['targetType'].unique())}")
print(f"  Target genders present: {sorted(df_long['target_gender'].unique())}")
print(f"  Shown emotions: {sorted(df_long['shownEmotion'].unique())}")
print(f"  Facial colorations: {sorted(df_long['facialColoration'].unique())}")
print(f"  Emotion choice IDs: {sorted(df_long['emotionChoiceID'].unique())}")
print(f"  Unique participants: {df_long['participantID'].nunique()}")
print(f"  Unique targets: {df_long['targetNumber'].nunique()}")
print()

# ============================================================================
# 3. CREATE CONGRUENT RATING DATAFRAME
# ============================================================================

print("=" * 72)
print("STEP 3: CONGRUENT RATING FILTER")
print("=" * 72)

# For the main hypotheses, we only use the rating corresponding to the
# depicted emotion (or the hypothesised emotion for neutral faces):
#   A  (angry face)   → keep _4 (Angry rating)
#   F  (fearful face) → keep _6 (Scared rating)
#   HC (happy face)   → keep _1 (Happy rating)
#   N  (neutral face) → keep _4 (Angry rating)  ← tests H4
congruent_map = {"A": "4", "F": "6", "HC": "1", "N": "4"}
df_long["congruent_choice"] = df_long["shownEmotion"].map(congruent_map)
df_congruent = df_long[df_long["emotionChoiceID"] == df_long["congruent_choice"]].copy()
df_congruent = df_congruent.reset_index(drop=True)

print(f"  Congruent observations: {df_congruent.shape[0]}")
print(f"\n  Condition breakdown (N obs, Mean rating, SD):")
summary = (
    df_congruent
    .groupby(["shownEmotion", "facialColoration"])["rating"]
    .agg(["count", "mean", "std"])
    .round(3)
)
print(summary.to_string(index=True))
print()

# ============================================================================
# 4. DESCRIPTIVE STATISTICS
# ============================================================================

print("=" * 72)
print("STEP 4: DESCRIPTIVE STATISTICS")
print("=" * 72)

print("\n--- Full dataset (all 6 rated emotions) ---")
desc_full = (
    df_long
    .groupby(["shownEmotion", "facialColoration", "ratedEmotion"])["rating"]
    .agg(["count", "mean", "std"])
    .round(3)
)
print(desc_full.to_string())

print("\n--- Congruent ratings by shownEmotion × facialColoration × target_gender ---")
desc_cong = (
    df_congruent
    .groupby(["shownEmotion", "facialColoration", "target_gender"])["rating"]
    .agg(["count", "mean", "std"])
    .round(3)
)
print(desc_cong.to_string())
print()

# ============================================================================
# 5. STATISTICAL ANALYSIS — FREQUENTIST (statsmodels)
# ============================================================================

if HAS_STATSMODELS:
    print("=" * 72)
    print("STEP 5: FREQUENTIST MIXED-EFFECTS MODELS (statsmodels)")
    print("=" * 72)

    # ------------------------------------
    # Prepare data for modelling
    # ------------------------------------
    mod_df = df_congruent.copy()

    # Contrast-code facialColoration: NR = 0, R = 1
    mod_df["color_R"] = (mod_df["facialColoration"] == "R").astype(int)

    # Create dummy variables for shownEmotion (reference = A)
    mod_df["emo_F"]  = (mod_df["shownEmotion"] == "F").astype(int)
    mod_df["emo_HC"] = (mod_df["shownEmotion"] == "HC").astype(int)
    mod_df["emo_N"]  = (mod_df["shownEmotion"] == "N").astype(int)

    # Dynamically check if target_gender has >1 level
    n_genders = mod_df["target_gender"].nunique()
    gender_term = " + C(target_gender)" if n_genders > 1 else ""
    if n_genders == 1:
        print(f"  [NOTE] Only one target_gender level present ({mod_df['target_gender'].unique()[0]}).")
        print(f"         target_gender is dropped from models to avoid singular matrix.\n")

    # ------------------------------------
    # Full interaction model
    # ------------------------------------
    print("--- Model: rating ~ facialColoration * shownEmotion (+ target_gender) ---\n")

    formula = f"rating ~ color_R * (emo_F + emo_HC + emo_N){gender_term}"

    # Try with random intercepts for both participant and target
    try:
        # statsmodels mixedlm only supports one grouping factor natively.
        # We use participantID as the primary grouping variable.
        model = smf.mixedlm(
            formula,
            data=mod_df,
            groups=mod_df["participantID"],
        )
        result = model.fit(reml=True)
        print(result.summary())
    except Exception as e:
        print(f"  [ERROR] Model fitting failed: {e}")
        result = None

    # ------------------------------------
    # Planned contrasts (H1–H4)
    # ------------------------------------
    if result is not None:
        print("\n" + "-" * 50)
        print("PLANNED CONTRASTS: R vs NR within each emotion")
        print("-" * 50)

        # The interaction model parameterises the effect of color_R as:
        #   For A  (reference): coef of color_R
        #   For F:  coef of color_R + coef of color_R:emo_F
        #   For HC: coef of color_R + coef of color_R:emo_HC
        #   For N:  coef of color_R + coef of color_R:emo_N

        params = result.params
        se = result.bse
        pvals = result.pvalues

        contrasts = {}
        # H1: Red effect on Angry faces (reference level)
        if "color_R" in params:
            contrasts["H1 (Angry: R vs NR)"] = {
                "estimate": params["color_R"],
                "SE": se["color_R"],
                "p": pvals["color_R"],
            }

        # H2: Red effect on Fearful faces
        if "color_R:emo_F" in params:
            est = params["color_R"] + params["color_R:emo_F"]
            # Approximate SE via delta method (conservative)
            contrasts["H2 (Fear: R vs NR)"] = {
                "estimate": est,
                "interaction_p": pvals["color_R:emo_F"],
            }

        # H3: Red effect on Happy faces
        if "color_R:emo_HC" in params:
            est = params["color_R"] + params["color_R:emo_HC"]
            contrasts["H3 (Happy: R vs NR)"] = {
                "estimate": est,
                "interaction_p": pvals["color_R:emo_HC"],
            }

        # H4: Red effect on Neutral faces (anger rating)
        if "color_R:emo_N" in params:
            est = params["color_R"] + params["color_R:emo_N"]
            contrasts["H4 (Neutral: R vs NR)"] = {
                "estimate": est,
                "interaction_p": pvals["color_R:emo_N"],
            }

        for label, vals in contrasts.items():
            print(f"\n  {label}:")
            for k, v in vals.items():
                if "p" in k:
                    print(f"    {k}: {v:.4f} {'*' if v < .05 else 'n.s.'}")
                else:
                    print(f"    {k}: {v:.4f}")

    # ------------------------------------
    # Alternative: Simple within-emotion t-tests (supplementary)
    # ------------------------------------
    print("\n\n--- Supplementary: Paired within-emotion comparisons (R vs NR) ---")
    from scipy import stats

    for emo in ["A", "F", "HC", "N"]:
        emo_data = df_congruent[df_congruent["shownEmotion"] == emo]
        # Aggregate to participant-level means within each coloration
        agg = emo_data.groupby(["participantID", "facialColoration"])["rating"].mean().unstack()
        if "NR" in agg.columns and "R" in agg.columns:
            paired = agg.dropna()
            if len(paired) > 1:
                t_stat, p_val = stats.ttest_rel(paired["R"], paired["NR"])
                d = (paired["R"] - paired["NR"]).mean() / (paired["R"] - paired["NR"]).std()
                emo_label = {"A": "Angry", "F": "Fearful", "HC": "Happy", "N": "Neutral"}[emo]
                print(f"  {emo_label}: M(R)={paired['R'].mean():.3f}, M(NR)={paired['NR'].mean():.3f}, "
                      f"t({len(paired)-1})={t_stat:.3f}, p={p_val:.4f}, d={d:.3f}")
    print()

else:
    print("\n[SKIPPED] Frequentist mixed-effects model (statsmodels not available).\n")

# ============================================================================
# 5b. FREQUENTIST TESTS — SCIPY FALLBACK (always runs)
# ============================================================================

from scipy import stats

print("=" * 72)
print("STEP 5b: PAIRED t-TESTS (scipy — R vs NR per emotion condition)")
print("=" * 72)

if df_congruent["participantID"].nunique() < df.shape[0]:
    dropped = set(df["participantID"]) - set(df_congruent["participantID"])
    if dropped:
        print(f"  [NOTE] Participants with no valid ratings excluded: {dropped}")

emo_labels = {"A": "Angry", "F": "Fearful", "HC": "Happy", "N": "Neutral"}
hyp_labels = {"A": "H1", "F": "H2", "HC": "H3", "N": "H4"}
expectations = {
    "A":  "Expect R > NR (significant)",
    "F":  "Expect no significant difference",
    "HC": "Expect R > NR (significant)",
    "N":  "Expect R > NR for anger ratings (significant)",
}

results_rows = []
for emo in ["A", "F", "HC", "N"]:
    sub = df_congruent[df_congruent["shownEmotion"] == emo]
    # Aggregate to participant-level means per coloration
    r_means = sub[sub["facialColoration"] == "R"].groupby("participantID")["rating"].mean()
    nr_means = sub[sub["facialColoration"] == "NR"].groupby("participantID")["rating"].mean()
    paired = pd.DataFrame({"R": r_means, "NR": nr_means}).dropna()

    if len(paired) > 1:
        diff = paired["R"] - paired["NR"]
        t_stat, p_val = stats.ttest_rel(paired["R"], paired["NR"])
        d = diff.mean() / diff.std()
        ci_low = diff.mean() - 1.96 * diff.sem()
        ci_high = diff.mean() + 1.96 * diff.sem()
        sig = "***" if p_val < .001 else "**" if p_val < .01 else "*" if p_val < .05 else "n.s."

        print(f"\n  {hyp_labels[emo]}: {emo_labels[emo]} faces")
        print(f"    Prediction: {expectations[emo]}")
        print(f"    M(R) = {paired['R'].mean():.3f}, M(NR) = {paired['NR'].mean():.3f}, "
              f"Diff = {diff.mean():.3f}")
        print(f"    95% CI of diff: [{ci_low:.3f}, {ci_high:.3f}]")
        print(f"    t({len(paired)-1}) = {t_stat:.3f}, p = {p_val:.4f} {sig}")
        print(f"    Cohen's d = {d:.3f}")

        results_rows.append({
            "Hypothesis": hyp_labels[emo], "Emotion": emo_labels[emo],
            "M_Red": round(paired["R"].mean(), 3), "M_Normal": round(paired["NR"].mean(), 3),
            "Diff": round(diff.mean(), 3), "CI_low": round(ci_low, 3), "CI_high": round(ci_high, 3),
            "t": round(t_stat, 3), "df": len(paired) - 1,
            "p": round(p_val, 4), "d": round(d, 3),
        })

results_df = pd.DataFrame(results_rows)
print("\n\n--- Summary Table ---")
print(results_df.to_string(index=False))

# ---- Manipulation check ----
print("\n\n--- Manipulation Check: Congruent vs Incongruent ratings ---")
# emotionChoiceID may be str or int depending on context; convert to str for safe matching
eid_str = df_long["emotionChoiceID"].astype(str)
cong_pairs = {("A", "4"), ("F", "6"), ("HC", "1")}
df_long["is_congruent"] = [
    1 if (se, ec) in cong_pairs else 0
    for se, ec in zip(df_long["shownEmotion"], eid_str)
]
mc_data = df_long[df_long["shownEmotion"] != "N"].copy()
cong_m = mc_data[mc_data["is_congruent"] == 1].groupby("participantID")["rating"].mean()
incong_m = mc_data[mc_data["is_congruent"] == 0].groupby("participantID")["rating"].mean()
mc_paired = pd.DataFrame({"Congruent": cong_m, "Incongruent": incong_m}).dropna()

t_mc, p_mc = stats.ttest_rel(mc_paired["Congruent"], mc_paired["Incongruent"])
diff_mc = mc_paired["Congruent"] - mc_paired["Incongruent"]
d_mc = diff_mc.mean() / diff_mc.std()

print(f"  M(Congruent) = {mc_paired['Congruent'].mean():.3f}, "
      f"M(Incongruent) = {mc_paired['Incongruent'].mean():.3f}")
print(f"  t({len(mc_paired)-1}) = {t_mc:.3f}, p = {p_mc:.2e}, d = {d_mc:.3f}")
print(f"  -> Participants {'DO' if p_mc < .05 else 'DO NOT'} reliably distinguish emotions.")

# ---- Exploratory: All 24 cells ----
print("\n\n--- Exploratory: R vs NR for all rated emotions x all face types ---")
choice_labels = {"1": "Happy", "2": "Sad", "3": "Surprised", "4": "Angry", "5": "Disgusted", "6": "Scared"}
df_long["_eid_str"] = df_long["emotionChoiceID"].astype(str)
for face_emo in ["A", "F", "HC", "N"]:
    sub = df_long[df_long["shownEmotion"] == face_emo]
    for cid, clabel in choice_labels.items():
        sub2 = sub[sub["_eid_str"] == cid]
        r_m = sub2[sub2["facialColoration"] == "R"].groupby("participantID")["rating"].mean()
        nr_m = sub2[sub2["facialColoration"] == "NR"].groupby("participantID")["rating"].mean()
        both = pd.DataFrame({"R": r_m, "NR": nr_m}).dropna()
        if len(both) > 1:
            diff = both["R"] - both["NR"]
            t, pv = stats.ttest_rel(both["R"], both["NR"])
            d = diff.mean() / diff.std() if diff.std() > 0 else 0
            sig = "*" if pv < .05 else ""
            print(f"  {emo_labels[face_emo]:>8} face, {clabel:>10} rating: "
                  f"D={diff.mean():+.3f}, d={d:+.3f}, p={pv:.3f} {sig}")
    print()

results_df.to_csv(os.path.join(OUTPUT_DIR, "hypothesis_test_results.csv"), index=False)
print(f"  Saved: {os.path.join(OUTPUT_DIR, 'hypothesis_test_results.csv')}\n")

# Clean up temp columns
df_long.drop(columns=["is_congruent", "_eid_str", "congruent_choice"], errors="ignore", inplace=True)


# ============================================================================
# 6. STATISTICAL ANALYSIS — BAYESIAN (bambi + arviz)
# ============================================================================

if HAS_BAYESIAN:
    print("=" * 72)
    print("STEP 6: BAYESIAN CUMULATIVE PROBIT MODEL (bambi + arviz)")
    print("=" * 72)

    bay_df = df_congruent.copy()

    # CRITICAL: Cast rating to ordered Categorical to prevent tuple indexing errors
    bay_df["rating"] = pd.Categorical(bay_df["rating"], categories=[1, 2, 3, 4, 5], ordered=True)

    # Dynamically build formula based on available target genders
    n_genders = bay_df["target_gender"].nunique()
    if n_genders > 1:
        bay_formula = "rating ~ facialColoration * shownEmotion + target_gender"
    else:
        bay_formula = "rating ~ facialColoration * shownEmotion"
        print(f"  [NOTE] Only one target_gender level. Dropping from Bayesian model.\n")

    print(f"  Formula: {bay_formula}")
    print(f"  Family:  cumulative (probit link)")
    print(f"  N obs:   {bay_df.shape[0]}\n")

    # ---- Fit Bayesian cumulative probit model ----
    try:
        model_bayes = bmb.Model(
            bay_formula,
            data=bay_df,
            family="cumulative",
            link="probit",
        )

        # Sample with NUTS (multi-core)
        print("  Fitting model (this may take several minutes)...")
        idata = model_bayes.fit(
            draws=2000,
            tune=1000,
            cores=4,
            chains=4,
            random_seed=42,
        )

        # ---- Results ----
        print("\n--- Posterior Summary (95% HDI) ---\n")
        summary = az.summary(idata, hdi_prob=0.95)
        print(summary.to_string())

        # ---- Posterior Predictive Checks ----
        print("\n  Generating posterior predictive checks...")
        model_bayes.predict(idata, kind="pps")
        fig_ppc = az.plot_ppc(idata, num_pp_samples=100)
        try:
            import matplotlib.pyplot as plt
            plt.savefig(os.path.join(OUTPUT_DIR, "posterior_predictive_check.png"), dpi=150, bbox_inches="tight")
            print(f"  Saved: {os.path.join(OUTPUT_DIR, 'posterior_predictive_check.png')}")
        except Exception:
            print("  [NOTE] Could not save PPC plot (matplotlib issue).")

        # ---- Interpretation of planned contrasts ----
        print("\n--- Hypothesis Testing via Credibility Intervals ---")
        print("  Effects whose 95% HDI excludes zero are credibly different from zero.")
        # Inspect interaction terms
        for param_name in summary.index:
            if "facialColoration" in str(param_name):
                row = summary.loc[param_name]
                hdi_low = row.get("hdi_2.5%", row.get("hdi_3%", None))
                hdi_high = row.get("hdi_97.5%", row.get("hdi_97%", None))
                excludes_zero = (hdi_low is not None and hdi_high is not None
                                 and not (hdi_low <= 0 <= hdi_high))
                status = "CREDIBLE (excludes 0)" if excludes_zero else "includes 0"
                print(f"    {param_name}: mean={row['mean']:.4f}, "
                      f"HDI=[{hdi_low:.4f}, {hdi_high:.4f}] → {status}")

    except Exception as e:
        print(f"  [ERROR] Bayesian model fitting failed: {e}")
        import traceback
        traceback.print_exc()

else:
    print("\n[SKIPPED] Bayesian pipeline (bambi/arviz not available).\n")


# ============================================================================
# 7. SAVE PROCESSED DATA
# ============================================================================

print("=" * 72)
print("STEP 7: SAVE PROCESSED DATA")
print("=" * 72)

df_long.to_csv(os.path.join(OUTPUT_DIR, "df_long_all_ratings.csv"), index=False)
print(f"  Saved: {os.path.join(OUTPUT_DIR, 'df_long_all_ratings.csv')} ({df_long.shape[0]} rows)")

df_congruent.to_csv(os.path.join(OUTPUT_DIR, "df_congruent_ratings.csv"), index=False)
print(f"  Saved: {os.path.join(OUTPUT_DIR, 'df_congruent_ratings.csv')} ({df_congruent.shape[0]} rows)")

print()
print("=" * 72)
print("ANALYSIS COMPLETE")
print(f"  Final N = {df_congruent['participantID'].nunique()} participants")
print(f"  Total congruent observations = {df_congruent.shape[0]}")
print("=" * 72)