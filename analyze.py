#!/usr/bin/env python3
"""
=============================================================================
Facial Redness & Emotion Perception — Replication Analysis Script
=============================================================================
Conceptual replication of Wolf, Leder, Röseler & Schütz (2021),
"Does facial redness really affect emotion perception?"
Cognition and Emotion, 35(8), 1607-1617.

Requirements:
    pip install pandas numpy statsmodels bambi arviz matplotlib seaborn

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
import sys

OUTPUT_DIR = "results" 
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- NEW LOGGING CODE ---
class Logger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # This handles the flush command by doing nothing, 
        # required for Python 3 compatibility.
        self.terminal.flush()
        self.log.flush()

# Redirect standard output to both the terminal and your new txt file
log_path = os.path.join(OUTPUT_DIR, "execution_log.txt")
sys.stdout = Logger(log_path)
# ------------------------

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
df = pd.read_csv(DATA_PATH)
print(f"  Raw import shape: {df.shape}")
df = df.iloc[2:].reset_index(drop=True)
print(f"  After removing Qualtrics metadata rows: {df.shape}")

# --- Step 2: Type conversion (CRITICAL) ---
df["Progress"] = pd.to_numeric(df["Progress"], errors="coerce")
df["Finished"] = pd.to_numeric(df["Finished"], errors="coerce")
df["consent_form"] = pd.to_numeric(df["consent_form"], errors="coerce")

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

# --- Step 8.5: Extract Demographics (Updated) ---
# Added 'occupation_1' to the extraction list
df_demographics = df[['participantID', 'gender', 'age', 'occupation_1']].copy()
df_demographics['age'] = pd.to_numeric(df_demographics['age'], errors='coerce')

# Map numeric Qualtrics codes to labels
gender_map = {1: "Female", 2: "Male", 3: "Non-binary", 4: "Other"}
df_demographics['gender_label'] = pd.to_numeric(df_demographics['gender'], errors='coerce').map(gender_map).fillna("Unknown")

# --- Step 9: Column pruning ---
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

value_cols = [c for c in df.columns if c.startswith("WF-") or c.startswith("WM-")]

df_long = df.melt(
    id_vars=["participantID"],
    value_vars=value_cols,
    var_name="item",
    value_name="rating",
)

df_long = df_long.dropna(subset=["rating"]).reset_index(drop=True)
print(f"  Long format (after dropping NaN): {df_long.shape[0]} observations")

pattern = r"^(W[FM])-(\d+)-(NR|R)-([A-Z]+)_(\d)$"
extracted = df_long["item"].str.extract(pattern)
extracted.columns = [
    "targetType", "targetNumber", "facialColoration", "shownEmotion", "emotionChoiceID"
]
df_long = pd.concat([df_long, extracted], axis=1)
df_long = df_long.dropna(subset=["targetType"]).reset_index(drop=True)
print(f"  After regex extraction: {df_long.shape[0]} observations")

df_long["target_gender"] = df_long["targetType"].map({"WF": "Female", "WM": "Male"})

emotion_choice_labels = {
    "1": "Happy", "2": "Sad", "3": "Surprised",
    "4": "Angry", "5": "Disgusted", "6": "Scared",
}
df_long["ratedEmotion"] = df_long["emotionChoiceID"].map(emotion_choice_labels)
df_long["rating"] = df_long["rating"].astype(int)

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

    mod_df = df_congruent.copy()
    mod_df["color_R"] = (mod_df["facialColoration"] == "R").astype(int)

    mod_df["emo_F"]  = (mod_df["shownEmotion"] == "F").astype(int)
    mod_df["emo_HC"] = (mod_df["shownEmotion"] == "HC").astype(int)
    mod_df["emo_N"]  = (mod_df["shownEmotion"] == "N").astype(int)

    n_genders = mod_df["target_gender"].nunique()
    gender_term = " + C(target_gender)" if n_genders > 1 else ""
    if n_genders == 1:
        print(f"  [NOTE] Only one target_gender level present ({mod_df['target_gender'].unique()[0]}).")
        print(f"         target_gender is dropped from models to avoid singular matrix.\n")

    print("--- Model: rating ~ facialColoration * shownEmotion (+ target_gender) ---\n")
    formula = f"rating ~ color_R * (emo_F + emo_HC + emo_N){gender_term}"

    try:
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

    if result is not None:
        print("\n" + "-" * 50)
        print("PLANNED CONTRASTS: R vs NR within each emotion")
        print("-" * 50)

        params = result.params
        se = result.bse
        pvals = result.pvalues

        contrasts = {}
        if "color_R" in params:
            contrasts["H1 (Angry: R vs NR)"] = {
                "estimate": params["color_R"],
                "SE": se["color_R"],
                "p": pvals["color_R"],
            }

        if "color_R:emo_F" in params:
            est = params["color_R"] + params["color_R:emo_F"]
            contrasts["H2 (Fear: R vs NR)"] = {
                "estimate": est,
                "interaction_p": pvals["color_R:emo_F"],
            }

        if "color_R:emo_HC" in params:
            est = params["color_R"] + params["color_R:emo_HC"]
            contrasts["H3 (Happy: R vs NR)"] = {
                "estimate": est,
                "interaction_p": pvals["color_R:emo_HC"],
            }

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

    print("\n\n--- Supplementary: Paired within-emotion comparisons (R vs NR) ---")
    from scipy import stats
    for emo in ["A", "F", "HC", "N"]:
        emo_data = df_congruent[df_congruent["shownEmotion"] == emo]
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
        print(f"    M(R) = {paired['R'].mean():.3f}, M(NR) = {paired['NR'].mean():.3f}, Diff = {diff.mean():.3f}")
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
choice_labels = {"1": "Happy", "2": "Sad", "3": "Surprised", "4": "Angry", "5": "Disgusted", "6": "Scared"}
df_long["_eid_str"] = df_long["emotionChoiceID"].astype(str)
cong_pairs = {("A", "4"), ("F", "6"), ("HC", "1")}
df_long["is_congruent"] = [
    1 if (se, ec) in cong_pairs else 0
    for se, ec in zip(df_long["shownEmotion"], df_long["_eid_str"])
]
mc_data = df_long[df_long["shownEmotion"] != "N"].copy()
cong_m = mc_data[mc_data["is_congruent"] == 1].groupby("participantID")["rating"].mean()
incong_m = mc_data[mc_data["is_congruent"] == 0].groupby("participantID")["rating"].mean()
mc_paired = pd.DataFrame({"Congruent": cong_m, "Incongruent": incong_m}).dropna()

t_mc, p_mc = stats.ttest_rel(mc_paired["Congruent"], mc_paired["Incongruent"])
diff_mc = mc_paired["Congruent"] - mc_paired["Incongruent"]
d_mc = diff_mc.mean() / diff_mc.std()

print(f"  M(Congruent) = {mc_paired['Congruent'].mean():.3f}, M(Incongruent) = {mc_paired['Incongruent'].mean():.3f}")
print(f"  t({len(mc_paired)-1}) = {t_mc:.3f}, p = {p_mc:.2e}, d = {d_mc:.3f}")
print(f"  -> Participants {'DO' if p_mc < .05 else 'DO NOT'} reliably distinguish emotions.")

# ---- Exploratory: All 24 cells ----
print("\n\n--- Exploratory: R vs NR for all rated emotions x all face types ---")
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
            print(f"  {emo_labels[face_emo]:>8} face, {clabel:>10} rating: D={diff.mean():+.3f}, d={d:+.3f}, p={pv:.3f} {sig}")
print()

results_df.to_csv(os.path.join(OUTPUT_DIR, "hypothesis_test_results.csv"), index=False)
print(f"  Saved: {os.path.join(OUTPUT_DIR, 'hypothesis_test_results.csv')}\n")

# ============================================================================
# 5c. VISUALISATIONS (matplotlib & seaborn)
# ============================================================================

print("=" * 72)
print("STEP 5c: GENERATING PLOTS FOR PRESENTATION")
print("=" * 72)

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set nice styling for presentation
    sns.set_theme(style="whitegrid", font_scale=1.1)
    
    # --- Plot 1: Main Effects Barplot ---
    plt.figure(figsize=(9, 6))
    ax = sns.barplot(
        data=df_congruent, 
        x="shownEmotion", 
        y="rating", 
        hue="facialColoration",
        order=["A", "F", "HC", "N"], 
        palette={"NR": "#A9CCE3", "R": "#F5B7B1"}, # Soft Blue vs Red
        capsize=0.05, 
        errwidth=1.5,
        errorbar=("ci", 95)
    )
    plt.title("Effect of Facial Redness on Emotion Ratings (95% CI)", pad=15, fontweight='bold')
    plt.ylabel("Intensity Rating (1-5)", fontweight='bold')
    plt.xlabel("Shown Emotion Condition", fontweight='bold')
    plt.ylim(1, 5)
    ax.set_xticklabels(["Angry (H1)", "Fearful (H2)", "Happy (H3)", "Neutral (H4)"])
    plt.legend(title="Facial Coloration", labels=["Normal (NR)", "Red (R)"], loc="upper right")
    
    plot1_path = os.path.join(OUTPUT_DIR, "plot_main_effects_bar.png")
    plt.tight_layout()
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"  Saved Plot: {plot1_path}")

    # --- Plot 2: Exploratory Differences Heatmap ---
    heatmap_data = []
    for face_emo in ["A", "F", "HC", "N"]:
        sub = df_long[df_long["shownEmotion"] == face_emo]
        for cid, clabel in choice_labels.items():
            sub2 = sub[sub["_eid_str"] == cid]
            r_m = sub2[sub2["facialColoration"] == "R"].groupby("participantID")["rating"].mean()
            nr_m = sub2[sub2["facialColoration"] == "NR"].groupby("participantID")["rating"].mean()
            both = pd.DataFrame({"R": r_m, "NR": nr_m}).dropna()
            if len(both) > 1:
                diff_mean = (both["R"] - both["NR"]).mean()
                heatmap_data.append({"Face": face_emo, "Rated Emotion": clabel, "Difference": diff_mean})
    
    if heatmap_data:
        df_heat = pd.DataFrame(heatmap_data).pivot(index="Rated Emotion", columns="Face", values="Difference")
        # Format the matrix layout
        df_heat = df_heat[["A", "F", "HC", "N"]]
        df_heat.columns = ["Angry Face", "Fearful Face", "Happy Face", "Neutral Face"]
        df_heat = df_heat.loc[["Angry", "Scared", "Happy", "Sad", "Disgusted", "Surprised"]]
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            df_heat, 
            annot=True, 
            cmap="vlag",   # Red/Blue diverging colormap
            center=0, 
            fmt=".2f", 
            cbar_kws={'label': 'Mean Diff (Red minus Normal)'},
            linewidths=1,
            linecolor='white'
        )
        plt.title("Exploratory Analysis: Impact of Redness Across All Ratings", pad=15, fontweight='bold')
        plt.xlabel("Target Stimulus", fontweight='bold')
        plt.ylabel("Emotion Being Rated", fontweight='bold')
        
        plot2_path = os.path.join(OUTPUT_DIR, "plot_exploratory_heatmap.png")
        plt.tight_layout()
        plt.savefig(plot2_path, dpi=300)
        plt.close()
        print(f"  Saved Plot: {plot2_path}")

# --- Plot 3: Age Distribution (Updated) ---
    if not df_demographics['age'].isna().all():
        plt.figure(figsize=(7, 5))
        # Increased bins from 10 to 20 for more detail
        sns.histplot(data=df_demographics.dropna(subset=['age']), x="age", bins=20, kde=True, color="#5DADE2")
        plt.title("Age Distribution of Participants", pad=15, fontweight='bold')
        plt.xlabel("Age", fontweight='bold')
        plt.ylabel("Number of Participants", fontweight='bold')
        
        plot3_path = os.path.join(OUTPUT_DIR, "plot_demographics_age.png")
        plt.tight_layout()
        plt.savefig(plot3_path, dpi=300)
        plt.close()
        print(f"  Saved Plot: {plot3_path}")

# --- Plot 4: Gender Distribution (Updated to Pie Chart) ---
    if not df_demographics['gender_label'].isna().all():
        plt.figure(figsize=(7, 7))
        gender_counts = df_demographics['gender_label'].value_counts()
        
        # Using matplotlib to create a clean pie chart
        plt.pie(
            gender_counts, 
            labels=gender_counts.index, 
            autopct='%1.1f%%', 
            startangle=140, 
            colors=sns.color_palette("Set2"),
            textprops={'fontweight': 'bold'}
        )
        plt.title("Gender Distribution of Participants", pad=15, fontweight='bold')
        
        plot4_path = os.path.join(OUTPUT_DIR, "plot_demographics_gender.png")
        plt.tight_layout()
        plt.savefig(plot4_path, dpi=300)
        plt.close()
        print(f"  Saved Plot: {plot4_path}")

    # --- Plot 5: Occupation Distribution (Pie Chart: Top Class vs Other) ---
    if 'occupation_1' in df_demographics.columns:
        plt.figure(figsize=(7, 7))
        # Get counts and find the most predominant class
        occ_counts = df_demographics['occupation_1'].value_counts()
        
        if not occ_counts.empty:
            top_label = occ_counts.index[0]
            
            # Group everything that isn't the top class into "Other"
            df_demographics['occ_grouped'] = df_demographics['occupation_1'].apply(
                lambda x: x if x == top_label else "Other"
            )
            grouped_counts = df_demographics['occ_grouped'].value_counts()
            
            plt.pie(grouped_counts, labels=grouped_counts.index, autopct='%1.1f%%', 
                    startangle=140, colors=sns.color_palette("Pastel1"), textprops={'fontweight': 'bold'})
            plt.title(f"Participant Occupations ({top_label} vs. Other)", pad=15, fontweight='bold')
            
            plot5_path = os.path.join(OUTPUT_DIR, "plot_demographics_occupation.png")
            plt.tight_layout()
            plt.savefig(plot5_path, dpi=300)
            plt.close()
            print(f"  Saved Plot: {plot5_path}")

     # --- Plot 6: Comparison of Effect Sizes (Our Study vs. Original Paper) ---
    # Values for Original Study are approximate based on Wolf et al. (2021) results
    comparison_data = [
        {"Hypothesis": "H1: Anger", "Study": "Our Study", "Effect Size (Cohen's d)": 0.132},
        {"Hypothesis": "H1: Anger", "Study": "Wolf et al. (2021)", "Effect Size (Cohen's d)": 0.05},
        {"Hypothesis": "H2: Fear", "Study": "Our Study", "Effect Size (Cohen's d)": -0.100},
        {"Hypothesis": "H2: Fear", "Study": "Wolf et al. (2021)", "Effect Size (Cohen's d)": -0.02},
        {"Hypothesis": "H3: Happy", "Study": "Our Study", "Effect Size (Cohen's d)": 0.123},
        {"Hypothesis": "H3: Happy", "Study": "Wolf et al. (2021)", "Effect Size (Cohen's d)": 0.01},
        {"Hypothesis": "H4: Neutral", "Study": "Our Study", "Effect Size (Cohen's d)": -0.237},
        {"Hypothesis": "H4: Neutral", "Study": "Wolf et al. (2021)", "Effect Size (Cohen's d)": 0.08},
    ]
    df_comp = pd.DataFrame(comparison_data)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_comp, x="Hypothesis", y="Effect Size (Cohen's d)", hue="Study", palette="muted")
    plt.axhline(0, color='black', linewidth=0.8)
    plt.title("Replication Comparison: Cohen's d Effect Sizes", pad=15, fontweight='bold')
    plt.ylim(-0.5, 0.5)
    plt.ylabel("Cohen's d (Effect Size)", fontweight='bold')
    
    plot6_path = os.path.join(OUTPUT_DIR, "plot_replication_comparison.png")
    plt.tight_layout()
    plt.savefig(plot6_path, dpi=300)
    plt.close()
    print(f"  Saved Plot: {plot6_path}")

    # --- Plot 7: Interaction Plot (Line Graph) ---
    plt.figure(figsize=(9, 6))
    sns.pointplot(
        data=df_congruent, 
        x="shownEmotion", 
        y="rating", 
        hue="facialColoration", 
        markers=["o", "s"], 
        linestyles=["-", "--"],
        palette={"NR": "#3498DB", "R": "#E74C3C"},
        order=["A", "F", "HC", "N"],
        capsize=.1
    )
    plt.title("Visual Interaction: Emotion x Coloration", pad=15, fontweight='bold')
    plt.ylabel("Mean Intensity Rating", fontweight='bold')
    plt.xlabel("Face Condition", fontweight='bold')
    plt.xticks(ticks=[0, 1, 2, 3], labels=["Angry", "Fearful", "Happy", "Neutral"])
    
    plot7_path = os.path.join(OUTPUT_DIR, "plot_interaction_lines.png")
    plt.tight_layout()
    plt.savefig(plot7_path, dpi=300)
    plt.close()
    print(f"  Saved Plot: {plot7_path}")

    # --- Plot 8: Manipulation Check (Violin Plot) ---
    if 'mc_data' in locals():
        plt.figure(figsize=(8, 6))
        # Create a violin plot to show the distribution of ratings
        ax = sns.violinplot(
            data=mc_data, 
            x="is_congruent", 
            y="rating", 
            palette={"0": "#E74C3C", "1": "#2ECC71"}, # Red for Incongruent, Green for Congruent
            inner="quartile",
            cut=0 # Prevents the violin from extending past 1 and 5
        )
        plt.title("Manipulation Check: Emotion Recognition", pad=15, fontweight='bold')
        plt.xlabel("Emotion - Face Pairing", fontweight='bold')
        plt.ylabel("Intensity Rating (1-5)", fontweight='bold')
        plt.xticks(ticks=[0, 1], labels=["Incongruent\n(Mismatch)", "Congruent\n(Match)"])
        
        plot8_path = os.path.join(OUTPUT_DIR, "plot_manipulation_check.png")
        plt.tight_layout()
        plt.savefig(plot8_path, dpi=300)
        plt.close()
        print(f"  Saved Plot: {plot8_path}")

    # --- Plot 9: Likert Scale Distribution (100% Stacked Bar Chart) ---
    plt.figure(figsize=(10, 7))
    
    # Calculate the percentage of each rating (1-5) per condition
    likert_counts = (
        df_congruent.groupby(['shownEmotion', 'facialColoration'])['rating']
        .value_counts(normalize=True)
        .unstack(fill_value=0) * 100
    )
    
    # Reorder index for logical grouping
    likert_counts = likert_counts.reindex([
        ('A', 'NR'), ('A', 'R'), 
        ('F', 'NR'), ('F', 'R'), 
        ('HC', 'NR'), ('HC', 'R'), 
        ('N', 'NR'), ('N', 'R')
    ])
    
    # Custom labels for the y-axis
    y_labels = [
        "Angry (Normal)", "Angry (Red)", 
        "Fearful (Normal)", "Fearful (Red)", 
        "Happy (Normal)", "Happy (Red)", 
        "Neutral (Normal)", "Neutral (Red)"
    ]
    
    # Plotting the stacked horizontal bar chart
    ax = likert_counts.plot(
        kind='barh', 
        stacked=True, 
        colormap='coolwarm', # Red/Blue diverging colormap representing 1 to 5
        edgecolor='white',
        figsize=(10, 7),
        width=0.85
    )
    
    plt.title("Distribution of Ordinal Ratings (1-5) per Condition", pad=15, fontweight='bold')
    plt.xlabel("Percentage of Ratings (%)", fontweight='bold')
    plt.ylabel("Face Condition", fontweight='bold')
    ax.set_yticklabels(y_labels)
    
    # Formatting the legend
    plt.legend(title="Chosen Rating", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.invert_yaxis() # Reverses the y-axis so Angry is at the top
    
    plot9_path = os.path.join(OUTPUT_DIR, "plot_likert_distribution.png")
    plt.tight_layout()
    plt.savefig(plot9_path, dpi=300)
    plt.close()
    print(f"  Saved Plot: {plot9_path}")

except ImportError:
    print("  [WARNING] seaborn or matplotlib not installed. Presentation plots were skipped.")
    print("  Install with: pip install seaborn matplotlib\n")


# Clean up temp columns used in Step 5b and 5c
df_long.drop(columns=["is_congruent", "_eid_str", "congruent_choice"], errors="ignore", inplace=True)


# ============================================================================
# 6. STATISTICAL ANALYSIS — BAYESIAN (bambi + arviz)
# ============================================================================

if HAS_BAYESIAN:
    print("\n" + "=" * 72)
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

print("\n" + "=" * 72)
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