**Role:** Act as an expert data scientist and statistician.
**Task:** Write a complete, well-commented R script (using `tidyverse`, `lme4`, `emmeans`, and `brms`) to load, clean, reshape, and analyze the dataset according to the rules and pre-registered OSF methodology below.

**IMPORTANT CONTEXT:** This dataset is an ongoing Qualtrics export. The code must be robust, reproducible, and dynamically written so that it can run without errors on both the current incomplete dataset and the final complete dataset.

### 1. Data Import & Qualtrics Cleaning Rules (Execute in exact order)
* **Step 1: Qualtrics Row Architecture:** Import the CSV normally so that the actual column names (Row 1) are preserved as the dataframe headers. Immediately after loading, slice out and discard the first two rows of data (which contain the Qualtrics question text and internal JSON Import IDs).
* **Step 2: Type Conversion (CRITICAL):** Because the initial skipped rows contained text, R will default to importing numeric columns as characters. Explicitly mutate `Progress`, `Finished`, `consent_form`, and all the survey rating columns into numeric data types before proceeding.
* **Step 3: Date Filter:** Filter the dataset to keep only results recorded on or after **March 19th, 2026** (using `StartDate` or `RecordedDate`).
* **Step 4: Standardize Participant IDs:** Clean the `participantID` column by correcting known typos: Replace `"ZP009"` with `"P009"`, and `"Po27"` with `"P027"`.
* **Step 5: Consent Filter:** Qualtrics exported this column as numeric recode values. Keep only rows where `consent_form == 1` (which corresponds to given consent). Drop any rows with `2` or `NA`.
* **Step 6: Completion Filter:** Keep only rows where `Finished == 1` (or True) or `Progress == 100`.
* **Step 7: Deduplication (Multiple Attempts):** Group the data by `participantID`, arrange by date (latest first), and retain only the most recent valid attempt per participant to resolve mid-survey dropouts or restarts.
* **Step 8: Exclusion Criteria:** Drop participant `"P046"` entirely from the dataset (due to failing colorblindness criteria).
* **Step 9: Column Pruning:** Drop all `ishihara` validation columns and standard Qualtrics metadata columns (Start Date, End Date, IP Address, Duration, UserLanguage, etc.). Keep only the participant ID and the target rating columns.

### 2. Experimental Design & Missing Data Logic
* **Trial Count:** A single participant is randomly assigned a subset of exactly 16 images (2 targets × 4 shown emotions × 2 facial colorations). 
* **Matrix Setup:** For each of the 16 images, the participant rates 6 emotion choices on a scale from **1 to 5**. This results in 96 total questions answered per participant.
* **Missing Data (NAs):** The survey uses "Force Response". If a cell is entirely blank/NA, it strictly means the image was not displayed to them due to randomization. Drop `NA` values dynamically during the reshaping process; do not impute them.

### 3. Data Dictionary & Variable Labeling Scheme
Every matrix question is labeled using the following exact format: 
`WF-[targetNumber]-[facialColoration]-[shownEmotion]_[emotionChoiceID]`

* **targetNumber:** Corresponds to the target ID in the Chicago Face Database (e.g., `001`, `012`). 
* **facialColoration:** `NR` (non-red, normal coloration) or `R` (red, modified face).
* **shownEmotion:** The actual expression on the face shown in the image: `N` (neutral), `HC` (happy closed-mouth), `A` (angry), or `F` (fearful).
* **emotionChoiceID:** The specific emotion the participant is currently rating from 1 to 5: `_1` = Happy, `_2` = Sad, `_3` = Surprised, `_4` = Angry, `_5` = Disgusted, `_6` = Scared.

### 4. Data Reshaping & Target Gender Mapping
1. **Pivot to Long Format:** Melt the wide dataset into a long format. Extract `targetNumber`, `facialColoration`, `shownEmotion`, and rated `emotionChoiceID` from the column headers into their own distinct columns using regex or tidyr separation functions.
2. **Target Gender:** The models require `target_gender` (Male/Female) as a fixed effect. Since this is not in the column names, create a placeholder dictionary/join-table in the code that maps `targetNumber` to `target_gender` and merge it into the long dataframe.
3. **Congruent Rating Filter:** For the main analysis, we ONLY use the rating corresponding to the emotion depicted in the image (or our specific hypothesized emotion). Create a filtered dataframe (`df_congruent`) using this logic:
   * If `shownEmotion == "A"`, keep the `_4` (Angry) rating.
   * If `shownEmotion == "F"`, keep the `_6` (Scared) rating.
   * If `shownEmotion == "HC"`, keep the `_1` (Happy) rating.
   * If `shownEmotion == "N"`, keep the `_4` (Angry) rating (To test H4: anger ratings on neutral faces).

### 5. Statistical Analysis (Pre-Registered Methodology)
Write the code to execute the following mixed-effects models on the `df_congruent` data to test our four specific hypotheses.

**Dependent Variable:** Congruent emotion intensity rating (1-5).
**Fixed Effects:** `facialColoration` (NR vs R) * `shownEmotion` (N, A, HC, F) + `target_gender`.
**Random Effects:** Intercepts for `participantID` and intercepts for `targetNumber` (if convergence allows).

**1. Frequentist Pipeline (`lme4`):** * Run a linear mixed model (`lmer`). 
* Use `emmeans` to conduct planned contrasts between Normal (NR) and Red (R) coloration within each emotion condition. 
  * **H1:** Test `R` vs `NR` for `A` faces (Expectation: Red > Normal, significant).
  * **H2:** Test `R` vs `NR` for `F` faces (Expectation: No significant difference).
  * **H3:** Test `R` vs `NR` for `HC` faces (Expectation: Red > Normal, significant).
  * **H4:** Test `R` vs `NR` for `N` faces (Expectation: Red > Normal, significant).

**2. Bayesian Pipeline (`brms`):** * Run a cumulative (ordered) probit model (`family = cumulative("probit")`) due to the ordinal nature of the 1-5 scale. 
* **CRITICAL:** Include multi-core processing arguments (e.g., `cores = parallel::detectCores()`) to speed up compilation and sampling.
* Use weakly informative priors (e.g., `normal(0,1)`). 
* Extract the estimated marginal means (or conditional effects) to test the same four specific contrasts (H1, H2, H3, H4) outlined in the frequentist pipeline.
* Report 95% credibility intervals for these contrasts, and include code for posterior predictive checks (`pp_check`).