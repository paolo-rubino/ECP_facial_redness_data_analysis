# Facial Redness & Emotion Perception Analysis

This repository contains a conceptual replication of the study by **Wolf, Leder, Röseler & Schütz (2021)**: *"Does facial redness really affect emotion perception?"* published in *Cognition and Emotion*.

## 📌 Project Overview
The project analyzes how facial coloration (Red vs. Normal) influences the perception and intensity ratings of various emotions (Angry, Fearful, Happy, and Neutral). 

The analysis pipeline includes:
* Data cleaning and preprocessing of Qualtrics exports.
* Frequentist analysis using Mixed-Effects Models (`statsmodels`) and paired t-tests (`scipy`).
* Bayesian analysis using Cumulative Probit Models (`bambi` & `arviz`).

## 🛠️ Setup Instructions

### 1. Environment (Recommended: Conda)
Because this project uses Bayesian libraries (`Bambi`/`PyMC`) that require C++ compilers, using **Conda** is highly recommended to manage binary dependencies and avoid compiler errors.

```bash
# Create the environment with Python 3.11
conda create -n facial_redness python=3.11

# Activate the environment
conda activate facial_redness

# Install dependencies
# Using conda-forge ensures all C-compilers are handled correctly
conda install -c conda-forge m2w64-toolchain libpython
pip install pandas numpy statsmodels bambi arviz matplotlib seaborn