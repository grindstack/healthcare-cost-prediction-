
# SOFTEC’26 ML Competition: Healthcare Cost Prediction

## Overview
This repository contains the solution for the SOFTEC’26 Machine Learning Hackathon, focused on predicting high-cost patients for targeted healthcare interventions. The challenge was to identify which patients would exceed $30,000 in healthcare costs in the following year using only current-year data.

## Problem Statement
Healthcare providers need to proactively identify high-risk, high-cost patients to optimize resource allocation and improve outcomes. The task is a binary classification problem: predict whether a patient’s annual healthcare cost will exceed $30,000 next year, based on historical and current-year data.

## Dataset Description
- **Sources:** 5 real-world healthcare datasets
	- Demographics & utilization
	- Procedure codes (CPT)
	- Diagnosis codes (ICD)
	- Hospital stay codes (DRG)
	- Date of birth
- **Features:** Hundreds of raw and engineered features
- **Structure:** Separate train/test splits, multiple CSVs per split

## Approach & Methodology
1. **Data Merging & Preprocessing**
	- Merged all datasets on patient keys
	- Filtered to December/year-end records for consistency
	- Removed leakage columns (e.g., NEXT_YEAR_COST)
2. **Feature Engineering**
	- Created 25+ features: cost ratios, HCC trends, utilization metrics, chronic burden, emergency/admission ratios
	- Aggregated CPT/ICD/DRG codes: count and diversity per patient
	- Built interaction and log-transformed features to capture non-linear effects
3. **Handling Data Quality**
	- Median imputation for missing values
	- Label encoding for categorical variables
4. **Class Imbalance Handling**
	- Used `scale_pos_weight` in models
	- Stratified cross-validation
5. **Modeling & Validation**
	- Ensemble of LightGBM, XGBoost, CatBoost (weighted averaging)
	- 5-fold stratified cross-validation
	- Threshold tuning via grid search to maximize F1 score

## Model Architecture
- **LightGBM**: Fast, robust to outliers, interpretable feature importance
- **XGBoost**: Strong regularization, handles complex patterns
- **CatBoost**: Native categorical support, symmetric trees
- **Ensemble**: Weighted average of model outputs (weights optimized on validation AUC)

## Evaluation Metrics
- **Primary:** F1 Score (threshold-optimized)
- **Secondary:** AUC-ROC, Precision, Recall
- **Leaderboard:** Public and Private scores (binary classification)

## Results
- **Public Leaderboard:** Peaked at Rank 2, final position Rank 22
- **Private Leaderboard:** Final Rank 7, Score: `0.46840` (+15 position jump)
- **Shortlisted:** Advanced to evaluation round

## Key Learnings
- Robust feature engineering and domain knowledge are critical for healthcare ML tasks
- Ensemble models with proper weighting outperform single models
- Threshold tuning (beyond default 0.5) can significantly improve F1 and practical utility
- Handling class imbalance and data leakage is essential for real-world deployment
- Validation on OOF predictions is vital to avoid overfitting

## Tech Stack
- Python 3.x
- pandas, numpy
- scikit-learn
- lightgbm, xgboost, catboost
- matplotlib, seaborn

## Future Improvements
- Incorporate advanced imputation (e.g., KNN, MICE) for missing values
- Explore deep learning models for richer feature extraction
- Automated feature selection and hyperparameter optimization
- Model explainability (e.g., SHAP values) for clinical adoption
- Deploy as a real-time risk scoring API

---


**Sponsored by [Soliton Technologies](https://www.solitontechnologies.com/)**
