# Mushroom Classification 🍄

Kaggle ML competition — classify mushrooms as edible or poisonous using 
categorical biological features.

## Problem
Binary classification on UCI mushroom dataset. All 21 features are categorical 
(cap shape, odor, gill color, ring type, etc.). No numerical features except 
`ring-number` and `number_of_bruises`.

## Dataset
UCI Mushroom Dataset — 21 categorical features describing physical 
characteristics of mushrooms. Target: `class` (edible / poisonous).

## Approach
1. **EDA** — class distribution, categorical feature vs target cross-tabs, 
   value frequency analysis
2. **Missing Value Handling** — `odor` column imputed with `'none'` 
   (no smell ≠ foul smell — mode imputation would wrongly bias toward poisonous)
3. **Outlier Analysis** — IQR on numerical cols; retained outliers since 
   tree-based models are robust to them
4. **Encoding** — LabelEncoder fit on combined train+test to handle unseen 
   categories; ID columns dropped
5. **Scaling** — StandardScaler on numerical columns for KNN and Logistic Regression
6. **Model Training** — 9 models: Logistic Regression, Decision Tree, Random Forest, 
   Gradient Boosting, AdaBoost, Extra Trees, KNN, Naive Bayes, XGBoost
7. **Hyperparameter Tuning** — GridSearchCV on Decision Tree, Random Forest, XGBoost
8. **Final Model** — Voting Ensemble (RF + Extra Trees + XGBoost, hard voting)

## Result
| Metric | Score |
|--------|-------|
| Best Model | Voting Ensemble (RF + ET + XGBoost) |
| Target | F1 ≥ 0.98 |

## Key Learnings
- Domain-aware imputation matters — `odor = none` is biologically meaningful, 
  not just a fill value
- Fitting LabelEncoder on combined train+test prevents unseen category errors at inference
- Voting ensemble more robust than any single model for safety-critical classification

## Tech Stack
`Python` `Pandas` `NumPy` `Scikit-learn` `XGBoost` `Matplotlib` `Seaborn`
