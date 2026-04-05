# Part C — Interview Ready

## Q1 — Complete ML Pipeline: 1000 Samples, 200 Features

### Step 1 — Exploratory Data Analysis

Before any modelling, inspect the dataset: check for missing values, class imbalance, feature distributions, and correlations. With 200 features, a correlation heatmap will likely reveal multicollinearity, many features may be redundant. Use `df.describe()` and `df.isnull().sum()` to guide cleaning decisions.

### Step 2 — Preprocessing

*   Impute missing values (median for numerical, mode for categorical)
*   Encode categoricals (OneHotEncoder or OrdinalEncoder)
*   Scale all numerical features with `StandardScaler`, essential for SVM, KNN, LR, and PCA

### Step 3 — Dimensionality Reduction: PCA

**Algorithm chosen: PCA**
**Why:** With 200 features and only 1000 samples, the curse of dimensionality is a real risk. Many models will overfit or be dominated by noise features. PCA reduces features to components that explain 95% of variance, typically 20–50 components for real-world data, while removing multicollinearity. This speeds up training and reduces overfitting for all downstream models.

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X_scaled)
# Typically: 200 features → ~30-50 components
```

### Step 4 — Baseline Model: Logistic Regression

**Algorithm chosen: Logistic Regression**
**Why:** Always start with a simple, interpretable baseline. LR is fast, gives probability scores, and its accuracy on the reduced features establishes a benchmark. If LR achieves 0.88 accuracy, any complex model below that is not worth deploying. LR also reveals whether the problem is linearly separable after PCA.

```python
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1.0, max_iter=500)
# 5-fold CV accuracy → baseline score
```

### Step 5 — Improved Model: Random Forest

**Algorithm chosen: Random Forest**
**Why:** RF handles non-linear patterns that LR misses, is robust to irrelevant features (even after PCA, some components may be noise), and provides feature importances — which components of the PCA space matter most. RF is less sensitive to hyperparameters than boosting models, making it a reliable step up from the baseline.

```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=200, random_state=42)
```

### Step 6 — Hyperparameter Tuning

Use `RandomizedSearchCV` with 5-fold CV to tune the best-performing model. Optimise for the business metric (F1 for imbalanced, AUC for probability calibration, accuracy for balanced classes).

### Step 7 — Final Model Selection & Deployment

Compare Logistic Regression, Random Forest, and optionally XGBoost using the `weekly_model_comparison()` function. Select the model with the best CV score. Deploy with:
*   Saved `StandardScaler` + `PCA` + model as a `sklearn.Pipeline`
*   Monitor prediction drift monthly
*   Retrain when accuracy drops below threshold

---

## Q2 — Coding: `weekly_model_comparison(X, y)`
[Solution](weekly_model_comparison.py)

---

## Q3 — PCA Drops 100 Features to 10 (95% Variance), But Accuracy Falls 0.92 → 0.85

**Three reasons why this can happen:**

### Reason 1 — The 5% Discarded Variance Was Discriminatively Important

PCA selects components that maximise **variance**, not **class separability**. The 5% variance discarded when going from 100 to 10 components may contain features that are low-variance overall but highly correlated with the target label. A feature that barely varies across the whole dataset can still be a perfect separator between two classes. PCA has no way of knowing this, it discards it as "noise" while it was actually signal.

**Fix:** Use `LinearDiscriminantAnalysis (LDA)` instead of PCA — LDA finds components that maximise class separation, not overall variance.

### Reason 2 — The Original Model Was Exploiting Feature Interactions

A tree-based model (Random Forest, XGBoost) can learn interactions between specific pairs of original features (e.g., "Feature 17 AND Feature 83 together predict class 1"). PCA mixes all features into linear combinations, the interaction between the original Feature 17 and Feature 83 is lost or diluted across multiple components. The model can no longer find this interaction in PCA space.

**Fix:** Test PCA only with linear models (LR, SVM). Keep the tree-based model on original features where interaction detection is a strength.

### Reason 3 — 10 Components Is Insufficient — 95% Variance Threshold Was Too Aggressive

The jump from 100 features to 10 is very aggressive. If the dataset requires 30+ components to represent its true structure, then 10 components underfits the feature space even if they explain 95% of variance. The 5% remaining in components 11–30 might represent fine-grained decision boundaries the model needed. Additionally, 95% of *total* variance may not equal 95% of *class-discriminative* variance, the variance not explained by the first 10 components may be disproportionately located in class-relevant directions.

**Fix:** Plot accuracy vs `n_components` (5, 10, 20, 30, 50, 80, 100) and find the elbow, the point where adding more components gives diminishing accuracy returns. Use that `n_components`, not the variance threshold directly.
