import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def weekly_model_comparison(X, y, use_pca=False, pca_variance=0.95):
    """
    Train LR, RF, XGBoost, SVM, and KNN with 5-fold CV.
    Returns a DataFrame sorted by mean Accuracy (descending).

    Parameters:
        X            : feature matrix (array-like)
        y            : target vector
        use_pca      : if True, prepend PCA to each pipeline
        pca_variance : float — variance to retain when use_pca=True (default 0.95)

    Returns:
        pd.DataFrame with columns: Model, Accuracy_mean, Accuracy_std,
                                   F1_mean, F1_std, TrainTime_mean_s
    """
    scaler = StandardScaler()
    pca    = PCA(n_components=pca_variance, svd_solver='full')

    base_models = {
        'Logistic Regression': LogisticRegression(C=1.0, max_iter=500, random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM (RBF)':           SVC(C=1.0, kernel='rbf', probability=True, random_state=42),
        'KNN (k=5)':           KNeighborsClassifier(n_neighbors=5),
    }

    if HAS_XGB:
        base_models['XGBoost'] = XGBClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=4,
            random_state=42, eval_metric='logloss', verbosity=0
        )

    results = []

    for name, model in base_models.items():
        if use_pca:
            pipeline = Pipeline([('scaler', scaler), ('pca', pca), ('model', model)])
        else:
            pipeline = Pipeline([('scaler', scaler), ('model', model)])

        cv_results = cross_validate(
            pipeline, X, y,
            cv=5,
            scoring=['accuracy', 'f1_macro'],
            return_train_score=False,
            n_jobs=-1
        )

        results.append({
            'Model':           name,
            'Accuracy_mean':   round(cv_results['test_accuracy'].mean(), 4),
            'Accuracy_std':    round(cv_results['test_accuracy'].std(), 4),
            'F1_mean':         round(cv_results['test_f1_macro'].mean(), 4),
            'F1_std':          round(cv_results['test_f1_macro'].std(), 4),
            'TrainTime_mean_s': round(cv_results['fit_time'].mean(), 4),
            'PCA_used':        use_pca,
        })

    df = pd.DataFrame(results).sort_values('Accuracy_mean', ascending=False).reset_index(drop=True)
    return df


# ── Run on Wine Dataset ───────────────────────────────────────────────────────

wine = load_wine()
X, y = wine.data, wine.target

print("Without PCA:")
df_no_pca = weekly_model_comparison(X, y, use_pca=False)
print(df_no_pca.to_string(index=False))

print("\nWith PCA (95% variance):")
df_pca = weekly_model_comparison(X, y, use_pca=True, pca_variance=0.95)
print(df_pca.to_string(index=False))

print(f"\nPCA retained components: ~{df_pca.shape[0]} features from {X.shape[1]} original")
