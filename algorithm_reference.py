import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# ══════════════════════════════════════════════════════════════════════════════
# WEEK 6 ALGORITHM QUICK REFERENCE
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("WEEK 6 — ALGORITHM QUICK REFERENCE")
print("=" * 65)

# ── 1. Logistic Regression ────────────────────────────────────────────────────
print("\n1. LOGISTIC REGRESSION")
print("   Desc     : Linear classifier using sigmoid to output class probabilities")
print("   Use when : Binary/multi-class, need probabilities, linearly separable data")
print("   Hyperparams: C (regularisation strength), max_iter")

from sklearn.linear_model import LogisticRegression
# Code example
lr = LogisticRegression(C=1.0, max_iter=500, random_state=42)
# lr.fit(X_train, y_train)
# y_pred = lr.predict(X_test)

# ── 2. Decision Tree ──────────────────────────────────────────────────────────
print("\n2. DECISION TREE")
print("   Desc     : Hierarchical if-else rules splitting data by feature thresholds")
print("   Use when : Need interpretable rules, mixed feature types, no scaling needed")
print("   Hyperparams: max_depth, min_samples_split")

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42)

# ── 3. Random Forest ──────────────────────────────────────────────────────────
print("\n3. RANDOM FOREST")
print("   Desc     : Ensemble of decorrelated decision trees via bagging + feature randomness")
print("   Use when : High accuracy needed, tabular data, feature importance required")
print("   Hyperparams: n_estimators, max_depth")

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)

# ── 4. AdaBoost ───────────────────────────────────────────────────────────────
print("\n4. ADABOOST")
print("   Desc     : Sequential ensemble; each model corrects previous errors via sample weights")
print("   Use when : Weak learners to combine, binary classification, low-noise data")
print("   Hyperparams: n_estimators, learning_rate")

from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42)

# ── 5. XGBoost ────────────────────────────────────────────────────────────────
print("\n5. XGBOOST")
print("   Desc     : Gradient boosting with regularisation, parallel tree construction")
print("   Use when : Tabular data competitions, high accuracy, handles missing values")
print("   Hyperparams: n_estimators, learning_rate, max_depth")

try:
    from xgboost import XGBClassifier
    xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=4,
                        random_state=42, eval_metric='logloss', verbosity=0)
    print("   XGBoost available ✅")
except ImportError:
    print("   XGBoost not installed — pip install xgboost")
    xgb = None

# ── 6. LightGBM ───────────────────────────────────────────────────────────────
print("\n6. LIGHTGBM")
print("   Desc     : Gradient boosting with leaf-wise tree growth; faster than XGBoost on large data")
print("   Use when : Large datasets, fast training needed, categorical features present")
print("   Hyperparams: num_leaves, learning_rate, n_estimators")

try:
    from lightgbm import LGBMClassifier
    lgbm = LGBMClassifier(n_estimators=100, learning_rate=0.05,
                          num_leaves=31, random_state=42, verbose=-1)
    print("   LightGBM available ✅")
except ImportError:
    print("   LightGBM not installed — pip install lightgbm")
    lgbm = None

# ── 7. Voting Classifier ──────────────────────────────────────────────────────
print("\n7. VOTING CLASSIFIER")
print("   Desc     : Combines multiple different models by majority vote or averaged probabilities")
print("   Use when : Diverse base models available, want stable generalisation")
print("   Hyperparams: voting ('hard'/'soft'), weights")

from sklearn.ensemble import VotingClassifier
voting = VotingClassifier(
    estimators=[('lr', LogisticRegression(max_iter=500)), ('rf', RandomForestClassifier(random_state=42))],
    voting='soft'
)

# ── 8. Stacking ───────────────────────────────────────────────────────────────
print("\n8. STACKING CLASSIFIER")
print("   Desc     : Uses predictions of base models as features for a meta-learner")
print("   Use when : Diverse base models, want to squeeze maximum accuracy from ensemble")
print("   Hyperparams: estimators (base), final_estimator (meta), cv")

from sklearn.ensemble import StackingClassifier
stacking = StackingClassifier(
    estimators=[('dt', DecisionTreeClassifier(max_depth=4)), ('rf', RandomForestClassifier(random_state=42))],
    final_estimator=LogisticRegression(max_iter=500),
    cv=5
)

# ── 9. SVM ────────────────────────────────────────────────────────────────────
print("\n9. SUPPORT VECTOR MACHINE (SVM)")
print("   Desc     : Finds maximum-margin hyperplane separating classes; kernel trick for non-linearity")
print("   Use when : High-dimensional data, small-medium datasets, clear margin of separation")
print("   Hyperparams: C, kernel ('rbf','linear','poly')")

from sklearn.svm import SVC
svm = SVC(C=1.0, kernel='rbf', probability=True, random_state=42)

# ── 10. KNN ───────────────────────────────────────────────────────────────────
print("\n10. K-NEAREST NEIGHBORS (KNN)")
print("    Desc     : Classifies by majority vote of K closest training points (lazy learner)")
print("    Use when : Small-medium datasets, local pattern matters, non-linear boundaries")
print("    Hyperparams: n_neighbors, metric ('euclidean','manhattan')")

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')

# ── 11. K-Means ───────────────────────────────────────────────────────────────
print("\n11. K-MEANS CLUSTERING")
print("    Desc     : Unsupervised; assigns points to K clusters by nearest centroid iteratively")
print("    Use when : Unlabelled data, known/estimated K, spherical clusters expected")
print("    Hyperparams: n_clusters, init ('k-means++'), n_init")

from sklearn.cluster import KMeans
km = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=42)

# ── 12. DBSCAN ────────────────────────────────────────────────────────────────
print("\n12. DBSCAN")
print("    Desc     : Density-based clustering; finds arbitrarily shaped clusters and labels noise")
print("    Use when : Unknown K, non-spherical clusters, outlier detection needed")
print("    Hyperparams: eps, min_samples")

from sklearn.cluster import DBSCAN
db = DBSCAN(eps=0.5, min_samples=5)

# ── 13. PCA ───────────────────────────────────────────────────────────────────
print("\n13. PCA (Principal Component Analysis)")
print("    Desc     : Unsupervised; projects data onto axes of maximum variance (eigenvectors)")
print("    Use when : High-dimensional data, remove multicollinearity, visualisation (2D/3D)")
print("    Hyperparams: n_components, svd_solver")

from sklearn.decomposition import PCA
pca = PCA(n_components=0.95, svd_solver='full')   # retain 95% variance

# ══════════════════════════════════════════════════════════════════════════════
# ALGORITHM SELECTION FLOWCHART (text-based)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("ALGORITHM SELECTION FLOWCHART")
print("=" * 65)
print("""
START
  │
  ├─ Do you have LABELS? ──────────────────────────────────────────────┐
  │                                                                     │
  NO                                                                   YES
  │                                                                     │
  ├─ Do you know K?                              ├─ Is target CONTINUOUS or DISCRETE?
  │    YES → K-Means                             │
  │    NO  → DBSCAN                              CONTINUOUS → Regression task (not this week)
  │                                              DISCRETE   → Classification
  └─ Want to visualise/compress?                             │
       YES → PCA                                             ├─ Need INTERPRETABILITY?
                                                             │    YES → Logistic Regression or Decision Tree
                                                             │
                                                             ├─ High ACCURACY priority?
                                                             │    YES → Random Forest / XGBoost / LightGBM
                                                             │    Tabular + speed → LightGBM
                                                             │    Tabular + accuracy → XGBoost or RF
                                                             │
                                                             ├─ High-dimensional / small dataset?
                                                             │    YES → SVM (kernel='rbf')
                                                             │
                                                             ├─ Local patterns / instance-based?
                                                             │    YES → KNN
                                                             │
                                                             └─ Multiple models available?
                                                                  Diverse → Voting (soft)
                                                                  Best accuracy → Stacking
""")

# ══════════════════════════════════════════════════════════════════════════════
# VERIFY CODE SNIPPETS ON WINE DATASET (3 algorithms)
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("VERIFICATION — Wine Dataset")
print("=" * 65)

wine = load_wine()
X, y = wine.data, wine.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_train)
X_te_sc = scaler.transform(X_test)

results = {}

# 1. Logistic Regression
lr.fit(X_tr_sc, y_train)
results['Logistic Regression'] = accuracy_score(y_test, lr.predict(X_te_sc))

# 2. Random Forest
rf.fit(X_train, y_train)
results['Random Forest'] = accuracy_score(y_test, rf.predict(X_test))

# 3. SVM
svm.fit(X_tr_sc, y_train)
results['SVM (RBF)'] = accuracy_score(y_test, svm.predict(X_te_sc))

print(f"\n  {'Model':<22} {'Accuracy':>10}")
print("  " + "─" * 34)
for name, acc in results.items():
    print(f"  {name:<22} {acc*100:>9.2f}%")
print("\n  ✅ All code snippets verified on Wine dataset")
