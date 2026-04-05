# Part D — AI-Augmented Task: Week 6 Study Guide

## Prompt Used

*"Generate a comprehensive Week 6 study guide covering all 12 algorithms and techniques: Logistic Regression, Decision Tree, Random Forest, AdaBoost, XGBoost, LightGBM, Voting, Stacking, SVM, KNN, K-Means, DBSCAN, and PCA. For each: key concept, when to use, common interview question, and a code pattern."*

---

## AI Output — Week 6 Study Guide

### 1. Logistic Regression
**Key concept:** Applies sigmoid to a linear combination of features to output class probabilities. Decision boundary is linear.
**When to use:** Binary/multi-class classification, need interpretable coefficients, data is roughly linearly separable.
**Interview Q:** *"Why is it called 'regression' if it's a classifier?"* — Because it models the log-odds (a regression) of the probability, then thresholds it.
**Code pattern:** `LogisticRegression(C=1.0, max_iter=500).fit(X_scaled, y)`

### 2. Decision Tree
**Key concept:** Recursively splits data on the feature/threshold that maximises information gain (or minimises Gini impurity). Produces human-readable rules.
**When to use:** Need explainability, regulatory compliance, mixed feature types, no scaling needed.
**Interview Q:** *"Why do decision trees overfit?"* — Fully grown trees memorise training data (high variance). Fix: `max_depth`, `min_samples_leaf`.
**Code pattern:** `DecisionTreeClassifier(max_depth=5, criterion='gini').fit(X, y)`

### 3. Random Forest
**Key concept:** Bagging + feature randomness. Trains B trees on bootstrap samples, each split considers only `sqrt(p)` features. Averages predictions to reduce variance.
**When to use:** High accuracy on tabular data, feature importance needed, robust to outliers.
**Interview Q:** *"How does RF reduce overfitting compared to a single tree?"* — Decorrelated trees via feature randomness; averaging cancels individual errors.
**Code pattern:** `RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)`

### 4. AdaBoost
**Key concept:** Sequential ensemble. Each weak learner focuses on points misclassified by the previous one by upweighting their sample weights.
**When to use:** Boosting weak learners (shallow stumps), binary classification, low-noise data.
**Interview Q:** *"How does AdaBoost differ from Gradient Boosting?"* — AdaBoost adjusts sample weights; GBM fits residuals directly.
**Code pattern:** `AdaBoostClassifier(n_estimators=100, learning_rate=0.5).fit(X, y)`

### 5. XGBoost
**Key concept:** Gradient boosting with L1/L2 regularisation, parallel tree building, and built-in handling of missing values.
**When to use:** Tabular data competitions, high accuracy, when you can tune hyperparameters.
**Interview Q:** *"What regularisation does XGBoost add over GBM?"* — L1 (alpha) and L2 (lambda) on leaf weights + tree complexity penalty.
**Code pattern:** `XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=4).fit(X, y)`

### 6. LightGBM
**Key concept:** Gradient boosting with leaf-wise (best-first) growth instead of level-wise. Faster on large datasets, handles categoricals natively.
**When to use:** Large datasets (>100K rows), fast training needed, high-cardinality categoricals.
**Interview Q:** *"What is the key difference between LightGBM and XGBoost tree growth?"* — LightGBM grows leaf-wise (splits the leaf with max gain); XGBoost grows level-wise.
**Code pattern:** `LGBMClassifier(num_leaves=31, learning_rate=0.05).fit(X, y)`

### 7. Voting Classifier
**Key concept:** Combines diverse models by majority vote (hard) or averaged probabilities (soft). Works best when base models make different errors.
**When to use:** Multiple trained models available, want stable generalisation without complex stacking.
**Interview Q:** *"When should you use soft vs hard voting?"* — Soft when models output calibrated probabilities; hard when only class labels are available.
**Code pattern:** `VotingClassifier(estimators=[('lr',lr),('rf',rf)], voting='soft').fit(X, y)`

### 8. Stacking
**Key concept:** Base models' out-of-fold predictions become features for a meta-learner. Learns how to best combine base model outputs.
**When to use:** Diverse base models available, maximum accuracy is the goal, enough data for meaningful OOF predictions.
**Interview Q:** *"Why do we use cross-validation for generating stacking features?"* — To prevent data leakage: meta-learner must not see predictions on data used to train base models.
**Code pattern:** `StackingClassifier(estimators=[...], final_estimator=LogisticRegression()).fit(X, y)`

### 9. SVM
**Key concept:** Finds the maximum-margin hyperplane separating classes. Kernel trick maps data to high-dimensional space for non-linear boundaries.
**When to use:** High-dimensional data (text, images), small-medium datasets, clear margin expected.
**Interview Q:** *"What are support vectors?"* — The training points closest to the decision boundary that determine the margin.
**Code pattern:** `SVC(C=1.0, kernel='rbf').fit(X_scaled, y)`

### 10. KNN
**Key concept:** Lazy learner — stores all training data. At inference, classifies by majority vote of K nearest neighbours by distance.
**When to use:** Small-medium datasets, non-linear boundaries, local patterns matter, quick prototyping.
**Interview Q:** *"What is the curse of dimensionality and how does it affect KNN?"* — In high dimensions, all points become equidistant, making "nearest" meaningless.
**Code pattern:** `KNeighborsClassifier(n_neighbors=5).fit(X_scaled, y)`

### 11. K-Means
**Key concept:** Unsupervised. Iteratively assigns points to nearest centroid and recomputes centroids until convergence. Minimises WCSS.
**When to use:** Customer segmentation, document clustering, K is known or can be estimated with elbow method.
**Interview Q:** *"Why must you scale before K-Means?"* — Distance-based: unscaled high-magnitude features dominate distance calculations.
**Code pattern:** `KMeans(n_clusters=3, init='k-means++', n_init=10).fit(X_scaled)`

### 12. DBSCAN
**Key concept:** Density-based. Core points have ≥ min_samples neighbours within eps. Connected core points form clusters. Non-core, non-reachable points are noise.
**When to use:** Unknown K, non-spherical clusters, outlier detection, spatial data.
**Interview Q:** *"How do you choose eps for DBSCAN?"* — Plot k-distance graph (sorted distances to kth neighbour); eps is at the "knee" of the curve.
**Code pattern:** `DBSCAN(eps=0.5, min_samples=5).fit(X_scaled)`

### 13. PCA
**Key concept:** Finds orthogonal axes (principal components) of maximum variance via eigendecomposition of the covariance matrix. Projects data onto top K components.
**When to use:** Dimensionality reduction before modelling, remove multicollinearity, 2D/3D visualisation of high-dimensional data.
**Interview Q:** *"Does PCA remove noise or signal?"* — Both: it removes low-variance directions which are often noise, but can accidentally discard discriminative low-variance signal.
**Code pattern:** `PCA(n_components=0.95).fit_transform(X_scaled)`

---

## Evaluation

### Accuracy Check

*   ✅ All 13 concepts correctly defined — no factual errors found.
*   ✅ "When to use" guidance is accurate and practical for each algorithm.
*   ✅ All interview questions are commonly asked and the answers are correct.
*   ✅ All code patterns are syntactically correct and representative.
*   ✅ The LightGBM leaf-wise vs XGBoost level-wise distinction is correctly stated.
*   ✅ The stacking cross-validation explanation correctly identifies data leakage risk.

### Missing Concepts Identified

*   ⚠️ **Silhouette score** not mentioned under K-Means — important evaluation metric for clustering.
*   ⚠️ **ARI / NMI** not mentioned — key metrics for comparing clustering to ground truth.
*   ⚠️ **Elbow method** not mentioned under K-Means — standard way to choose K.
*   ⚠️ **Explained variance ratio** not mentioned under PCA — critical for choosing n_components.
*   ⚠️ **OOB error** not mentioned under Random Forest — a key advantage over other ensemble methods.
*   ⚠️ **class_weight='balanced'** not mentioned for LR/RF/SVM — critical for imbalanced datasets.

All missing concepts have been covered in the respective assignment solutions above.
