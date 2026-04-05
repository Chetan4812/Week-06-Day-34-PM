# Week-06-Day-34-PM

**Take-Home Assignment: PCA, Clustering & Week 6 Comprehensive Review**
**Day 34 | PM Session | Week 6 — Machine Learning & AI**

---

# Part A — Concept Application (40%)

*   Quick reference for all 13 algorithms/techniques: 1-line description, when to use, code example, 2 key hyperparameters
*   Algorithm selection flowchart (text-based)
*   Verify 3 algorithms (LR, RF, SVM) on the Wine dataset <br>
[Solution](algorithm_reference.py)

### Algorithm Selection Flowchart (Summary)

```
Has labels? → NO  → Know K? → YES: K-Means | NO: DBSCAN | Visualise/compress: PCA
            → YES → Need interpretability?  → LR or DT
                  → High accuracy?          → RF / XGBoost / LightGBM
                  → High-dimensional?       → SVM
                  → Local patterns?         → KNN
                  → Multiple models?        → Voting (diverse) / Stacking (max accuracy)
```

---

## Part B — Stretch Problem (30%)

*   Load image, flatten colour channels
*   Apply PCA with n_components = 5, 20, 50, 100
*   Reconstruct and display compressed images
*   Compute compression ratio and MSE for each <br>
[Solution](pca_image_compression.py)

---

## Part C — Interview Ready (20%)

**Q1 — Complete ML pipeline: 1000 samples, 200 features. 3 algorithms, why each.**

**Q2 (Coding) — `weekly_model_comparison(X, y)` with 5-fold CV + PCA option**

**Q3 — PCA reduces 100→10 features (95% var), accuracy drops 0.92→0.85. Why? (3 reasons)** <br>
[Written Answers](interview_questions.md) | [Q2 Code](weekly_model_comparison.py)

---

## Part D — AI-Augmented Task (10%)

**Prompt:** *"Generate a comprehensive Week 6 study guide covering all 13 algorithms with key concepts, interview questions, and code patterns."* <br>
[AI Output & Evaluation](AI_output.md)

---

## File Index

| File | Purpose |
| :--- | :--- |
| `algorithm_reference.py` | Part A — All 13 algorithms: description, hyperparams, use case, flowchart, Wine verification |
| `pca_image_compression.py` | Part B — PCA compression at n=5,20,50,100 with MSE and compression ratio |
| `weekly_model_comparison.py` | Part C Q2 — `weekly_model_comparison()` with PCA option, 5-fold CV |
| `interview_questions.md` | Part C Q1 + Q3 — Full pipeline answer, 3 reasons for PCA accuracy drop |
| `AI_output.md` | Part D — AI study guide for all 13 algorithms + accuracy check + missing concepts |
