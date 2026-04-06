# Credit Card Fraud Detection

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-006600)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-9558B2)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-explainability-blueviolet)
![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end fraud detection pipeline on a highly imbalanced dataset (~0.17% fraud), covering deep EDA, systematic resampling experiments, classical ML with hyperparameter tuning, unsupervised anomaly detection, SHAP-based model explainability, and production-oriented threshold analysis.

## Dataset

[Credit Card Fraud Detection — Kaggle (ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

- **284,807** transactions over two days by European cardholders
- **492** fraudulent transactions (0.173%)
- Features V1–V28 are PCA-transformed (originals are confidential); `Time` and `Amount` are untransformed

## Results at a Glance

| Metric | Best Model (LightGBM + Random Oversample) |
|---|---|
| **AUPRC** | 0.9151 |
| **AUROC** | 0.9847 |
| **F1** | 0.9005 |
| **F2** | 0.8866 |
| **Precision** | 0.9247 |
| **Recall** | 0.8776 |

## Project Structure
```
Credit-Card-Fraud-Detection/
├── credit_card_fraud_detection.ipynb   # Full pipeline (single notebook)
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md
```

## Notebook Roadmap

### Part 1 — Exploratory Data Analysis
Class distribution, Amount/Time fraud-vs-legit comparisons, top-correlated feature box plots, compact correlation heatmap, PCA & t-SNE dimensionality reduction visualizations.

### Part 2 — Preprocessing & Sampling Strategy Experiments
StandardScaler on Time/Amount → stratified 80/20 split → five resampling strategies applied on training set only:

| Strategy | Type |
|---|---|
| No Resampling | Baseline |
| Random Oversample (ROS) | Naive oversampling |
| SMOTE | Synthetic oversampling |
| ADASYN | Adaptive synthetic |
| SMOTE + Tomek Links | Combined over + undersampling |

Includes distribution shift checks and a discussion on data leakage prevention.

### Part 3 — Classical ML Modeling
Four models × five sampling strategies = 20 combinations, each tuned via `RandomizedSearchCV` with `StratifiedKFold(n_splits=5)`:

- Logistic Regression (baseline)
- Random Forest
- XGBoost
- LightGBM

Results presented as a Model × Sampling heatmap with AUPRC and AUROC.

### Part 4 — Anomaly Detection Perspective
Reframes fraud detection as anomaly detection:

- **Isolation Forest** — fully unsupervised (full-data) and semi-supervised (non-fraud only) variants
- **Autoencoder (PyTorch)** — trained exclusively on normal transactions; uses reconstruction error as anomaly score

Side-by-side comparison with supervised methods on AUROC and AUPRC.

### Part 5 — Model Explainability (SHAP)
Global and local explanations for the best model:

- Summary plot (beeswarm) — feature importance ranking
- Bar plot — mean |SHAP| values
- Dependence plots — top 4 features
- Waterfall plots — individual transaction explanations (true positive, false negative, false positive)

### Part 6 — Comprehensive Evaluation & Conclusion
- Unified metrics table across all methods (supervised + unsupervised)
- Overlaid ROC and Precision-Recall curves
- Best model confusion matrix
- Threshold sweep (0.01–0.99) with F1 and F2 optimization
- Business scenario comparison table (high recall vs. balanced vs. high precision)

**Key insight:** Threshold tuning is as important as model selection — the optimal F1 threshold (0.96) differs significantly from the default (0.5), and the right choice depends on business cost trade-offs.

## Reproducing

1. **Clone the repo**
```bash
   git clone https://github.com/ry2406/Credit-Card-Fraud-Detection.git
   cd Credit-Card-Fraud-Detection
```

2. **Install dependencies**
```bash
   pip install -r requirements.txt
```

3. **Download the dataset**

   Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the project root directory.

4. **Run the notebook**
```bash
   jupyter notebook credit_card_fraud_detection.ipynb
```

> **Note:** The full training run (20 model × sampling combinations with `RandomizedSearchCV`) is CPU-intensive. On a laptop, consider reducing `n_iter` or the number of sampling strategies. A multi-core server (e.g., 48-core HPC node) completes the full run in ~20–30 minutes.

## Tech Stack

- **Core:** Python, NumPy, Pandas, Matplotlib, Seaborn
- **ML:** scikit-learn, XGBoost, LightGBM
- **Deep Learning:** PyTorch (Autoencoder)
- **Resampling:** imbalanced-learn (SMOTE, ADASYN, SMOTE+Tomek)
- **Explainability:** SHAP

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Author

**Ruide Yin** — [GitHub](https://github.com/ry2406)
