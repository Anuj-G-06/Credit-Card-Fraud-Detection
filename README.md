# Credit Card Fraud Detection

A machine learning pipeline for detecting fraudulent credit card transactions using a highly imbalanced real-world dataset. Compares multiple classification algorithms and identifies key fraud indicators from PCA-transformed features.

---

## Dataset

The dataset contains transactions made by European cardholders over two days in September 2013.

| Property | Value |
|---|---|
| Total Transactions | 284,807 |
| Fraudulent | 492 (0.172%) |
| Legitimate | 284,315 (99.828%) |
| Features | 30 (V1–V28 from PCA, Time, Amount) |
| Target | Class (0 = Legitimate, 1 = Fraud) |

Features V1 through V28 are principal components obtained from PCA transformation (original features withheld for confidentiality). Only `Time` and `Amount` are untransformed.

---

## Methodology

### 1. Exploratory Data Analysis

- Correlation heatmap across all 30 features
- Distribution analysis of key PCA components (V17, V12, V14, V10) showing strong fraud signal
- Relationship between fraud and transaction amount (higher amounts → more fraud)
- Temporal patterns in fraudulent transactions (less time → more fraud)
- Scatter plots revealing feature dependencies on Amount and Time

### 2. Preprocessing

- **Feature Scaling**: StandardScaler applied to normalize all features to the same range
- **Train-Test Split**: 70/30 and 80/20 splits compared to evaluate impact of training set size
- No resampling applied — class imbalance preserved for realistic evaluation

### 3. Model Training & Comparison

Three classifiers were trained and evaluated:

**Logistic Regression**
- Baseline linear model
- Tested with both 70/30 and 80/20 splits

**Decision Tree Classifier**
- Non-linear decision boundaries
- Evaluated across both split ratios

**Random Forest Classifier**
- Ensemble of 200 decision trees
- Best overall performance

---

## Results

| Model | Accuracy |
|---|---|
| Logistic Regression | 99.93% |
| Decision Tree | 99.92% |
| **Random Forest** | **99.96%** |

**Key Findings:**
- Random Forest Classifier achieved the best results with **99.96% accuracy**
- Features V17, V12, V14, and V10 showed the strongest correlation with fraud
- Increasing training set size (70→80%) did not significantly improve any model
- Higher transaction amounts and lower time values correlated with fraudulent activity

---

## Project Structure

```
├── Fraud Credit Card Transaction Detector.ipynb   # Full analysis notebook
├── Mini-project.pdf                                # Project report
├── creditcard.csv                                  # Dataset
└── README.md
```

---

## Tech Stack

- **Python**: pandas, NumPy, scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Models**: Logistic Regression, Decision Tree, Random Forest
- **Environment**: Jupyter Notebook

---

## How to Run

```bash
git clone https://github.com/Anuj-G-06/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection

pip install pandas numpy matplotlib seaborn scikit-learn

jupyter notebook "Fraud Credit Card Transaction Detector.ipynb"
```

---

## Author

**Anuj Gupta** — [LinkedIn](https://www.linkedin.com/in/anuj-gupta-2k/) · [GitHub](https://github.com/Anuj-G-06)
