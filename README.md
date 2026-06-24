# Customer Segmentation with RFM + K-Means

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-KMeans-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter&logoColor=white)
![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?style=flat&logo=kaggle&logoColor=white)

Unsupervised machine learning project that segments e-commerce customers using RFM (Recency, Frequency, Monetary) analysis and K-Means clustering.

---

## Overview

This project builds a customer segmentation model on the [Online Retail Dataset](https://www.kaggle.com/datasets/vijayuv/onlineretail) from UCI, which contains ~500k real transactions from a UK-based e-commerce store.

The goal is to identify meaningful customer segments that can drive targeted marketing strategies.

---

## Customer Segments Found

| Cluster | Name | Recency | Frequency | Monetary | Strategy |
|---------|------|---------|-----------|----------|----------|
| 0 | Moderates | ~32 days | ~3 orders | ~$829 | Reactivation campaigns |
| 1 | Inactive | ~243 days | ~1 order | ~$310 | Aggressive win-back or deprioritize |
| 2 | VIP | ~2 days | ~63 orders | ~$117k | Loyalty programs, exclusive access |
| 3 | Liquid | ~5 days | ~19 orders | ~$8k | Upselling toward VIP |

---

## Project Structure

```
├── data/
│   ├── OnlineRetail.csv                          # Raw dataset
│   └── out/
│       └── data_processed_w_clusters.parquet     # Output with cluster labels
├── script.ipynb          # Main notebook
├── requirements.txt
└── README.md
```

---

## Pipeline

```
Raw Data → Cleaning → RFM Table → Scaling → Elbow Method →
Silhouette Score → K-Means → PCA Visualization → Export
```

### 1. Data Cleaning

- Removed rows with missing `CustomerID` (~135k rows, 25% of dataset)
- Removed cancelled transactions (`InvoiceNo` starting with `C`)
- Removed rows with `UnitPrice = 0`
- Fixed data types: `InvoiceDate` → datetime, `CustomerID` → int

### 2. RFM Feature Engineering

- **Recency** — days since last purchase (reference date: day after last transaction)
- **Frequency** — number of unique invoices per customer
- **Monetary** — total spend per customer (`Quantity × UnitPrice`)

### 3. Preprocessing

- Applied `StandardScaler` to normalize all three RFM variables
- Prevents Monetary from dominating distance calculations

### 4. Optimal K Selection

- **Elbow Method** — plotted inertia for K=1 to K=10
- **Silhouette Score** — confirmed K=4 and K=5 as top candidates
- Final choice: **K=4** for business interpretability

### 5. Modeling & Visualization

- Trained final K-Means model with `n_clusters=4`, `random_state=42`
- Reduced dimensions with PCA (2 components — explains 85.7% of variance)
- Scatter plot of clusters in 2D PCA space

---

## Key Results

- **4,338 unique customers** analyzed
- **85.7% variance** preserved after PCA reduction
- **Silhouette Score: 0.618** at K=4 — well-separated clusters
- Actionable segments ready for marketing team consumption

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Cyac221/clustering-kmeans.git
cd clustering-kmeans
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Mac/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download `OnlineRetail.csv` from [Kaggle](https://www.kaggle.com/datasets/vijayuv/onlineretail) and place it in the `data/` folder.

### 5. Run the notebook

```bash
jupyter notebook script.ipynb
```

Run all cells from top to bottom. Output parquet file will be saved to `data/out/`.

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| `pandas` | Data cleaning and RFM feature engineering |
| `numpy` | Numerical operations |
| `scikit-learn` | KMeans, StandardScaler, PCA, silhouette_score |
| `matplotlib` | Cluster visualization |
| `pyarrow` | Parquet export |
| Jupyter Notebook | Interactive analysis environment |

---

## Data Source

[Online Retail Dataset](https://www.kaggle.com/datasets/vijayuv/onlineretail) — UCI Machine Learning Repository / Kaggle
