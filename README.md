# Machine Learning Laboratory

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains practical laboratory experiments and assignments for the Machine Learning course. Each lab explores core machine learning concepts, exploratory data analysis (EDA), preprocessing techniques, and model development using Python and industry-standard data science libraries.

---

## 📁 Repository Structure

```text
Machine_Learning_LAB/
├── data/                                  # Local dataset storage (optional)
│   └── boston.csv                         # Boston Housing dataset
├── LAB 1/                                 # Laboratory Experiment 1
│   ├── boston-house-price-regression.ipynb # Main Jupyter Notebook
│   └── README.md                          # Detailed Lab 1 Documentation
├── requirements.txt                       # Project dependencies
└── README.md                              # Main repository documentation
```

---

## 📑 Laboratory Index

| Lab | Title | Key Topics | Primary Models / Algorithms |
| :--- | :--- | :--- | :--- |
| **[Lab 1](./LAB%201/README.md)** | **Boston House Price Prediction** | Exploratory Data Analysis (EDA), Feature Scaling, Outlier Inspection | Linear Regression, Polynomial Regression (Degree 2) |

---

## 🔬 Lab 1 Overview: Boston House Price Prediction

- **Notebook:** [`LAB 1/boston-house-price-regression.ipynb`](./LAB%201/boston-house-price-regression.ipynb)
- **Documentation:** [`LAB 1/README.md`](./LAB%201/README.md)
- **Objective:** Predict median house prices (`MEDV`) in the Boston area by analyzing tabular housing characteristics, testing feature scaling methods, and comparing a standard Linear Regression baseline against a Polynomial Regression model.

### Key Highlights
1. **Exploratory Data Analysis (EDA):**
   - Summary statistics and missing value checks.
   - Outlier analysis on crime rate (`CRIM`) using box plots.
   - Multicollinearity and feature relationship inspection via full correlation heatmaps.
   - Scatter analysis between average rooms per dwelling (`RM`) and house price (`MEDV`).
   - Kernel Density Estimation (KDE) for non-retail business acreage (`INDUS`).
2. **Feature Scaling Comparison:**
   - Empirical comparison of three scalers on continuous feature distributions:
     - **StandardScaler:** $\mu = 0, \sigma = 1$
     - **MinMaxScaler:** Rescales to $[0, 1]$ range
     - **RobustScaler:** Median-centered, scaled by Interquartile Range (IQR) to mitigate outlier influence.
3. **Model Training & Evaluation:**
   - 80/20 train/test held-out validation (`random_state=42`).
   - **Baseline:** Multiple Linear Regression on 13 input features.
   - **Polynomial Expansion:** Degree-2 polynomial feature mapping ($13 \rightarrow 104$ features) including quadratic and interaction terms.
   - Evaluation with **MAE**, **MSE**, and **$R^2$ Score**.

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.9 or higher installed on your system.

### 2. Clone the Repository
```bash
git clone https://github.com/anirudhx07/Machine_Learning_LAB.git
cd Machine_Learning_LAB
```

### 3. Set Up Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Windows (CMD):
.\venv\Scripts\activate.bat
# Linux / macOS:
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Obtain the Dataset
Place `boston.csv` into a `data/` directory at the project root:
```text
Machine_Learning_LAB/
└── data/
    └── boston.csv
```
The notebook automatically searches for `data/boston.csv` locally or `/kaggle/input/datasets/fedesoriano/the-boston-houseprice-data/boston.csv` when running in a Kaggle environment.

### 6. Launch Jupyter
```bash
jupyter notebook "LAB 1/boston-house-price-regression.ipynb"
```

---

## 📊 Dataset & Ethical Notice

The Boston Housing dataset is a classic regression benchmark originally published by Harrison & Rubinfeld (1978).

> [!WARNING]
> **Ethical Note:** The Boston Housing dataset contains a variable (`B`) derived from the proportion of Black residents by town. This metric reflects historical segregation and systemic bias. Modern machine learning practices discourage using this dataset for real-world predictive applications. It is utilized here strictly for educational purposes to study regression mechanics and exploratory analysis.

---

## 🛠️ Tech Stack & Dependencies

- **Language:** [Python 3.9+](https://www.python.org/)
- **Data Manipulation:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Data Visualization:** [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/)
- **Machine Learning:** [Scikit-Learn](https://scikit-learn.org/)
- **Interactive Computing:** [Jupyter Notebook](https://jupyter.org/)

---

## 📜 License
This project is for academic and educational purposes under the [MIT License](LICENSE).
