# Breast Cancer Classification Models

This repository contains the implementation and analysis of machine learning models for breast cancer diagnosis using the **Breast Cancer Wisconsin (Diagnostic)** dataset. The project was initially developed collaboratively by **Christine Wu**, **Haoyuan Liu**, and **Harsimran Kaur** as part of the **UC Berkeley Extension ML Course** and was later extended and refined by **Christine Wu** and **Kitty Li** for a machine learning course at **NYU**.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Methodology](#methodology)
4. [Results](#results)
5. [Technologies and Tools](#technologies-and-tools)
6. [Contributors](#contributors)
7. [Usage Instructions](#usage-instructions)
8. [Acknowledgments](#acknowledgments)

---

## Introduction

Breast cancer is one of the most commonly diagnosed cancers worldwide. Early detection and accurate classification of tumors as **Malignant (M)** or **Benign (B)** are critical for effective treatment. This project aims to automate breast cancer diagnosis by evaluating and comparing the following machine learning models:

1. **Support Vector Machines (SVM)**
2. **Decision Trees (DT)**
3. **Random Forests (RF)**
4. **Stacked Ensemble Model**

We utilized advanced techniques such as **Principal Component Analysis (PCA)** for dimensionality reduction and **DBSCAN Clustering** for exploratory analysis, alongside traditional feature selection methods.

---

## Dataset

The project utilizes the [Breast Cancer Wisconsin (Diagnostic) dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data), which contains 569 instances and 32 attributes, including:

- **30 numerical features** extracted from Fine Needle Aspiration (FNA) biopsies.
- **Target variable**: `diagnosis` (Benign or Malignant).

### Data Cleaning
- Removed rows with zero or invalid values.
- Detected and removed outliers using a custom function based on the IQR rule.

### Feature Selection
- Applied **Decision Tree** and **Random Forest** feature importance scores to identify the top 20 features.

---

## Methodology

1. **Data Visualization**:
   - Histograms and heat maps for feature correlations.
   - Pairwise plots to assess separability of malignant vs. benign cases.

2. **Dimensionality Reduction**:
   - Conducted PCA to reduce data dimensions and visualize separability in 2D.

3. **Model Development**:
   - Trained and tuned SVM, Decision Tree, Random Forest, and Stacked Ensemble models.
   - Hyperparameter tuning via **GridSearchCV**.

4. **Performance Metrics**:
   - Accuracy, R², Mean Squared Error (MSE), Root Mean Squared Error (RMSE).

---

## Results

### Key Findings:
- **Support Vector Machine (SVM)** achieved the highest accuracy of **98.2%** after feature selection and tuning.
- PCA and DBSCAN clustering confirmed clear class separability along principal components.
- While the **Stacked Ensemble Model** provided robust results (97.3% accuracy), it failed to outperform the standalone SVM.

### Model Performance Summary:

| Model           | Accuracy Before FS | Accuracy After FS |
|------------------|--------------------|-------------------|
| Decision Tree    | 95.50%            | 95.50%           |
| Random Forest    | 96.40%            | 97.30%           |
| Support Vector Machine (SVM) | 97.30% | **98.20%**       |
| Stacked Ensemble | 97.30%            | 97.30%           |

---

## Technologies and Tools

- **Programming Language**: Python
- **Libraries and Frameworks**:
  - `scikit-learn` for model implementation and evaluation.
  - `matplotlib`, `seaborn` for data visualization.
  - `pandas`, `numpy` for data manipulation and analysis.
- **Algorithms**:
  - Support Vector Machine (SVM)
  - Decision Tree (DT)
  - Random Forest (RF)
  - Principal Component Analysis (PCA)
  - DBSCAN Clustering

---

## Contributors

This project was developed and refined in two phases:

1. **UC Berkeley Extension ML Course**:
   - **Christine Wu**
   - **Haoyuan Liu**
   - **Harsimran Kaur**

2. **NYU ML Course**:
   - **Christine Wu**
   - **Kitty Li**

---

## Usage Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/breast-cancer-classification.git
   cd breast-cancer-classification
