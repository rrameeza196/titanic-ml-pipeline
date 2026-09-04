# Titanic Survival Prediction — End-to-End ML Pipeline

A complete, working machine learning pipeline built on the classic Titanic dataset — from raw data to an evaluated model, using pandas and scikit-learn.

## What this project does

1. **Data Exploration** — load the Titanic dataset, check missing values, and look at survival rate by `Sex` and `Pclass`.
2. **Data Preprocessing** — fill missing `Age` (median) and `Embarked` (mode), encode `Sex` (binary) and `Embarked` (one-hot).
3. **Train/Test Split** — 80/20 stratified split, with features scaled using `StandardScaler`.
4. **Model Training & Evaluation** — train and compare two classifiers:
   - Logistic Regression
   - Decision Tree
   
   Evaluated using accuracy, confusion matrix, and precision/recall.
5. **Reflection** — key takeaways and ideas for next steps (feature engineering, cross-validation, hyperparameter tuning).

## Results

| Model | Accuracy |
|---|---|
| Logistic Regression | ~80.4% |
| Decision Tree | ~77.7% |

`Sex` was the strongest predictor of survival, followed by `Pclass` and `Fare` — consistent with the historical "women and children first" evacuation pattern.

## How to run

1. Open [Google Colab](https://colab.research.google.com).
2. Upload `Titanic_ML_Pipeline.ipynb` via **File > Upload notebook**.
3. Click **Connect** in the top-right corner.
4. Run **Runtime > Run all** (internet access is required — the dataset is loaded from a URL).

No local setup or installation needed.

## Dataset

[Titanic dataset](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv) (same data as Kaggle's `train.csv`).

## Tools used

- Python, pandas, numpy
- scikit-learn (LogisticRegression, DecisionTreeClassifier, preprocessing, metrics)
- matplotlib
