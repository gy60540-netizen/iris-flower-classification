# Iris Flower Classification

A machine learning project developed as part of the **CodeAlpha Data Science Internship — Task 1**. The project uses flower measurements to predict the species of an Iris flower with Python and Scikit-learn.

## Overview

The Iris dataset contains measurements of three Iris flower species. This project trains and compares three machine learning models to identify the most suitable model for classifying these flowers.

The project also evaluates the selected model on a separate test dataset to understand how well it predicts flower species.

## Objectives

- Understand and prepare the Iris dataset.
- Use flower measurements to predict the species.
- Train and compare three machine learning models.
- Evaluate model performance using accuracy and a classification report.
- Demonstrate predictions on sample flower measurements.

## Dataset

The project uses `Iris.csv`, which contains **150 records** representing three Iris species.

| Detail | Description |
|---|---|
| Total records | 150 |
| Input features | 4 flower measurements |
| Target column | `Species` |
| Flower species | Setosa, Versicolor, Virginica |
| Missing values | None found |

### Input Features

The model uses these four measurements:

- `SepalLengthCm` — Length of the sepal
- `SepalWidthCm` — Width of the sepal
- `PetalLengthCm` — Length of the petal
- `PetalWidthCm` — Width of the petal

The `Id` column, when present, is removed because it identifies a record rather than describing the flower.

## Machine Learning Models

The project compares three models:

1. **Logistic Regression** — A model used to classify observations into categories.
2. **K-Nearest Neighbors (KNN)** — Predicts a flower's species by comparing its measurements with nearby examples.
3. **Random Forest Classifier** — Combines multiple decision trees to make a prediction.

The project uses feature scaling for Logistic Regression and KNN. The models are compared using 5-fold cross-validation on the training data.

## Project Workflow

1. Load the dataset and check its columns and values.
2. Separate the flower measurements from the species labels.
3. Split the data into 80% training data and 20% test data.
4. Compare the three models using cross-validation on the training data.
5. Select the model with the highest mean cross-validation accuracy.
6. Train the selected model on the full training dataset.
7. Evaluate it on the held-out test dataset.
8. Display predictions for three sample flowers.

## Results

The following results were obtained from the project run:

### Model Comparison

| Model | Mean Cross-Validation Accuracy |
|---|---:|
| Logistic Regression | 95.83% |
| K-Nearest Neighbors (KNN) | 96.67% |
| Random Forest Classifier | 95.00% |

K-Nearest Neighbors was selected because it achieved the highest mean cross-validation accuracy.

### Final Test Performance

| Metric | Result |
|---|---:|
| Selected model | K-Nearest Neighbors (KNN) |
| Test samples | 30 |
| Correct predictions | 28 |
| Test accuracy | 93.33% |

### Classification Summary

| Species | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Iris-setosa | 1.00 | 1.00 | 1.00 |
| Iris-versicolor | 0.83 | 1.00 | 0.91 |
| Iris-virginica | 1.00 | 0.80 | 0.89 |

The model correctly classified all 10 Setosa and all 10 Versicolor test samples. Of the 10 Virginica samples, it correctly classified 8 and predicted 2 as Versicolor.

*Note: These results are from one train-test split. Performance may differ with another dataset split or a different dataset.*

## Sample Predictions

The project also tests the selected model on three example flower measurements.

| Sample | Predicted Species |
|---|---|
| Sample 1 | Iris-setosa |
| Sample 2 | Iris-versicolor |
| Sample 3 | Iris-virginica |

These are demonstration predictions using predefined measurements, not interactive user input.

## Project Structure

```text
iris-flower-classification/
├── Iris.csv
├── iris_classification.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Requirements

- Python 3
- NumPy
- Pandas
- Scikit-learn

## How to Run the Project

### 1. Clone the repository

```bash
git clone <your-repository-url>
```

### 2. Open the project folder

```bash
cd CodeAlpha_IrisFlowerClassification
```

### 3. Install the required libraries

```bash
python -m pip install -r requirements.txt
```

### 4. Run the Python script

```bash
python iris_classification.py
```

Make sure `Iris.csv` is in the same folder as `iris_classification.py`.

## What I Learned

Through this project, I practiced:

- Working with a dataset using Pandas.
- Separating input features and target labels.
- Splitting data into training and test sets.
- Training and comparing machine learning models.
- Using cross-validation for model comparison.
- Evaluating predictions using accuracy, precision, recall, F1-score, and a confusion matrix.

## Future Improvements

- Test the model on additional unseen flower measurements.
- Experiment with different model settings.
- Add a simple interface where users can enter flower measurements and receive predictions.

## Author

Shreyash

**Project:** CodeAlpha Data Science Internship — Task 1: Iris Flower Classification
