"""
CodeAlpha Data Science Internship
Task 1: Iris Flower Classification
Author: Shreyash
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


# -------------------- Configuration --------------------

FEATURE_COLUMNS = [
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm",
]

TARGET_COLUMN = "Species"

CLASS_ORDER = [
    "Iris-setosa",
    "Iris-versicolor",
    "Iris-virginica",
]

VALID_SPECIES = set(CLASS_ORDER)
RANDOM_STATE = 42


# -------------------- Dataset Validation --------------------

def load_and_validate_dataset(csv_path: Path) -> pd.DataFrame:
    """Load the CSV and validate its schema and data."""

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}\n"
            "Place Iris.csv in the same folder as this Python script."
        )

    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError("The dataset is empty.")

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # The Id column is an identifier, not a flower measurement.
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])

    # Check missing values in required columns.
    missing_counts = df[required_columns].isnull().sum()
    missing_counts = missing_counts[missing_counts > 0]

    if not missing_counts.empty:
        details = ", ".join(
            f"{column}: {count}"
            for column, count in missing_counts.items()
        )
        raise ValueError(
            f"Missing values found ({details}). "
            "Please clean the dataset before training."
        )

    print("- Missing values check: passed.")

    # Validate the four measurement columns.
    for column in FEATURE_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise TypeError(
                f"Feature '{column}' must contain numeric values."
            )

        if not np.isfinite(df[column]).all():
            raise ValueError(
                f"Feature '{column}' contains infinite or invalid values."
            )

        if (df[column] < 0).any():
            raise ValueError(
                f"Feature '{column}' contains negative measurements."
            )

    print("- Numeric and measurement checks: passed.")

    # Validate target labels.
    unexpected_species = set(df[TARGET_COLUMN].unique()) - VALID_SPECIES

    if unexpected_species:
        raise ValueError(
            f"Unexpected species labels: {unexpected_species}"
        )

    absent_species = VALID_SPECIES - set(df[TARGET_COLUMN].unique())

    if absent_species:
        raise ValueError(
            f"Expected species are missing from the dataset: "
            f"{absent_species}"
        )

    print("- Species label checks: passed.")

    # Report duplicates without automatically deleting observations.
    duplicate_count = int(df.duplicated().sum())

    print(f"- Duplicate rows found: {duplicate_count}")
    if duplicate_count:
        print("- Duplicate observations are preserved.")

    return df


# -------------------- Model Definitions --------------------

def build_models():
    """Create the candidate classification models."""

    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]),

        "K-Nearest Neighbors (KNN)": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=3)),
        ]),

        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
        ),
    }


# -------------------- Main Workflow --------------------

def main():
    print("=" * 65)
    print("IRIS FLOWER CLASSIFICATION - CODEALPHA TASK 1")
    print("=" * 65)

    project_dir = Path(__file__).resolve().parent
    csv_file = project_dir / "Iris.csv"

    print(f"\n[1] Loading dataset: {csv_file.name}")

    try:
        df = load_and_validate_dataset(csv_file)

    except FileNotFoundError as error:
        print(f"[Dataset Error] {error}", file=sys.stderr)
        sys.exit(1)

    except pd.errors.ParserError as error:
        print(f"[CSV Parsing Error] {error}", file=sys.stderr)
        sys.exit(1)

    except UnicodeDecodeError as error:
        print(f"[File Encoding Error] {error}", file=sys.stderr)
        sys.exit(1)

    except OSError as error:
        print(f"[File Error] {error}", file=sys.stderr)
        sys.exit(1)

    except (ValueError, TypeError) as error:
        print(f"[Validation Error] {error}", file=sys.stderr)
        sys.exit(1)

    print(f"- Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\n- Species distribution:")
    print(df[TARGET_COLUMN].value_counts().to_string())

    # Separate input features (X) and target labels (y).
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # Split before cross-validation so the test set stays held out.
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    except ValueError as error:
        print(f"[Split Error] {error}", file=sys.stderr)
        sys.exit(1)

    print("\n[2] Stratified train-test split")
    print(f"- Training samples: {len(X_train)}")
    print(f"- Test samples: {len(X_test)}")

    # Every class must have at least 5 training samples for 5-fold CV.
    train_class_counts = y_train.value_counts()

    if (
        len(train_class_counts) != len(CLASS_ORDER)
        or train_class_counts.min() < 5
    ):
        print(
            "[Validation Error] Each species needs at least "
            "5 training samples for 5-fold stratified CV.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Compare models using training data only.
    print("\n[3] Model selection: 5-fold cross-validation")

    models = build_models()
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    cv_results = {}

    try:
        for name, model in models.items():
            scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=cv,
                scoring="accuracy",
                error_score="raise",
            )

            mean_score = float(scores.mean())
            std_score = float(scores.std())

            cv_results[name] = {
                "model": model,
                "mean_accuracy": mean_score,
                "std_accuracy": std_score,
            }

            print(
                f"- {name}: "
                f"{mean_score * 100:.2f}% "
                f"(standard deviation: {std_score * 100:.2f}%)"
            )

    except (ValueError, TypeError) as error:
        print(f"[Training Error] {error}", file=sys.stderr)
        sys.exit(1)

    # Select the model with the highest mean CV accuracy.
    best_model_name = max(
        cv_results,
        key=lambda name: cv_results[name]["mean_accuracy"],
    )

    selected_model = cv_results[best_model_name]["model"]

    print(f"\n- Selected model: {best_model_name}")
    print(
        "- Selection criterion: highest mean cross-validation accuracy."
    )

    # Refit the selected model using the complete training set.
    selected_model.fit(X_train, y_train)

    # Evaluate once on the held-out test set.
    print("\n" + "=" * 65)
    print(f"[4] FINAL TEST EVALUATION: {best_model_name}")
    print("=" * 65)

    y_test_pred = selected_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print(f"\nTest accuracy: {test_accuracy * 100:.2f}%")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            y_test_pred,
            labels=CLASS_ORDER,
            zero_division=0,
        )
    )

    cm = confusion_matrix(
        y_test,
        y_test_pred,
        labels=CLASS_ORDER,
    )

    cm_df = pd.DataFrame(
        cm,
        index=CLASS_ORDER,
        columns=CLASS_ORDER,
    )

    print("Confusion matrix (rows = actual, columns = predicted):")
    print(cm_df)

    # Demonstrate predictions on three example measurements.
    print("\n" + "=" * 65)
    print("[5] SAMPLE PREDICTIONS")
    print("=" * 65)

    sample_df = pd.DataFrame(
        [
            [5.1, 3.5, 1.4, 0.2],
            [6.0, 2.9, 4.5, 1.5],
            [6.8, 3.0, 5.5, 2.1],
        ],
        columns=FEATURE_COLUMNS,
    )

    sample_predictions = selected_model.predict(sample_df)

    for index, (features, prediction) in enumerate(
        zip(sample_df.to_dict(orient="records"), sample_predictions),
        start=1,
    ):
        print(f"Sample {index}:")
        print(f"  Measurements: {features}")
        print(f"  Predicted species: {prediction}")

    print("\n" + "=" * 65)
    print("CLASSIFICATION PIPELINE COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()
