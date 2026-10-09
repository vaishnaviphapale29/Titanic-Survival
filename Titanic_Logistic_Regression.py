import pandas as pd
import joblib
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

# ---------------------------------------------------------
# Project Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "TitanicDataset.csv"
MODEL_PATH = BASE_DIR / "marvelloustitanic.pkl"

CONFUSION_MATRIX_PATH = BASE_DIR / "confusion_matrix_heatmap.png"
SURVIVAL_COUNT_PATH = BASE_DIR / "survival_count_plot.png"
ACTUAL_PREDICTED_PATH = BASE_DIR / "actual_vs_predicted.png"

# ---------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ---------------------------------------------------------
# Display Section
# ---------------------------------------------------------

def DisplayInfo(title):

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

# ---------------------------------------------------------
# Visualization 1: Survival Count Plot
# ---------------------------------------------------------

def PlotSurvivalCount(df):

    DisplayInfo("VISUALIZATION 1 : SURVIVAL COUNT PLOT")

    plt.figure(figsize=(8, 6))

    ax = sns.countplot(
        data=df,
        x="Survived"
    )

    ax.set_title("Titanic Passenger Survival Count")
    ax.set_xlabel("Survival Status (0 = No, 1 = Yes)")
    ax.set_ylabel("Number of Passengers")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Not Survived", "Survived"])

    for container in ax.containers:
        ax.bar_label(container)

    plt.tight_layout()
    plt.savefig(SURVIVAL_COUNT_PATH, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


    print("Survival count plot saved:", SURVIVAL_COUNT_PATH)

# ---------------------------------------------------------
# Visualization 2: Confusion Matrix Heatmap
# ---------------------------------------------------------

def PlotConfusionMatrix(Y_test, Y_pred):

    DisplayInfo("VISUALIZATION 2 : CONFUSION MATRIX HEATMAP")

    cm = confusion_matrix(
        Y_test,
        Y_pred,
        labels=[0, 1]
    )

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Survived", "Survived"],
        yticklabels=["Not Survived", "Survived"]
    )

    plt.title("Titanic Logistic Regression - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")

    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    print("Confusion matrix heatmap saved:", CONFUSION_MATRIX_PATH)

# ---------------------------------------------------------
# Visualization 3: Actual vs Predicted Plot
# ---------------------------------------------------------

def PlotActualVsPredicted(Y_test, Y_pred):

    DisplayInfo("VISUALIZATION 3 : ACTUAL VS PREDICTED")

    actual = list(Y_test)
    predicted = list(Y_pred)

    sample_numbers = list(range(1, len(actual) + 1))

    plt.figure(figsize=(12, 6))

    plt.plot(
        sample_numbers,
        actual,
        marker="o",
        linestyle="-",
        label="Actual Survival"
    )

    plt.plot(
        sample_numbers,
        predicted,
        marker="x",
        linestyle="--",
        label="Predicted Survival"
    )

    plt.yticks(
        [0, 1],
        ["Not Survived", "Survived"]
    )

    plt.xlabel("Test Passenger Number")
    plt.ylabel("Survival Status")
    plt.title("Actual vs Predicted Titanic Survival")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(ACTUAL_PREDICTED_PATH, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    print("Actual vs predicted plot saved:", ACTUAL_PREDICTED_PATH)

# ---------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------

def LoadDataset(DataPath=None):

    DisplayInfo("STEP 1 : LOAD DATASET")

    try:

        df = pd.read_csv(DATASET_PATH)

        print("Dataset loaded successfully")

        print("\nFirst 5 records:")
        print(df.head())

        print("\nShape of dataset:")
        print(df.shape)

        print("\nColumn names:")
        print(df.columns.tolist())

        return df

    except Exception as e:

        logging.error("Error while loading dataset: %s", e)
        print("Error:", e)

        return None

# ---------------------------------------------------------
# Show Dataset Information
# ---------------------------------------------------------

def ShowData(df):

    DisplayInfo("DATASET INFORMATION")

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nDataset shape:")
    print(df.shape)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nData types:")
    print(df.dtypes)

# ---------------------------------------------------------
# Clean Dataset
# ---------------------------------------------------------

def CleanTitanicData(df):

    DisplayInfo("STEP 2 : DATA CLEANING")

    df = df.copy()

    drop_columns = [
        "Passengerid",
        "zero"
    ]

    existing_columns = [
        col for col in drop_columns
        if col in df.columns
    ]

    print("\nColumns removed:")
    print(existing_columns)

    df.drop(
        columns=existing_columns,
        inplace=True
    )

    print("\nDataset after removing unwanted columns:")
    print(df.head())

    print("\nShape after removing columns:")
    print(df.shape)

    if "Survived" not in df.columns:
        raise ValueError("Target column 'Survived' not found")

    feature_columns = [
        col for col in df.columns
        if col != "Survived"
    ]

    for col in feature_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    return df

# ---------------------------------------------------------
# Save Model
# ---------------------------------------------------------

def SaveModel(model, filename=MODEL_PATH):

    try:

        joblib.dump(model, filename)

        print("\nModel saved successfully as:", filename)

    except Exception as e:

        print("Error while saving model:", e)

# ---------------------------------------------------------
# Load Model
# ---------------------------------------------------------

def LoadSavedModel(filename=MODEL_PATH):

    try:

        model = joblib.load(filename)

        print("\nModel loaded successfully from:", filename)

        return model

    except Exception as e:

        print("Error while loading model:", e)

        return None

# ---------------------------------------------------------
# Train Titanic Model
# ---------------------------------------------------------

def TrainTitanicModel(df):

    DisplayInfo("STEP 3 : PREPARE FEATURES AND TARGET")

    X = df.drop("Survived", axis=1)
    Y = df["Survived"]

    print("\nFeatures:")
    print(X.head())

    print("\nTarget:")
    print(Y.head())

    print("\nShape of X:")
    print(X.shape)

    print("\nShape of Y:")
    print(Y.shape)

    # -----------------------------------------------------
    # Train Test Split
    # -----------------------------------------------------

    DisplayInfo("STEP 4 : TRAIN TEST SPLIT")

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42,
        stratify=Y
    )

    print("X_train shape:", X_train.shape)
    print("X_test shape :", X_test.shape)
    print("Y_train shape:", Y_train.shape)
    print("Y_test shape :", Y_test.shape)

    # -----------------------------------------------------
    # Create Numeric Preprocessing Pipeline
    # -----------------------------------------------------

    DisplayInfo("STEP 5 : CREATE ML PIPELINE")

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ])

    model_pipeline = Pipeline([
        (
            "preprocessor",
            numeric_pipeline
        ),
        (
            "model",
            LogisticRegression(max_iter=1000)
        )
    ])

    print("\nPipeline created successfully")

    # -----------------------------------------------------
    # Model Training
    # -----------------------------------------------------

    DisplayInfo("STEP 6 : MODEL TRAINING")

    model_pipeline.fit(X_train, Y_train)

    print("Model trained successfully")

    # -----------------------------------------------------
    # Model Prediction
    # -----------------------------------------------------

    DisplayInfo("STEP 7 : MODEL PREDICTION")

    Y_pred = model_pipeline.predict(X_test)

    print("\nActual values:")
    print(Y_test.values)

    print("\nPredicted values:")
    print(Y_pred)

    # -----------------------------------------------------
    # Model Evaluation
    # -----------------------------------------------------

    DisplayInfo("STEP 8 : MODEL EVALUATION")

    accuracy = accuracy_score(Y_test, Y_pred)

    precision = precision_score(
        Y_test,
        Y_pred,
        zero_division=0
    )

    recall = recall_score(
        Y_test,
        Y_pred,
        zero_division=0
    )

    f1 = f1_score(
        Y_test,
        Y_pred,
        zero_division=0
    )

    print("\nAccuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)

    print("\nConfusion Matrix:")

    cm = confusion_matrix(Y_test, Y_pred)
    print(cm)

    print("\nClassification Report:")

    print(
        classification_report(
            Y_test,
            Y_pred,
            zero_division=0
        )
    )

    # -----------------------------------------------------
    # Generate Visualizations
    # -----------------------------------------------------

    PlotConfusionMatrix(Y_test, Y_pred)
    PlotActualVsPredicted(Y_test, Y_pred)

    # -----------------------------------------------------
    # Cross Validation
    # -----------------------------------------------------

    DisplayInfo("STEP 9 : CROSS VALIDATION")

    cv_scores = cross_val_score(
        model_pipeline,
        X_train,
        Y_train,
        cv=5,
        scoring="accuracy"
    )

    print("Cross Validation Scores:")
    print(cv_scores)

    print("\nMean Cross Validation Accuracy:", cv_scores.mean())

    # -----------------------------------------------------
    # Model Coefficients
    # -----------------------------------------------------

    DisplayInfo("STEP 10 : MODEL COEFFICIENTS")

    logistic_model = model_pipeline.named_steps["model"]

    print("\nIntercept:")
    print(logistic_model.intercept_)

    print("\nCoefficients:")

    for feature, coefficient in zip(
        X.columns,
        logistic_model.coef_[0]
    ):

        print(feature, ":", coefficient)

    # -----------------------------------------------------
    # Save Complete Pipeline
    # -----------------------------------------------------

    DisplayInfo("STEP 11 : SAVE MODEL")

    SaveModel(
        model_pipeline,
        MODEL_PATH
    )

    # -----------------------------------------------------
    # Load Saved Pipeline
    # -----------------------------------------------------

    DisplayInfo("STEP 12 : LOAD SAVED MODEL")

    loaded_model = LoadSavedModel(MODEL_PATH)

    # -----------------------------------------------------
    # Test Loaded Model
    # -----------------------------------------------------

    if loaded_model is not None:

        loaded_prediction = loaded_model.predict(X_test)

        print("\nPrediction using loaded model:")
        print(loaded_prediction)

        loaded_accuracy = accuracy_score(
            Y_test,
            loaded_prediction
        )

        print("\nLoaded model accuracy:", loaded_accuracy)

    return model_pipeline

# ---------------------------------------------------------
# Main Titanic Function
# ---------------------------------------------------------

def TitanicLogisticRegression(DataPath=None):

    try:

        df = LoadDataset(DataPath)

        if df is None:
            return

        # Generate survival count plot from original dataset
        PlotSurvivalCount(df)

        ShowData(df)

        df = CleanTitanicData(df)

        TrainTitanicModel(df)

    except Exception as e:

        logging.exception("Application failed")
        print("\nError:", e)

# ---------------------------------------------------------
# Program Execution
# ---------------------------------------------------------

def main():

    TitanicLogisticRegression(DATASET_PATH)

if __name__ == "__main__":
    main()