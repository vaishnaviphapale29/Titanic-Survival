# Titanic Survival Prediction Using Logistic Regression

## Project Overview

This project uses Logistic Regression, a supervised machine learning algorithm, to predict the survival of Titanic passengers based on passenger-related features.

The project includes dataset loading, data cleaning, preprocessing, model training, prediction, evaluation, cross-validation, and model saving using Joblib. It also generates visualizations to understand survival distribution and evaluate model predictions.

## Features

- Dataset loading using Pandas
- Dataset information and missing-value analysis
- Data cleaning and unwanted column removal
- Missing-value handling using SimpleImputer
- Feature scaling using StandardScaler
- Machine learning pipeline using Scikit-learn
- Logistic Regression model training
- Train-test split with stratification
- Model evaluation using Accuracy, Precision, Recall, F1-score, Confusion Matrix, and Classification Report
- 5-fold cross-validation
- Model coefficients and intercept display
- Model saving and loading using Joblib
- Survival Count Plot
- Confusion Matrix Heatmap
- Actual vs Predicted Plot
- Visualization images saved as PNG files and displayed using `plt.show()`

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Logging
- Pathlib

## Dataset Information

Dataset file: `TitanicDataset.csv`

The dataset contains passenger information and the target column `Survived`.

- `Survived = 0`: Passenger did not survive
- `Survived = 1`: Passenger survived

The program removes the `Passengerid` and `zero` columns when they are present. The remaining columns, excluding `Survived`, are used as input features.

## Project Structure

```text
Titanic_Survival/
│
├── Titanic_Logistic_Regression.py
├── TitanicDataset.csv
├── requirements.txt
├── README.md
├── marvelloustitanic.pkl
├── survival_count_plot.png
├── confusion_matrix_heatmap.png
└── actual_vs_predicted.png
```

*Note: The model file and visualization images are generated when the program runs. Add them to GitHub only if you want to include the generated outputs in the repository.*

## Machine Learning Workflow

1. Load the Titanic dataset.
2. Display the first five records, dataset shape, column names, missing values, and data types.
3. Remove unwanted columns when present.
4. Convert feature columns to numeric values.
5. Separate input features (`X`) and target (`Y`).
6. Split the dataset into training and testing sets using an 80:20 ratio.
7. Create a preprocessing pipeline using median imputation and StandardScaler.
8. Train the Logistic Regression model.
9. Predict passenger survival on the test dataset.
10. Evaluate the model using classification metrics.
11. Generate the three visualizations.
12. Perform 5-fold cross-validation on the training data.
13. Display model coefficients and intercept.
14. Save the trained pipeline using Joblib.
15. Load the saved model and check its predictions.

## Data Preprocessing

The project uses a Scikit-learn Pipeline consisting of:

- **SimpleImputer:** Replaces missing numerical values using the median.
- **StandardScaler:** Scales numerical features.
- **LogisticRegression:** Performs binary classification.

The complete pipeline is saved so the preprocessing and prediction steps can be reused together.

## Visualizations

### 1. Survival Count Plot

Displays the number of passengers who survived and those who did not survive.

Output file: `survival_count_plot.png`

### 2. Confusion Matrix Heatmap

Shows the relationship between actual and predicted survival classes, helping identify correct and incorrect predictions.

Output file: `confusion_matrix_heatmap.png`

### 3. Actual vs Predicted Plot

Compares actual survival labels with the model's predicted labels for passengers in the test dataset.

Output file: `actual_vs_predicted.png`

All three plots are saved as high-resolution PNG images and displayed using `plt.show()`.

## Model Evaluation

The model is evaluated using the following metrics:

- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** Proportion of predicted survivors who actually survived.
- **Recall:** Proportion of actual survivors correctly identified.
- **F1-score:** Harmonic mean of precision and recall.
- **Confusion Matrix:** Summarizes correct and incorrect predictions.
- **Classification Report:** Displays precision, recall, F1-score, and support.
- **Cross-validation:** Reports five accuracy scores and their mean.

The actual metric values are displayed when the program runs.

## Model Saving and Loading

The trained machine learning pipeline is saved using Joblib.

Model file: `marvelloustitanic.pkl`

The program reloads the saved pipeline and evaluates its predictions on the test data.

## Installation

Clone the repository:

```bash
git clone https://github.com/vaishnaviphapale29/Titanic-Survival.git
```

Navigate to the project directory:

```bash
cd Titanic-Survival
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

## How to Run

Ensure that `TitanicDataset.csv` is present in the same folder as the Python script.

Run the project:

```bash
python Titanic_Logistic_Regression.py
```

The program displays dataset information, training progress, evaluation metrics, and visualization windows. It also saves the model and plot images in the project folder.

## Purpose

The purpose of this project is to develop a machine learning model that predicts Titanic passenger survival using Logistic Regression. It aims to demonstrate the complete machine learning workflow, including data preprocessing, model training, prediction, and performance evaluation. The project also uses visualizations to understand passenger survival distribution and compare actual outcomes with model predictions. Additionally, it demonstrates cross-validation and model saving and loading using Joblib.