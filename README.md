# Bank Customer Churn MLOps

A machine learning project to predict whether a bank customer is likely to churn using customer profile attributes and a production-style MLOps workflow. The solution includes data preparation, feature engineering, model training with hyperparameter tuning, experiment tracking with MLflow, and a Streamlit-based prediction app for inference.

Live app: https://bank-customer-churn-mlops-pratapv.streamlit.app/

## Overview

This project focuses on the bank customer churn problem: predicting whether a customer will leave the bank based on their credit score, geography, age, tenure, balance, product usage, activity status, and salary profile.

The model is built using a preprocessing pipeline and an XGBoost classifier, tuned with `GridSearchCV`. The final trained model is uploaded to Hugging Face Hub and then downloaded by the Streamlit application for online prediction.

## Business Problem

Customer churn is a critical metric for banks because losing customers can significantly reduce revenue and increase acquisition costs. By identifying customers at high risk of churn early, a bank can take proactive retention actions such as:

- personalized offers
- loyalty programs
- customer support outreach
- targeted financial engagement strategies

## Project Goals

- build a reliable churn prediction model
- automate the training pipeline using MLOps practices
- track experiments and model metrics with MLflow
- store datasets and model artifacts in Hugging Face
- deploy a simple interactive inference app with Streamlit

## Project Architecture

The project is organized into the following components:

- `src/app.py` — Streamlit user interface for inference
- `src/config.py` — Hugging Face configuration values
- `src/model_building/prep.py` — dataset split and upload to Hugging Face
- `src/model_building/train.py` — model training, hyperparameter search, and artifact upload
- `src/model_building/features.py` — selected numerical and categorical features
- `src/utils/logs.py` — logger setup
- `data/bank_customer_churn.csv` — raw dataset used for the project
- `Dockerfile` — container setup for serving the Streamlit app

## Data

The project uses the Bank Customer Churn dataset with features such as:

- `CreditScore`
- `Geography`
- `Age`
- `Tenure`
- `Balance`
- `NumOfProducts`
- `HasCrCard`
- `IsActiveMember`
- `EstimatedSalary`
- target variable: `Exited`

The data is split into training and testing sets with a fixed random state to preserve reproducibility.

## Model Pipeline

The churn model pipeline includes:

- preprocessing with `StandardScaler` for numeric features
- one-hot encoding for categorical features
- `XGBClassifier` as the predictive model
- `GridSearchCV` for hyperparameter tuning
- class weighting to address the class imbalance in churn prediction
- a decision threshold of `0.45` for converting probabilities into churn/not-churn labels

## Experiment Tracking and Model Registry

This project uses:

- `mlflow` for experiment tracking
- Hugging Face Hub for dataset and model artifact storage
- joblib serialization for saving the trained model

The configured repository names in `src/config.py` are:

- dataset repo: `PratapVanka/bank-customer-churn`
- model repo: `PratapVanka/churn-model`

## Streamlit App

The interactive app is available online here:

https://bank-customer-churn-mlops-pratapv.streamlit.app/

The app lets users enter a customer's information and receive a churn prediction. It downloads the trained model from Hugging Face and runs inference directly in the browser.

## Repository Structure

```text
bank-customer-churn-mlops/
├── Dockerfile
├── pyproject.toml
├── README.md
├── data/
│   └── bank_customer_churn.csv
├── src/
│   ├── app.py
│   ├── config.py
│   ├── model_building/
│   │   ├── data_register.py
│   │   ├── features.py
│   │   ├── prep.py
│   │   └── train.py
│   └── utils/
│       └── logs.py
└── bank_customer_churn_mlops.egg-info/
```

## Local Setup

1. Clone the repository

```bash
git clone <repository-url>
cd bank-customer-churn-mlops
```

2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies

```bash
pip install -e .
```

4. Set the Hugging Face token if you plan to upload or refresh artifacts

```bash
export HF_TOKEN="your_hugging_face_token"
```

## Running the Project

### 1) Prepare the dataset

```bash
python src/model_building/prep.py
```

This script reads the raw dataset, splits it into train/test sets, and uploads the generated CSV files to the configured Hugging Face dataset repository.

### 2) Train the model

```bash
python src/model_building/train.py
```

This script:

- loads the processed dataset from Hugging Face
- builds a preprocessing + XGBoost pipeline
- performs hyperparameter tuning with `GridSearchCV`
- logs runs in MLflow
- saves the best model as `best_churn_model.joblib`
- uploads the model artifact to Hugging Face

### 3) Launch the Streamlit app

```bash
streamlit run src/app.py
```

After launch, open the local URL in the browser (usually http://localhost:8501) to interact with the app.

## Docker Usage

The project includes a Dockerfile for containerized deployment of the Streamlit app.

```bash
docker build -t bank-churn-app .
docker run -p 8501:8501 bank-churn-app
```

Then open:

```text
http://localhost:8501
```

## Example Inference

The app asks for information such as:

- credit score
- geography
- age
- tenure
- account balance
- number of products
- credit card ownership
- activity status
- estimated salary

It then predicts a probability of churn and converts it into a binary churn decision based on the configured threshold.

## Notes

- The app uses a public model repository and downloads the trained artifact at runtime.
- The production workflow is designed for experimentation and iterative model improvement.
- MLflow tracking is configured for local experiment monitoring via `http://localhost:5000`.

## Summary

This project demonstrates a practical end-to-end MLOps workflow for a churn prediction use case. It blends data science, model training, tracking, artifact management, and application deployment in a simple and understandable structure suitable for learning and portfolio purposes.
