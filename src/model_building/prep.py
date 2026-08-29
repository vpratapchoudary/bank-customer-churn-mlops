import os

import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import login, HfApi

from config import huggingface_config
from model_building.features import numeric_features, categorical_features, target
from utils.logs import logger


# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = f"hf://datasets/{huggingface_config['user_id']}/{huggingface_config['dataset_repo_name']}/{huggingface_config['dataset_filename']}"
bank_dataset = pd.read_csv(DATASET_PATH)
logger.info(f"Dataset loaded successfully from {DATASET_PATH}.")

# Define predictor matrix (X) using selected numeric and categorical features
X = bank_dataset[numeric_features + categorical_features]

# Define target variable
y = bank_dataset[target]

# Split dataset into train and test
# Split the dataset into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,              # Predictors (X) and target variable (y)
    test_size=0.2,     # 20% of the data is reserved for testing
    random_state=42    # Ensures reproducibility by setting a fixed random seed
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)

files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id=f"{huggingface_config['user_id']}/{huggingface_config['dataset_repo_name']}",
        repo_type="dataset",
    )