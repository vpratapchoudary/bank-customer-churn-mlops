import os
from pathlib import Path

from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo

from config import huggingface_config
from utils.logs import logger

DATA_DIR = (Path(__file__).resolve().parent.parent / "data").resolve()

repo_id = f"{huggingface_config['user_id']}/{huggingface_config['dataset_repo_name']}"
repo_type = "dataset"

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    logger.info(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    logger.info(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    logger.info(f"Space '{repo_id}' created.")

api.upload_folder(
    folder_path=str(DATA_DIR),
    repo_id=repo_id,
    repo_type=repo_type,
)
logger.info(f"Data uploaded to space '{repo_id}'.")