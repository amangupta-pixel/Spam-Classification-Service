import os
import sys
import numpy as np
import pandas as pd

"""
Defining common constant variable for training pipeline
"""

TARGET_COLUMN = "label"
FILE_NAME: str = "sms_spam_dataset.csv"
PIPELINE_NAME: str = "Spam-Ham-Classification"
ARTIFACT_DIR: str = "Artifacts"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
SCHEMA_FILE_PATH = os.path.join("data_schema","schema.yaml")

"Data Ingestion related constants starting with DATA_INGESTION VAR NAME"

DATA_INGESTION_DATABASE_NAME: str = "TESTAI"
DATA_INGESTION_COLLECTION_NAME: str = "SpamData"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION: float = 0.2

"Data Validation related constants starting with DATA_VALIDATION VAR NAME"

DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_INVALID_DIR: str = "invalid"

"Data Transformation related constants starting with DATA_TRANSFORMATION VAR NAME"

DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

# KNN Imputer to replace NAN values
DATA_TRANSFORMATION_IMPUTER_PARAMS: dict = {
    "missing_values": np.nan,
    "n_neighbours": 3,
    "weights": "uniform"
}

DATA_TRANSFORMATION_TRAIN_FILE_PATH: str = "train.npy"
DATA_TRANSFORMATION_TEST_FILE_PATH: str = "test.npy"
PREPROCESSING_OBJECT_FILE_NAME: str = "preprocessing.pkl"

"Model Training related constants starting with MODEL_TRAINER VAR NAME"

MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD: float = 0.05