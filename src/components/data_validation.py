import os
import sys
import pandas as pd
import numpy as np
from src.logging.logger import logging
from src.exception.exception import CustomException
from src.constant.training_pipeline import SCHEMA_FILE_PATH
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig
from src.utils.main_utils.utils import read_yaml_file

class DataValidation:
    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,
                 data_validation_config:DataValidationConfig):
        
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)

        except Exception as e:
            raise CustomException(e,sys)
        
    def read_data(self,file_path:str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e,sys)
        
    def validate_no_of_columns(self,dataframe:pd.DataFrame,dataset_type:str) -> bool:

        expected_columns = list(self._schema_config["processed_data"]["dataset_type"]["columns"].keys())
        logging.info(f"Validating {dataset_type} columns")
        logging.info(f"Expected columns: {len(expected_columns)}")
        logging.info(f"Actual columns: {len(dataframe.columns)}")

        return set(expected_columns) == set(dataframe.columns)
    

    def is_numerical_column_exists(self,dataframe:pd.DataFrame) -> bool:

        return dataframe.columns.dtype != 'O'

    def is_categorical_column_exists(self,dataframe:pd.DataFrame) -> bool:
        return dataframe.columns.dtype == 'O'     
        
    def initiate_data_validation(self) -> DataValidationArtifact:
        train_file_path = self.data_ingestion_artifact.trained_file_path
        test_file_path = self.data_ingestion_artifact.test_file_path

        # File-Level Validation (FIRST CHECK)
        os.makedirs(os.path.dirname(train_file_path),exist_ok=True)
        os.makedirs(os.path.dirname(test_file_path),exist_ok=True)

        train_dataframe = self.read_data(train_file_path)
        test_dataframe = self.read_data(test_file_path)

        # Schema Validation (SECOND CHECK)

        status = self.validate_no_of_columns(train_dataframe,dataset_type="train_data")
        if not status:
            raise CustomException("Train dataframe has invalid number of columns")
        
        status = self.validate_no_of_columns(test_dataframe,dataset_type="test_data")
        if not status:
            raise CustomException("Test dataframe has invalid number of columns")
        
        # DataType Validation (THIRD CHECK)

        status = self.is_numerical_column_exists(train_dataframe)
        if not status:
            raise CustomException("Numerical columns don't exist in training dataframe")
        
        status = self.is_categorical_column_exists(train_dataframe)
        if not status:
            raise CustomException("Categorical columns don't exist in training dataframe")
        
        status = self.is_numerical_column_exists(test_dataframe)
        if not status:
            raise CustomException("Numerical columns don't exist in test dataframe")
        
        status = self.is_categorical_column_exists(test_dataframe)
        if not status:
            raise CustomException("Categorical columns doesn't exist in test dataframe")
        

        

        


        








        