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
import warnings
warnings.filterwarnings('ignore')

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
        try:
            expected_columns = list(self._schema_config["processed_data"][dataset_type]["columns"].keys())
            logging.info(f"Validating {dataset_type} columns")
            logging.info(f"Expected columns: {len(expected_columns)}")
            logging.info(f"Actual columns: {len(dataframe.columns)}")

            logging.info("Schema Validation completed")
            return set(expected_columns) == set(dataframe.columns)
        except Exception as e:
            raise CustomException(e,sys)
    

    def is_numerical_column_exists(self,dataframe:pd.DataFrame) -> bool:
        try:
            numerical_columns = dataframe.select_dtypes(include=[np.number]).columns
            logging.info("Numerical Datatype Validation completed")
            return len(numerical_columns) > 0
        except Exception as e:
            raise CustomException(e,sys)

    def is_categorical_column_exists(self,dataframe:pd.DataFrame) -> bool:
        try:
            categorical_columns = dataframe.select_dtypes(include=['object', 'category']).columns
            logging.info("Categorical Datatype Validation completed")
            return len(categorical_columns) > 0
        except Exception as e:
            raise CustomException(e,sys)
    
    def is_nan_values_exist(self,dataframe:pd.DataFrame) -> pd.DataFrame:
        try:
            if dataframe.isnull().values.any():
                dataframe = dataframe.dropna()
                
            logging.info("Null Values Validation completed")    
            return dataframe
        except Exception as e:
            raise CustomException(e,sys)
    
    def validate_labels_integrity(self,dataframe:pd.DataFrame) -> bool:
        try:
            actual_labels = set(map(str,dataframe['label'].unique()))
            expected_labels = set(map(str, self._schema_config["target_column"]["classes"]))
            
            # Check if actual labels match the expected labels from schema
            logging.info("Labels Integrity Validation completed")
            return actual_labels == expected_labels
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            # File-Level Validation (FIRST CHECK)

            os.makedirs(os.path.dirname(train_file_path),exist_ok=True)
            os.makedirs(os.path.dirname(test_file_path),exist_ok=True)

            train_dataframe = self.read_data(train_file_path)
            logging.info(f"Ingested Train dataframe shape: {train_dataframe.shape}")
            test_dataframe = self.read_data(test_file_path)
            logging.info(f"Ingested Test dataframe shape: {test_dataframe.shape}")
            
            # Schema Validation (SECOND CHECK)

            status = self.validate_no_of_columns(train_dataframe,dataset_type="train_data")
            if not status:
                raise CustomException("Train dataframe has invalid number of columns", sys)
            
            status = self.validate_no_of_columns(test_dataframe,dataset_type="test_data")
            if not status:
                raise CustomException("Test dataframe has invalid number of columns", sys)
            
            # DataType Validation (THIRD CHECK)

            status = self.is_numerical_column_exists(train_dataframe)
            if not status:
                raise CustomException("Numerical columns don't exist in training dataframe", sys)
            
            status = self.is_categorical_column_exists(train_dataframe)
            if not status:
                raise CustomException("Categorical columns don't exist in training dataframe", sys)
            
            status = self.is_numerical_column_exists(test_dataframe)
            if not status:
                raise CustomException("Numerical columns don't exist in test dataframe", sys)
            
            status = self.is_categorical_column_exists(test_dataframe)
            if not status:
                raise CustomException("Categorical columns don't exist in test dataframe", sys)
            
            # Null & Empty Values Validation (FORTH CHECK)
            
            train_dataframe = self.is_nan_values_exist(train_dataframe)
            test_dataframe = self.is_nan_values_exist(test_dataframe)
            logging.info(f"Cleaned Train dataframe shape: {train_dataframe.shape}")
            logging.info(f"Cleaned Test dataframe shape: {test_dataframe.shape}")

            # Label Integrity Validation (FIFTH CHECK)

            status = self.validate_labels_integrity(train_dataframe)
            if not status:
                raise CustomException("Labels integrity mismatched in training dataframe", sys)
            
            status = self.validate_labels_integrity(test_dataframe)
            if not status:
                raise CustomException("Labels integrity mismatched in test dataframe", sys)

            os.makedirs(os.path.dirname(self.data_validation_config.valid_train_file_path),exist_ok=True)
            os.makedirs(os.path.dirname(self.data_validation_config.valid_test_file_path),exist_ok=True)
            
            train_dataframe.to_csv(self.data_validation_config.valid_train_file_path,index=False,header=True)
            test_dataframe.to_csv(self.data_validation_config.valid_test_file_path,index=False,header=True)
            
            data_validation_artifact = DataValidationArtifact(
                validation_status = status,
                valid_train_file_path = self.data_validation_config.valid_train_file_path,
                valid_test_file_path = self.data_validation_config.valid_test_file_path,
                invalid_train_file_path = self.data_validation_config.invalid_train_file_path,
                invalid_test_file_path = self.data_validation_config.invalid_test_file_path
            )

            return data_validation_artifact
        except Exception as e:
            raise CustomException(e,sys)
        