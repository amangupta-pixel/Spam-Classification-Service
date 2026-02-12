import os
import sys
import ast
import numpy as np
import pandas as pd
from src.constant.training_pipeline import TARGET_COLUMN
from src.exception.exception import CustomException
from src.logging.logger import logging
from src.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from src.entity.config_entity import DataTransformationConfig
import gensim.downloader as api
from src.utils.main_utils.utils import save_numpy_array_data, save_object

class DataTransformation:
    def __init__(self,data_validation_artifact:DataValidationArtifact,
                 data_transformation_config:DataTransformationConfig):
        
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise CustomException(e,sys)
        
    @staticmethod    
    def read_data(file_path:str) -> pd.DataFrame:
        
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e,sys)
        
    def avgWord2Vec(self,tokens:str,model,vector_size:int):

        try:
            try:
                token_list = ast.literal_eval(tokens)
            except (ValueError, SyntaxError):
                token_list = []
            vectors = [model[word] for word in token_list if word in model]
            if len(vectors) == 0:
                return np.zeros(vector_size)
            return np.mean(vectors,axis=0)
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        
        try:
            
            os.makedirs(os.path.dirname(self.data_validation_artifact.valid_train_file_path),exist_ok=True)
            os.makedirs(os.path.dirname(self.data_validation_artifact.valid_test_file_path),exist_ok=True)

            train_df = self.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = self.read_data(self.data_validation_artifact.valid_test_file_path)

            X_train = train_df["text"].values
            y_train = train_df[TARGET_COLUMN].values

            X_test = test_df["text"].values
            y_test = test_df[TARGET_COLUMN].values

            # Using Google Pretrained Word2Vec Model
            word2vec = api.load("word2vec-google-news-300")
            vector_size = word2vec.vector_size

            X_train_vec = np.array([self.avgWord2Vec(tokens, word2vec, vector_size) for tokens in X_train])

            X_test_vec = np.array([self.avgWord2Vec(tokens, word2vec, vector_size) for tokens in X_test])

            # Combining features and target
            train_arr = np.c_[X_train_vec,y_train]
            test_arr = np.c_[X_test_vec,y_test]

            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path,train_arr)
            
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path,test_arr)

            save_object(self.data_transformation_config.transformed_object_file_path,word2vec)

            save_object("final_model/preprocessor.pkl",word2vec)

            data_transformation_artifact = DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path
            )

            return data_transformation_artifact

        except Exception as e:
            raise CustomException(e,sys)
        