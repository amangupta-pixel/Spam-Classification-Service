import os
import sys
import numpy as np
import pandas as pd
from src.exception.exception import CustomException
from src.logging.logger import logging
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact)

from src.utils.main_utils.utils import (
    load_numpy_array_data, 
    load_object, 
    save_object,
    )
from src.utils.ml_utils.metric.classification_metric import get_classification_score
from src.utils.ml_utils.model.estimator import SpamModel
from sklearn.svm import SVC

class ModelTrainer:
    def __init__(self,data_transformation_artifact:DataTransformationArtifact,
                 model_trainer_config:ModelTrainerConfig):
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config
        except Exception as e:
            raise CustomException(e,sys)
        
    def train_model(self,X_train,X_test,y_train,y_test):

        try:

            model = SVC(kernel="linear", probability=True)
            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)

            train_metric = get_classification_score(y_true=y_train,y_pred=y_train_pred)

            y_test_pred = model.predict(X_test)
            test_metric = get_classification_score(y_true=y_test,y_pred=y_test_pred)

            logging.info(f"Training accuracy: {train_metric.accuracy_score}")
            logging.info(f"Test accuracy: {test_metric.accuracy_score}")

            preprocessor = load_object(self.data_transformation_artifact.transformed_object_file_path)
            
            spam_model = SpamModel(preprocessor,model)

            save_object(self.model_trainer_config.trained_model_file_path,spam_model)

            save_object("final_model/model.pkl",spam_model)

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=train_metric,
                test_metric_artifact=test_metric
            )

            return model_trainer_artifact

        except Exception as e:
            raise CustomException(e,sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            os.makedirs(os.path.dirname(self.data_transformation_artifact.transformed_train_file_path),exist_ok=True)
            os.makedirs(os.path.dirname(self.data_transformation_artifact.transformed_test_file_path),exist_ok=True)

            train_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)

            X_train, y_train = train_arr[:,:-1], train_arr[:,-1]
            X_test, y_test = test_arr[:,:-1], test_arr[:,-1]

            return self.train_model(X_train,X_test,y_train,y_test)

        except Exception as e:
            raise CustomException(e,sys)