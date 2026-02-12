import os
import sys
import pandas as pd
import numpy as np
import pymongo
from dotenv import load_dotenv
load_dotenv()
MONGO_DB_URL = os.getenv("MONGO_DB_URL")
import re # Regular Expression
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from typing import Tuple
from src.exception.exception import CustomException
from src.logging.logger import logging
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from sklearn.model_selection import train_test_split

class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig):
        try:
            self.config = data_ingestion_config
            self.stopwords = stopwords.words('english')
            self.lemmatizer = WordNetLemmatizer()
        except Exception as e:
            raise CustomException(e,sys)

    def export_collection_as_dataframe(self) -> pd.DataFrame:
        
        try:

            logging.info("Connecting to MongoDB for Data Ingestion")
            client = pymongo.MongoClient(MONGO_DB_URL)
            collection = client[self.config.database_name][self.config.collection_name]
            df = pd.DataFrame(list(collection.find()))
            if "_id" in df.columns:
                df.drop(columns="_id",inplace=True)

            logging.info(f"Data fetched successfully with shape: {df.shape}")
            return df
    
        except Exception as e:
            raise CustomException(e,sys)
    
    def export_dataframe_as_feature_store(self,dataframe:pd.DataFrame):
        try:
            feature_store_file_path = self.config.feature_store_file_path

            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)

            dataframe.to_csv(feature_store_file_path,index=False,header=True)
            return dataframe
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def preprocess_data(self,text:str) -> list: # text -> str content under each df['message'] 
        try:
            
            text = re.sub("[^a-zA-Z]"," ", text)
            text = text.lower().split()
            text = [self.lemmatizer.lemmatize(word) for word in text if word not in self.stopwords]
                
            # logging.info("Data preprocessing done")
            return text # List[str]
        except Exception as e:
            raise CustomException(e,sys)
        
    def prepare_target(self,dataframe:pd.DataFrame) -> np.ndarray:
        try:
            # Explicitly map 'spam' to 1 and 'ham' to 0
            y = dataframe['label'].map({'ham': 0, 'spam': 1}).astype(int).values
            logging.info("Target variable encoded")
            return y

        except Exception as e:
            raise CustomException(e,sys)
        

    def split_data(self,X:list,y:np.ndarray) -> Tuple[list,list,np.ndarray,np.ndarray]:
        try:
            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=self.config.train_test_split_ratio, random_state=42)

            logging.info("Train-Test Split Completed")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            raise CustomException(e,sys)
        
    def save_data(self,X_train,X_test,y_train,y_test) -> None:
        try:
            os.makedirs(os.path.dirname(self.config.training_file_path),exist_ok=True)
            os.makedirs(os.path.dirname(self.config.testing_file_path),exist_ok=True)

            train_df = pd.DataFrame({"text":X_train,"label":y_train})
            test_df = pd.DataFrame({"text":X_test,"label":y_test})

            train_df.to_csv(self.config.training_file_path,index=False,header=True)
            test_df.to_csv(self.config.testing_file_path,index=False,header=True)

            logging.info("Train and Test datasets saved successfully")
        except Exception as e:
            raise CustomException(e,sys)

    
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            df = self.export_collection_as_dataframe()
            df = self.export_dataframe_as_feature_store(df)
            df["tokens"] = df["message"].apply(self.preprocess_data)
            y = self.prepare_target(df)
            X_train, X_test, y_train, y_test = self.split_data(df["tokens"],y)
            self.save_data(X_train,X_test,y_train,y_test)

            return DataIngestionArtifact(
                trained_file_path=self.config.training_file_path,
                test_file_path=self.config.testing_file_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)
