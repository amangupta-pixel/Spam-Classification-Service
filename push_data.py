# Data Load into MongoDB Database using ETL pipeline

import os
import sys
import pandas as pd
import pymongo
import json
from dotenv import load_dotenv
from src.exception.exception import CustomException
load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")

class SpamHamDataExtract:
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise CustomException(e,sys)
        
    def csv_to_json_transformer(self,file_path):
        try:
            df = pd.read_csv(file_path)

            records = list(json.loads(df.T.to_json()).values())
            return records
        except Exception as e:
            raise CustomException(e,sys)
        
    def load_into_mongodb(self,records,database,collection):
        try:
            mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            db = mongo_client[database]
            coll = db[collection]

            coll.delete_many({}) # Deletes existing data to avoid duplicates
            result = coll.insert_many(records)

            return len(result.inserted_ids)
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    FILE_PATH = "data/sms_spam_dataset.csv"
    DATABASE = "TESTAI"
    COLLECTION = "SpamData"
    obj = SpamHamDataExtract()
    records = obj.csv_to_json_transformer(file_path=FILE_PATH)
    no_of_records = obj.load_into_mongodb(records,DATABASE,COLLECTION)
    print(no_of_records)
    
