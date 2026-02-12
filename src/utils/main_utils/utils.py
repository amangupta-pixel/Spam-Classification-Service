import os
import sys
import yaml
import pickle
import pandas as pd
import numpy as np
from src.exception.exception import CustomException
from src.logging.logger import logging

def read_yaml_file(file_path:str) -> dict:
    try:
        with open(file_path,"rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise CustomException(e,sys)
    
def save_numpy_array_data(file_path:str,array:np.ndarray):
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,"wb") as file_obj:
            np.save(file_obj,array) 
    except Exception as e:
        raise CustomException(e,sys)
    
def save_object(file_path:str,obj:object):
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)

        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
def load_numpy_array_data(file_path:str) -> np.ndarray:
    try:
        with open(file_path,"rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path:str) -> object:
    try:
        with open(file_path,"rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
