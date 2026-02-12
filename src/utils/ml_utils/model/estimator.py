import sys
import os
import re
import numpy as np
from nltk.stem import WordNetLemmatizer
from src.exception.exception import CustomException
from nltk.corpus import stopwords

class SpamModel:
    def __init__(self,preprocessor,model):
        try:
            self.preprocessor = preprocessor
            self.model = model
            self.lemmatizer = WordNetLemmatizer()
        except Exception as e:
            raise CustomException(e,sys)
        
    def _preprocess_data(self,text:str) -> list:
        try:
            text = re.sub("[^a-zA-Z]"," ", text)
            text = text.lower().split()
            text = [self.lemmatizer.lemmatize(word) for word in text if word not in stopwords.words('english')]
            return text
        except Exception as e:
            raise CustomException(e,sys)

    def _avgWord2Vec(self,tokens:list):
        try:
            vector_size = self.preprocessor.vector_size
            vectors = [self.preprocessor[word] for word in tokens if word in self.preprocessor]
            if len(vectors) == 0:
                return np.zeros(vector_size)
            return np.mean(vectors,axis=0)
        except Exception as e:
            raise CustomException(e,sys)

    def predict(self,x):
        try:
            tokens_list = x.apply(self._preprocess_data)

            vectorized_x = np.array([self._avgWord2Vec(tokens) for tokens in tokens_list])

            y_hat = self.model.predict(vectorized_x)
            return y_hat
        except Exception as e:
            raise CustomException(e,sys)

    def predict_proba(self, x):
        try:
            tokens_list = x.apply(self._preprocess_data)

            vectorized_x = np.array([self._avgWord2Vec(tokens) for tokens in tokens_list])
            return self.model.predict_proba(vectorized_x)
        except Exception as e:
            raise CustomException(e, sys)
