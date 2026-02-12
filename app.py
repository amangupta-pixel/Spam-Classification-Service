import os
import sys
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
mongo_db_url = os.getenv("MONGO_DB_URL")
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import Response 
from starlette.responses import RedirectResponse
from uvicorn import run as app_run
import pymongo
from src.exception.exception import CustomException
from src.pipeline.training_pipeline import TrainingPipeline
from src.constant.training_pipeline import DATA_INGESTION_DATABASE_NAME, DATA_INGESTION_COLLECTION_NAME
from src.utils.main_utils.utils import load_object
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


client = pymongo.MongoClient(mongo_db_url)
collection = client[DATA_INGESTION_DATABASE_NAME][DATA_INGESTION_COLLECTION_NAME]


class TextIn(BaseModel):
    text: str

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"]
)

from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory="./templates")

@app.get("/",tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def train_route():
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response("Training is successfull")
    except Exception as e:
        raise CustomException(e,sys)
    
@app.post("/predict")
async def predict_route(request:Request,file:UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)

        text_column = "text"

        if text_column not in df.columns:
            raise CustomException(f"Column '{text_column}' not found in uploaded file")
        
        spam_model = load_object("final_model/model.pkl")

        predictions = spam_model.predict(df[text_column])

        label_map = {
            0: "Ham",
            1: "Spam"
        }

        df["prediction"] = [label_map[p] for p in predictions]

        os.makedirs("prediction_output",exist_ok=True)
        df.to_csv("prediction_output/output.csv",index=False,header=True)

        table_html = df.to_html(classes="table table striped")
        return templates.TemplateResponse(
            "table.html",
            {"request":request,"table":table_html}
        )

    except Exception as e:
        raise CustomException(e,sys)
    
@app.post("/predict-text")
async def predict_text_route(text_input:TextIn):
    try:
        df = pd.DataFrame([text_input.text],columns=["text"])

        spam_model = load_object("final_model/model.pkl")

        predictions = spam_model.predict(df["text"])
        probabilities = spam_model.predict_proba(df["text"])

        label_map = {
            0: "Ham",
            1: "Spam"
        }

        prediction_label = label_map[predictions[0]]
        # Get the probability of the predicted class
        confidence_score = probabilities[0][int(predictions[0])]

        return {
                "input_text": text_input.text,
                "Result": prediction_label,
                "Confidence_Score": f"{confidence_score:.2%}"
                }
        

    except Exception as e:
        raise CustomException(e,sys)
    
if __name__=="__main__":
    app_run(app,host="127.0.0.1",port=8000)