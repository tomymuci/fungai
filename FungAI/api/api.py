from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from FungAI.interface.main import evaluate, pred
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def root() :
    return {"Is it working ?" : True}

@app.get("/evaluate")
def eval() :
    metrics = evaluate()
    return dict(Loss = float(metrics["loss"]), Accuracy = float(metrics["Accuracy"]))

@app.get("/predict")
def predict(image = None) :

    prediction = pred(new_image=image)

    return prediction
