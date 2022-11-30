from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from FungAI.interface.main import evaluate


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
def evaluate() :
    metrics = evaluate()
    return dict(Loss = float(metrics["loss"]), Accuracy = float(metrics["accuracy"]))

@app.get("/predict")
def predict() :
    pred = pred()

    return {"genus" : pred}
