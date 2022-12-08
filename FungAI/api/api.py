from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from FungAI.interface.main import evaluate, pred
from PIL import Image
import io

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

@app.post("/predict")
async def predict(image: UploadFile = File(...)) :

    _img = await image.read()
    img = Image.open(io.BytesIO(_img))
    prediction = pred(new_image = img)

    return prediction
