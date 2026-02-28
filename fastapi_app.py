from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
from utils import get_prediction

app = FastAPI(title="Dog Breed Classifier API")

# load model once at startup
model = tf.keras.models.load_model("model/20250218-124402-full-dataset-mobilenetv2-Adam.keras")

@app.get("/")
def home():
    return {"message": "Dog Breed Classifier API running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    label, confidence = get_prediction(model, image_bytes)
    return {
        "predicted_breed": label,
        "confidence": confidence
    }