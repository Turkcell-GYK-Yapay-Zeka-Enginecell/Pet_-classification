from fastapi import FastAPI, File, UploadFile
from keras.models import load_model
from PIL import Image
import numpy as np
import io
import json

app = FastAPI()
model = load_model("pet_classifier.h5")

# Class names dosyasını yükle
with open("class_names.json", "r") as f:
    class_names = json.load(f)

def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img) / 255.
    arr = np.expand_dims(arr, 0)
    return arr

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    arr = preprocess(image_bytes)
    preds = model.predict(arr)
    class_idx = np.argmax(preds[0])
    prob = float(preds[0][class_idx])
    if prob > 0.5:
        return {"class": class_names[class_idx], "probability": prob}
    else:
        return {"error": "Bulunamadı - sınıfa atama yapılamadı"}
