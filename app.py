import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np
import json

model = load_model("pet_classifier.h5")

# Class names dosyasını yükle
with open("class_names.json", "r") as f:
    class_names = json.load(f)

st.title("Pet Sınıflandırıcı")
img_file = st.file_uploader("Resim Yükle", type=["jpg", "jpeg", "png"])

if img_file:
    img = Image.open(img_file).convert("RGB")
    st.image(img, caption="Yüklenen Resim", use_column_width=True)

    # Tahmin yap
    arr = np.array(img.resize((224, 224))) / 255.
    arr = np.expand_dims(arr, 0)
    preds = model.predict(arr)
    class_idx = np.argmax(preds[0])
    prob = float(preds[0][class_idx])

    if prob > 0.5:
        st.success(f"**Tahmini Sınıf:** {class_names[class_idx]}")
        st.info(f"**Güven Oranı:** {prob:.2%}")
    else:
        st.error("Bulunamadı - sınıfa atama yapılamadı.")
