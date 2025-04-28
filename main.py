from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from preprocess import preprocess_image

app = FastAPI()

# Загрузка модели и label binarizer
model = load_model("best_model_resnet_like.keras")
label_binarizer = joblib.load("label_binarizer.pkl")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = preprocess_image(image_bytes)

        # Предсказание
        probs = model.predict(image)[0]  # Вероятности
        predicted_index = np.argmax(probs)
        predicted_label = label_binarizer.classes_[predicted_index]

        return JSONResponse({
            "predicted_class": str(predicted_label),
            "probabilities": {
                str(label): float(prob) 
                for label, prob in zip(label_binarizer.classes_, probs)
            }
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
