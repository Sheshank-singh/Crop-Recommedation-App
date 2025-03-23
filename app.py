import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# Load the trained model and scaler
try:
    with open("crop_model.pkl", "rb") as model_file:
        classifier = pickle.load(model_file)

    with open("scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
except FileNotFoundError:
    raise FileNotFoundError("Ensure 'crop_model.pkl' and 'scaler.pkl' exist in the working directory.")

# Initialize FastAPI app
app = FastAPI()

# Define the input data format
class CropInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

# Home route
@app.get("/")
def home():
    return {"message":  " Hello Welcome to the Crop Recommendation API. Use /predict to get predictions."}

# Prediction API endpoint
@app.post("/predict")
def predict_crop(data: CropInput):
    # Convert input into a NumPy array
    input_data = np.array([[data.N, data.P, data.K, data.temperature, data.humidity, data.ph, data.rainfall]])

    # Scale the input data
    try:
        scaled_input = scaler.transform(input_data)
    except Exception as e:
        return {"error": f"Scaler transformation failed: {str(e)}"}

    # Predict the crop
    try:
        prediction = classifier.predict(scaled_input)[0]
        return {"message": f"This crop ({prediction}) is best for your field!"}
    except Exception as e:
        return {"error": f"Model prediction failed: {str(e)}"}

