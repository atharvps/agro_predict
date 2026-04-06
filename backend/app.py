from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any
import pickle
import os

# ⚠️ CRITICAL: We MUST import the classes from your pipeline into this namespace.
# Without this, pickle.load() will fail because it won't recognize the custom objects.
from crop_pipeline import RandomForestRegressor, DecisionTreeRegressor, DecisionTreeNode

# Initialize FastAPI App
app = FastAPI(
    title="Crop Yield Prediction API",
    description="Production API for pure Python Random Forest Crop Yield Model",
    version="1.0.0"
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to hold model and encoders
MODEL_DATA: Dict[str, Any] = {}

# Define strictly typed JSON request schema
class CropPredictionRequest(BaseModel):
    State: str = Field(..., example="West Bengal")
    District: str = Field(..., example="PURULIA")
    Crop: str = Field(..., example="Wheat")
    Season: str = Field(..., example="Rabi")
    Area: float = Field(..., gt=0, example=2979.0, description="Area in Hectares")
    Rainfall: float = Field(..., ge=0, example=186.34, description="Rainfall in mm")
    Temperature: float = Field(..., example=13.17, description="Temperature in Celsius")
    Crop_Year: int = Field(..., gt=1900, example=2000)

@app.on_event("startup")
def load_model():
    """Loads the pre-trained model and encoders into memory when the server starts."""
    global MODEL_DATA
    model_path = "model.pkl"
    
    if not os.path.exists(model_path):
        raise RuntimeError(f"❌ '{model_path}' not found! Run export_model.py first.")
        
    print("🔄 Loading Custom Random Forest Model into memory...")
    try:
        with open(model_path, "rb") as file:
            MODEL_DATA = pickle.load(file)
        print("✅ Model loaded successfully!")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load model: {str(e)}")

@app.get("/")
def health_check():
    return {"status": "healthy", "message": "Crop Yield API is running."}

@app.post("/predict")
def predict_yield(request: CropPredictionRequest):
    """Predicts crop yield based on JSON input parameters."""
    try:
        rf_model = MODEL_DATA.get("model")
        encoders = MODEL_DATA.get("encoders")
        
        if not rf_model or not encoders:
            raise HTTPException(status_code=500, detail="Model is not loaded properly.")

        # --- SAFE ENCODING FALLBACK ---
        # .get(key, 0) ensures that if a user passes a new/unknown State or Crop, 
        # it defaults to 0 instead of crashing the server with a KeyError.
        state_idx = encoders['State'].get(request.State.strip(), 0)
        district_idx = encoders['District'].get(request.District.strip(), 0)
        crop_idx = encoders['Crop'].get(request.Crop.strip(), 0)
        season_idx = encoders['Season'].get(request.Season.strip(), 0)
        
        # Format the exact feature vector structure expected by the Custom RF model
        features = [
            state_idx,
            district_idx,
            crop_idx,
            season_idx,
            request.Area,
            request.Rainfall,
            request.Temperature,
            request.Crop_Year
        ]
        
        # Execute prediction (Model expects a 2D array [row1, row2...], so we wrap in [])
        prediction_result = rf_model.predict([features])[0]
        
        # Prevent mathematically impossible negative yields
        final_yield = max(0.0, prediction_result)
        
        return {
            "predicted_yield_t_ha": round(final_yield, 4),
            "estimated_production_tonnes": round(final_yield * request.Area, 2),
            "status": "success"
        }
        
    except Exception as e:
        # Catch any unexpected algorithmic errors
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")