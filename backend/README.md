Crop Yield Prediction API (Pure Python ML)

This is a production-ready FastAPI backend serving a custom-built Random Forest Regressor. Zero external ML libraries (scikit-learn, pandas, numpy) were used in building the predictive engine.

1. Local Setup

Clone this repository and navigate to the folder.

Install dependencies:

pip install -r requirements.txt


Generate the model binary by running the pipeline export script:

python export_model.py


Start the API Server:

uvicorn app:app --host 0.0.0.0 --port 8000 --reload


2. API Endpoints

GET / : Health check.

GET /docs : Interactive Swagger UI (Test the API in your browser).

POST /predict : Submit crop features and get a yield prediction.