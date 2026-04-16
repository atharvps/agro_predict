"""
Script to train the existing pure Python Random Forest model 
and save it as a pickle file for production deployment.
"""
import pickle
import math

# Import your existing code from the pipeline file
from crop_pipeline import load_and_preprocess_data, time_based_split, RandomForestRegressor

def build_and_save_model():
    print("🚀 Step 1: Loading and preprocessing data...")
    dataset_path = "dataset.csv"
    data, encoders = load_and_preprocess_data(dataset_path)
    
    if len(data) == 0:
        print("❌ Error: Dataset empty. Cannot train.")
        return
        
    print("✂️ Step 2: Splitting data chronologically...")
    X_train, y_train, X_test, y_test = time_based_split(data, test_ratio=0.2)
    
    print("🌲 Step 3: Training the Custom Random Forest...")
    num_features = len(X_train[0])
    max_feat = max(1, int(math.sqrt(num_features)))
    
    rf_model = RandomForestRegressor(
        n_trees=8,               
        max_depth=12,            
        min_samples_split=50,    
        max_features=max_feat,   
        num_thresholds=10        
    )
    rf_model.fit(X_train, y_train)
 
    
    print("💾 Step 4: Saving model and encoders to model.pkl...")
    # We save a dictionary containing BOTH the model and the encoding dictionaries
    export_data = {
        "model": rf_model,
        "encoders": encoders
    }
    
    with open("model.pkl", "wb") as file:
        pickle.dump(export_data, file)
        
    print("✅ Model successfully saved! You can now start the FastAPI server.")

if __name__ == "__main__":
    build_and_save_model()