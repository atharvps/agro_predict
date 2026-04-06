"""
Complete Machine Learning Pipeline for Crop Yield Prediction
- No Sklearn, Pandas, or Numpy
- Pure Python & Matplotlib
- Random Forest Regressor from scratch optimized for 350k rows
"""

import os
import subprocess
import csv
import math
import random
import matplotlib.pyplot as plt

# =====================================================================
# 1. FULL GOOGLE COLAB SETUP CODE (KAGGLE DATASET DOWNLOAD)
# =====================================================================
def setup_colab_environment():
    """
    Automates the Colab environment setup, Kaggle authentication, and dataset extraction.
    Run this block directly in Google Colab.
    """
    print("--- [1/8] Starting Google Colab & Kaggle Setup ---")
    
    # Install Kaggle via pip
    subprocess.run(["pip", "install", "-q", "kaggle"])
    
    kaggle_dir = os.path.expanduser("~/.kaggle")
    kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")
    
    # Check if Kaggle token is available
    if not os.path.exists(kaggle_json_path):
        if os.path.exists("kaggle.json"):
            os.makedirs(kaggle_dir, exist_ok=True)
            os.rename("kaggle.json", kaggle_json_path)
            os.chmod(kaggle_json_path, 0o600)
            print("✅ 'kaggle.json' successfully configured.")
        else:
            print("⚠️ WARNING: 'kaggle.json' not found in current directory.")
            print("Please upload your kaggle.json API token to the Colab workspace to download the dataset.")
            print("To proceed using an existing local 'dataset.csv', skipping download...")
            return
    
    dataset_name = "atharvpratapsingh/crop-production-india-with-weather-features"
    zip_file = "crop-production-india-with-weather-features.zip"
    
    # Download and extract the dataset using Kaggle CLI
    if not os.path.exists("crop_yield.csv") and not os.path.exists("dataset.csv"):
        print(f"Downloading dataset: {dataset_name} ...")
        subprocess.run(["kaggle", "datasets", "download", "-d", dataset_name])
        
        if os.path.exists(zip_file):
            print("Extracting dataset...")
            subprocess.run(["unzip", "-o", "-q", zip_file])
            print("✅ Dataset downloaded and extracted successfully.")
        else:
            print("❌ Failed to download dataset. Check your kaggle.json validity.")

# =====================================================================
# 2. CORRECTED DATA PREPROCESSING FUNCTION
# =====================================================================
# =====================================================================
# 2A. OUTLIER HANDLING HELPERS
# =====================================================================
OUTLIER_CAPS = {
    # Based on your dataset's distribution; tune later if needed
    'Area': (0.0, 200000.0),
    'Rainfall': (0.0, 1800.0),
    'Temperature': (0.0, 40.0),
    'Yield': (0.0, 120.0)   # clip extreme target outliers too
}

def clip_value(value, col_name):
    low, high = OUTLIER_CAPS[col_name]
    if value < low:
        return low
    if value > high:
        return high
    return value
def load_and_preprocess_data(filepath):
    """
    Automatically detects columns, parses CSV, cleans absent values, avoids target leakage,
    and applies label encoding using dictionaries.
    """
    print("--- [2/8] Preprocessing Data & Applying Label Encoding ---")
    dataset = []

    # Nested dictionaries for Label Encoding
    encoders = {
        'State': {},
        'District': {},
        'Crop': {},
        'Season': {}
    }

    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            raw_headers = next(reader)

            # Map clean column names to their indices to handle whitespace issues
            col_map = {h.strip(): i for i, h in enumerate(raw_headers)}

            # Verify required columns
            required_cols = ['State', 'District', 'Crop', 'Season', 'Area', 'Rainfall', 'Temperature', 'Crop_Year', 'Yield']
            for col in required_cols:
                if col not in col_map and col + ' ' not in col_map:  # Backup handling for trailing spaces
                    pass

            # Safely fetch index (handles trailing spaces automatically)
            def get_idx(name):
                return col_map.get(name, col_map.get(name + ' ', -1))

            for row in reader:
                try:
                    # 1. Target Extraction (Yield)
                    yield_str = row[get_idx('Yield')].strip()
                    if not yield_str:
                        continue
                    y_val = clip_value(float(yield_str), 'Yield')

                    # 2. Continuous Input Features
                    area_str = row[get_idx('Area')].strip()
                    rain_str = row[get_idx('Rainfall')].strip()
                    temp_str = row[get_idx('Temperature')].strip()
                    year_str = row[get_idx('Crop_Year')].strip()

                    area = clip_value(float(area_str) if area_str else 0.0, 'Area')
                    rainfall = clip_value(float(rain_str) if rain_str else 0.0, 'Rainfall')
                    temperature = clip_value(float(temp_str) if temp_str else 0.0, 'Temperature')
                    crop_year = int(year_str)

                    # 3. Categorical Strings
                    state = row[get_idx('State')].strip()
                    district = row[get_idx('District')].strip()
                    crop = row[get_idx('Crop')].strip()
                    season = row[get_idx('Season')].strip()

                    # 4. Dynamic Label Encoding
                    if state not in encoders['State']: encoders['State'][state] = len(encoders['State'])
                    if district not in encoders['District']: encoders['District'][district] = len(encoders['District'])
                    if crop not in encoders['Crop']: encoders['Crop'][crop] = len(encoders['Crop'])
                    if season not in encoders['Season']: encoders['Season'][season] = len(encoders['Season'])

                    # 5. Feature Vector (NOTE: 'Production' is explicitly excluded to prevent target leakage)
                    features = [
                        encoders['State'][state],        # 0
                        encoders['District'][district],  # 1
                        encoders['Crop'][crop],          # 2
                        encoders['Season'][season],      # 3
                        area,                            # 4
                        rainfall,                        # 5
                        temperature,                     # 6
                        crop_year                        # 7
                    ]

                    dataset.append((features, y_val))

                except (ValueError, IndexError):
                    continue  # Bypass malformed rows

    except FileNotFoundError:
        print(f"❌ Error: Could not find '{filepath}'. Ensure dataset is unzipped.")
        exit(1)

    return dataset, encoders


# =====================================================================
# 3. CORRECTED TRAIN-TEST SPLIT FUNCTION
# =====================================================================
def time_based_split(dataset, test_ratio=0.2):
    """
    Strictly chronological train-test split based on Crop_Year (index 7).
    Prevents temporal data leakage.
    """
    print("--- [3/8] Executing Chronological Train-Test Split ---")
    
    # Sort dataset ascending by Crop_Year
    sorted_data = sorted(dataset, key=lambda x: x[0][7])
    
    split_index = int(len(sorted_data) * (1 - test_ratio))
    
    train_data = sorted_data[:split_index]
    test_data = sorted_data[split_index:]
    
    X_train = [row[0] for row in train_data]
    y_train = [row[1] for row in train_data]
    
    X_test = [row[0] for row in test_data]
    y_test = [row[1] for row in test_data]
    
    return X_train, y_train, X_test, y_test

# =====================================================================
# 4. FULL DECISION TREE IMPLEMENTATION
# =====================================================================
class DecisionTreeNode:
    def __init__(self, depth):
        self.depth = depth
        self.is_leaf = False
        self.value = None
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None

class DecisionTreeRegressor:
    def __init__(self, max_depth=10, min_samples_split=5, max_features=None, num_thresholds=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.num_thresholds = num_thresholds
        self.root = None

    def fit(self, X, y):
        if self.max_features is None:
            self.max_features = len(X[0])
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        num_samples = len(y)
        node = DecisionTreeNode(depth)
        node.value = sum(y) / num_samples if num_samples > 0 else 0
        
        # Stopping criteria
        if depth >= self.max_depth or num_samples < self.min_samples_split:
            node.is_leaf = True
            return node
            
        # Feature Sampling (Random Subspace Method)
        num_total_features = len(X[0])
        features = random.sample(range(num_total_features), self.max_features)
        
        best_feature = None
        best_threshold = None
        min_mse = float('inf')
        
        # Optmized splitting evaluation
        for feature_idx in features:
            feature_values = [row[feature_idx] for row in X]
            f_min, f_max = min(feature_values), max(feature_values)
            
            if f_min == f_max:
                continue
                
            # Threshold sampling
            thresholds = [random.uniform(f_min, f_max) for _ in range(self.num_thresholds)]
            
            for thresh in thresholds:
                # SSE Optimization for speed - avoids creating intermediate arrays
                left_count, right_count = 0, 0
                left_sum, right_sum = 0.0, 0.0
                left_sum_sq, right_sum_sq = 0.0, 0.0
                
                for i in range(num_samples):
                    val = feature_values[i]
                    target = y[i]
                    if val <= thresh:
                        left_count += 1
                        left_sum += target
                        left_sum_sq += target * target
                    else:
                        right_count += 1
                        right_sum += target
                        right_sum_sq += target * target
                        
                if left_count == 0 or right_count == 0:
                    continue
                    
                # Calculate Sum of Squared Errors (SSE)
                left_sse = left_sum_sq - (left_sum * left_sum / left_count)
                right_sse = right_sum_sq - (right_sum * right_sum / right_count)
                
                # Weighted MSE
                weighted_mse = (left_sse + right_sse) / num_samples
                
                if weighted_mse < min_mse:
                    min_mse = weighted_mse
                    best_feature = feature_idx
                    best_threshold = thresh

        if best_feature is None:
            node.is_leaf = True
            return node
            
        # Execute the best geometric partition
        left_X, left_y, right_X, right_y = [], [], [], []
        for i in range(num_samples):
            if X[i][best_feature] <= best_threshold:
                left_X.append(X[i])
                left_y.append(y[i])
            else:
                right_X.append(X[i])
                right_y.append(y[i])
                
        node.feature_index = best_feature
        node.threshold = best_threshold
        node.left = self._build_tree(left_X, left_y, depth + 1)
        node.right = self._build_tree(right_X, right_y, depth + 1)
        
        return node

    def predict(self, X):
        return [self._predict_row(row, self.root) for row in X]

    def _predict_row(self, row, node):
        if node.is_leaf:
            return node.value
        if row[node.feature_index] <= node.threshold:
            return self._predict_row(row, node.left)
        else:
            return self._predict_row(row, node.right)

# =====================================================================
# 5. FULL RANDOM FOREST IMPLEMENTATION
# =====================================================================
class RandomForestRegressor:
    def __init__(self, n_trees=5, max_depth=12, min_samples_split=10, max_features=None, num_thresholds=10):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.num_thresholds = num_thresholds
        self.trees = []

    def fit(self, X, y):
        print(f"--- [5/8] Training Random Forest ({self.n_trees} Trees) ---")
        self.trees = []
        n_samples = len(X)
        
        for i in range(self.n_trees):
            print(f"   -> Growing Tree {i+1}/{self.n_trees} using Bootstrap Sampling...")
            
            # Fast Bootstrap Sampling
            indices = [random.randint(0, n_samples - 1) for _ in range(n_samples)]
            X_sample = [X[idx] for idx in indices]
            y_sample = [y[idx] for idx in indices]
                
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                num_thresholds=self.num_thresholds
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        print("--- Executing Ensemble Prediction ---")
        # Collect predictions from all trees
        tree_preds = [tree.predict(X) for tree in self.trees]
        y_pred = []
        
        # Average the predictions
        for i in range(len(X)):
            avg_pred = sum(preds[i] for preds in tree_preds) / self.n_trees
            y_pred.append(avg_pred)
            
        return y_pred

# =====================================================================
# 6. METRICS FUNCTIONS
# =====================================================================
def calculate_metrics(y_true, y_pred):
    print("--- [6/8] Calculating Evaluation Metrics ---")
    n = len(y_true)
    if n == 0:
        return 0.0, 0.0, 0.0
        
    mae = sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / n
    mse = sum((yt - yp)**2 for yt, yp in zip(y_true, y_pred)) / n
    rmse = math.sqrt(mse)
    
    mean_y = sum(y_true) / n
    sst = sum((yt - mean_y)**2 for yt in y_true)
    
    r2 = 1.0 - (mse * n / sst) if sst != 0 else 0.0
    
    return mae, rmse, r2

# =====================================================================
# 7. VISUALIZATION CODE
# =====================================================================
def plot_visualizations(y_true, y_pred):
    print("--- [7/8] Generating Diagnostic Plots ---")
    residuals = [yt - yp for yt, yp in zip(y_true, y_pred)]
    
    plt.figure(figsize=(18, 5))
    
    # Visual 1: Actual vs Predicted
    plt.subplot(1, 3, 1)
    plt.scatter(y_true, y_pred, alpha=0.3, color='blue', s=10)
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
    plt.title('Actual vs Predicted Yield')
    plt.xlabel('Actual Historical Yield')
    plt.ylabel('Model Predicted Yield')
    
    # Visual 2: Residual Histogram
    plt.subplot(1, 3, 2)
    plt.hist(residuals, bins=50, color='green', alpha=0.7, edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--')
    plt.title('Residual Error Distribution')
    plt.xlabel('Error Magnitude (Residual)')
    plt.ylabel('Absolute Frequency')
    
    # Visual 3: Residuals vs Predicted
    plt.subplot(1, 3, 3)
    plt.scatter(y_pred, residuals, alpha=0.3, color='purple', s=10)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.title('Residuals vs Predicted')
    plt.xlabel('Predicted Yield')
    plt.ylabel('Residual Deviations')
    
    plt.tight_layout()
    plt.savefig('model_diagnostics.png')
    print("✅ Visualizations saved as 'model_diagnostics.png'.")
    plt.show()

# =====================================================================
# 8. MAIN EXECUTION BLOCK
# =====================================================================
if __name__ == "__main__":
    # 1. Run environment setup
    #setup_colab_environment()
    
    # Ensure correct filename mapping (Colab unzips generally extract the exact CSV name)
    # Check current directory for csv files

    #csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    #dataset_path = csv_files[0] if csv_files else 'dataset.csv'
    
    dataset_path = "dataset.csv"

    if not os.path.exists(dataset_path):
        raise FileNotFoundError("❌ dataset.csv not found in current directory")

    print(f"✅ Using dataset file: {dataset_path}")


    # 2. Preprocess
    data, encoders = load_and_preprocess_data(dataset_path)
    print(f"   -> Valid records loaded: {len(data)}")
    
    if len(data) == 0:
        print("❌ Dataset empty or parsing failed. Aborting pipeline.")
        exit(1)
    
    # 3. Train-Test Split
    X_train, y_train, X_test, y_test = time_based_split(data, test_ratio=0.2)
    print(f"   -> Training set: {len(X_train)} samples")
    print(f"   -> Testing set:  {len(X_test)} samples")
    
    # 4. Determine Random Subspace features (sqrt of total features)
    num_features = len(X_train[0])
    max_feat = max(1, int(math.sqrt(num_features)))
    
    # 5. Initialize Model
    # Note: Reduced n_trees to 5 for demonstration so the pure python script completes 
    # within minutes rather than hours on 350k rows. Increase for higher accuracy.
    rf_model = RandomForestRegressor(
        n_trees=5,               
        max_depth=12,            
        min_samples_split=15,    
        max_features=max_feat,   
        num_thresholds=10        
    )
    
    # 6. Train Model
    rf_model.fit(X_train, y_train)
    
    # 7. Predictions
    y_pred = rf_model.predict(X_test)
    
    # 8. Metrics
    mae, rmse, r2 = calculate_metrics(y_test, y_pred)
    
    print("\n" + "=" * 50)
    print("🚀 PIPELINE EXECUTION COMPLETE - RESULTS")
    print("=" * 50)
    print(f"Mean Absolute Error (MAE)  : {mae:.4f}")
    print(f"Root Mean Squared Error    : {rmse:.4f}")
    print(f"Coefficient of Determ (R²): {r2:.4f}")
    print("=" * 50 + "\n")
    
    # 9. Visualization
    plot_visualizations(y_test, y_pred)
    print("--- [8/8] Pipeline gracefully terminated ---")