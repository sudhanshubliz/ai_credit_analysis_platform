import shap
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Paths
model_path = "models/trained_model.pkl"
data_path = "data/training_data.csv"  # Replace with actual training data path
explainer_path = "models/shap_explainer.pkl"

# Load the trained model
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Load training data
training_data = pd.read_csv(data_path)  # Ensure this matches your training data format
feature_columns = training_data.columns[:-1]  # Adjust based on your dataset
X_train = training_data[feature_columns]

# Create SHAP explainer
explainer = shap.Explainer(model, X_train)

# Save SHAP explainer to file
with open(explainer_path, "wb") as f:
    pickle.dump(explainer, f)

print(f"SHAP explainer saved successfully to {explainer_path}")