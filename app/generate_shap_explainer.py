import shap
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from reports.pdf_report import generate_pdf_report

# Paths
data_path = "data/training_data.csv"
explainer_path = "models/shap_explainer.pkl"
model_path = "models/trained_model.pkl"

# Load training data
training_data = pd.read_csv(data_path)

# Features and target
X = training_data.drop(columns=["Defaulted"])
y = training_data["Defaulted"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the trained model
with open(model_path, "wb") as f:
    pickle.dump(model, f)

# Create SHAP explainer
explainer = shap.Explainer(model, X_train)

# Save SHAP explainer
with open(explainer_path, "wb") as f:
    pickle.dump(explainer, f)

print(f"SHAP explainer saved to {explainer_path}")
print(f"Trained model saved to {model_path}")