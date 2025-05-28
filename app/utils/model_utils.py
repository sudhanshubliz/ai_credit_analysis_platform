import pickle
import os

def load_model(model_path):
    """
    Load a machine learning model from a file.
    Args:
        model_path (str): Path to the model file.
    Returns:
        object: Loaded model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    with open(model_path, "rb") as f:
        return pickle.load(f)

def predict_risk(model, input_data):
    """
    Predict the risk level given input data.
    Args:
        model (object): Trained model.
        input_data (dict): Input feature data.
    Returns:
        tuple: Predicted probability and risk level.
    """
    prob = model.predict_proba([list(input_data.values())])[0][1]
    if prob > 0.7:
        risk_level = "High"
    elif prob > 0.3:
        risk_level = "Medium"
    else:
        risk_level = "Low"
    return prob, risk_level