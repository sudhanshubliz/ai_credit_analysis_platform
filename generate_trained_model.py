import shap
import streamlit as st
import matplotlib.pyplot as plt
import pickle
from reports.pdf_report import generate_pdf_report

# Function to load SHAP explainer and model
def load_shap_explainer():
    """
    Load the SHAP explainer, sample data, and model.
    Returns:
        explainer: The SHAP explainer object.
        sample_data: The sample data used for SHAP explanations.
        model: The trained model for predictions.
    """
    # Example paths; replace with actual file paths
    explainer_path = "models/shap_explainer.pkl"
    sample_data_path = "data/sample_data.pkl"
    model_path = "models/trained_model.pkl"

    with open(explainer_path, "rb") as f:
        explainer = pickle.load(f)
    with open(sample_data_path, "rb") as f:
        sample_data = pickle.load(f)
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return explainer, sample_data, model

# Function to calculate SHAP values
def get_shap_values(explainer, input_df):
    """
    Calculate SHAP values for the input data using the SHAP explainer.
    Args:
        explainer: The SHAP explainer object.
        input_df: The input data as a DataFrame.
    Returns:
        shap_values: Computed SHAP values.
    """
    shap_values = explainer(input_df)
    return shap_values

# Streamlit app
st.title("Credit Risk Analysis Platform")
st.write("This app predicts credit risk and provides SHAP explanations for predictions.")

# Example input form
loan_amount = st.number_input("Loan Amount", min_value=1000, max_value=1000000, step=1000)
annual_revenue = st.number_input("Annual Revenue", min_value=10000, max_value=10000000, step=10000)
input_df = {"Loan Amount": [loan_amount], "Annual Revenue": [annual_revenue]}  # Convert to DataFrame as needed

if st.button("Predict Risk"):
    # Load explainer and model
    explainer, sample_data, shap_model = load_shap_explainer()
    pred_prob = shap_model.predict_proba(input_df)[0][1]  # Assuming binary classification

    # Risk level determination
    risk_level = "High" if pred_prob > 0.7 else "Medium" if pred_prob > 0.3 else "Low"

    # Recommendation
    recommendation = "Consider approving with caution." if risk_level == "Medium" else \
                      "High risk! Recommend rejection." if risk_level == "High" else \
                      "Low risk! Recommend approval."

    st.success(f"Predicted Risk: {pred_prob:.2%} | Risk Level: {risk_level}")
    st.write(f"Recommendation: {recommendation}")

    # SHAP explanation
    shap_values = get_shap_values(explainer, input_df)
    st.subheader("🔍 SHAP Explanation")
    st.write("The chart below shows which features most influenced this prediction.")
    fig = shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

    # Generate PDF report
    if st.button("📄 Generate PDF Report"):
        data_dict = {
            "Loan Amount": loan_amount,
            "Annual Revenue": annual_revenue,
            "Predicted Risk": f"{pred_prob:.2%}",
            "Risk Level": risk_level,
            "Recommendation": recommendation,
            "Analyst": st.session_state['username']
        }
        pdf_path = generate_pdf_report(data_dict)
        with open(pdf_path, "rb") as f:
            st.download_button("⬇️ Download PDF Report", f, file_name="pdf_report.pdf")