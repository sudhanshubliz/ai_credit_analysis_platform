import shap
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np  # Added NumPy import
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

    try:
        with open(explainer_path, "rb") as f:
            explainer = pickle.load(f)
        with open(sample_data_path, "rb") as f:
            sample_data = pickle.load(f)
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except EOFError as e:
        raise ValueError(f"Error loading file: {e}. The file may be empty or corrupted. Please verify the file integrity.") from e
    except FileNotFoundError as e:
        raise ValueError(f"Missing file: {e}. Ensure the required file exists at the specified path.") from e

    return explainer, sample_data, model

# Function to calculate SHAP values and generate a SHAP waterfall plot
def get_shap_values_and_save(explainer, input_df, output_index=0, save_path="shap_waterfall_plot.png"):
    """
    Calculate SHAP values for the input data and save the SHAP waterfall plot as an image.
    Args:
        explainer: The SHAP explainer object.
        input_df: The input data as a pandas DataFrame.
        output_index: The index of the output to explain (default is 0 for the first output).
        save_path: The file path to save the SHAP waterfall plot image.
    Returns:
        shap_values: Computed SHAP values.
        fig: Matplotlib figure object for the SHAP waterfall plot of the selected output.
    Raises:
        ValueError: If the SHAP explainer or input data is invalid.
    """
    try:
        # Ensure explainer is valid
        if explainer is None:
            raise ValueError("SHAP explainer is not defined. Please provide a valid SHAP explainer.")

        # Ensure input data is a DataFrame
        if not isinstance(input_df, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")

        # Calculate SHAP values
        shap_values = explainer(input_df)

        # Validate SHAP values
        if shap_values is None or not isinstance(shap_values, shap._explanation.Explanation):
            raise ValueError("SHAP values must be of type shap._explanation.Explanation.")

        # Debugging outputs
        print("SHAP Values Type:", type(shap_values))
        print("SHAP Values Shape:", shap_values.shape)
        print("Base Values Type:", type(shap_values.base_values))

        # Extract SHAP values and base values for the selected output index
        explanation = shap.Explanation(
            values=shap_values.values[0, :, output_index],  # SHAP values for the first sample and selected output
            base_values=shap_values.base_values[0, output_index],  # Base value for the first sample and selected output
            data=input_df.iloc[0]  # Input data for the first sample
        )

        # Generate waterfall plot
        ax = shap.plots.waterfall(explanation, show=False)

        # Extract the figure from the Axes object
        fig = ax.figure

        # Save the figure
        fig.savefig(save_path, bbox_inches="tight")

        return shap_values, fig

    except Exception as e:
        raise ValueError(f"An error occurred while calculating SHAP values or generating the waterfall plot: {e}")

# Streamlit app
st.title("Credit Risk Analysis Platform")
st.write("This app predicts credit risk and provides SHAP explanations for predictions.")

# Example input form
st.subheader("Enter the following details:")
loan_amount = st.number_input("Loan Amount", min_value=1000, max_value=1000000, step=1000)
annual_revenue = st.number_input("Annual Revenue", min_value=10000, max_value=10000000, step=10000)

# Additional features
debt_to_income_ratio = st.number_input("Debt-to-Income Ratio", min_value=0.0, max_value=1.0, step=0.01)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, step=1)
num_open_accounts = st.number_input("Number of Open Accounts", min_value=0, max_value=50, step=1)

# Convert input data to DataFrame
input_df = pd.DataFrame({
    "Loan Amount": [loan_amount],
    "Annual Revenue": [annual_revenue],
    "Debt-to-Income Ratio": [debt_to_income_ratio],
    "Credit Score": [credit_score],
    "Number of Open Accounts": [num_open_accounts]
    # Add more features as needed
})

if st.button("Predict Risk"):
    try:
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
        try:
            save_path = "shap_waterfall_plot.png"
            shap_values, fig = get_shap_values_and_save(explainer, input_df, output_index=0, save_path=save_path)
            st.subheader("🔍 SHAP Explanation")
            st.write("The chart below shows which features most influenced this prediction.")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error generating SHAP waterfall plot: {e}")

        # Generate PDF report
        if st.button("📄 Generate PDF Report"):
            data_dict = {
                "Loan Amount": loan_amount,
                "Annual Revenue": annual_revenue,
                "Debt-to-Income Ratio": debt_to_income_ratio,
                "Credit Score": credit_score,
                "Number of Open Accounts": num_open_accounts,
                "Predicted Risk": f"{pred_prob:.2%}",
                "Risk Level": risk_level,
                "Recommendation": recommendation,
                "Analyst": st.session_state.get('username', 'Anonymous')
            }
            pdf_path = generate_pdf_report(data_dict)
            with open(pdf_path, "rb") as f:
                st.download_button("⬇️ Download PDF Report", f, file_name="pdf_report.pdf")
    except ValueError as e:
        st.error(f"Error: {e}")