import pickle
import pandas as pd

# Sample dataset
data = {
    "Loan Amount": [5000, 15000, 25000, 10000, 50000, 30000],
    "Annual Revenue": [100000, 200000, 300000, 120000, 400000, 350000],
    "Debt-to-Income Ratio": [0.3, 0.5, 0.6, 0.4, 0.7, 0.5],
    "Credit Score": [700, 650, 600, 750, 580, 680],
    "Number of Open Accounts": [5, 10, 15, 8, 20, 12],
}

# Convert to DataFrame
sample_data = pd.DataFrame(data)

# Save the dataset to a pickle file
with open("data/sample_data.pkl", "wb") as f:
    pickle.dump(sample_data, f)

print("Sample dataset saved to data/sample_data.pkl")