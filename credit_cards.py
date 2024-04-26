import pandas as pd

# Load the dataset
file_path = 'CC GENERAL.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(data.head())
print(data.columns.tolist())
