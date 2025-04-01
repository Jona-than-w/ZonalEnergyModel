import pandas as pd

# Load the dataset
file_path = "PUB_DemandZonal_2025_v72.csv"
df = pd.read_csv(file_path)

# Display the first few rows to understand its structure
df.head()
