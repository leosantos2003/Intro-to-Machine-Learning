import pandas as pd

# Path of the file to read
iowa_file_path = 'train.csv'

# Read the file into the variable home_data
home_data = pd.read_csv(iowa_file_path)

# Print summary statistics
print(home_data.describe())