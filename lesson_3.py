import pandas as pd

# Path of the file to read
iowa_file_path = 'train.csv'

# Read the file into the variable home_data
home_data = pd.read_csv(iowa_file_path)

# Print summary statistics
print("\nFile summary statistics:\n")
print(home_data.describe())

# Set prediction target as y
y = home_data.SalePrice

# Create a list of features for the DataFrame
feature_names = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

# Create DataFrame called X holding the preditive features
X = home_data[feature_names]

# Print description from X
print("\nDataFrame statistics:\n")
print(X.describe())

# Print the top few lines from X
print("\nDataFrame top few lines:\n")
print(X.head())

from sklearn.tree import DecisionTreeRegressor

# Create a DecisionTreeRegressor and save it iowa_model
iowa_model = DecisionTreeRegressor(random_state=1)

# Fit the model
iowa_model.fit(X, y)

# Make predictions
predictions = iowa_model.predict(X)

print("\nModel predictions for SalePrice:\n")
print(predictions[:5])
print("\nActual values for SalePrice:\n")
print(y.head())
print("\n")