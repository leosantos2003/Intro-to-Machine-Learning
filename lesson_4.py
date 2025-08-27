import pandas as pd
from sklearn.tree import DecisionTreeRegressor

iowa_file_path = 'train.csv'
home_data = pd.read_csv(iowa_file_path)

y = home_data.SalePrice
feature_columns = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[feature_columns]

# Specify model
iowa_model = DecisionTreeRegressor()
# Fit model
iowa_model.fit(X, y)

print("\nFirst in-sample predictions:", iowa_model.predict(X.head()))
print("\nActual target values for those homes:", y.head().tolist())
print("\n")

from sklearn.model_selection import train_test_split

# Split up the data
# 
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
