import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

iowa_file_path = 'train.csv'
home_data = pd.read_csv(iowa_file_path)

# Create target object and call it y
y = home_data.SalePrice
# Create X
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[features]

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Specify Model
iowa_model = DecisionTreeRegressor(random_state=1)
# Fit Model
iowa_model.fit(train_X, train_y)

# Make validation predictions and calculate mean absolute error
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("\nValidation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))

# Using best value for max_leaf_nodes
iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
iowa_model.fit(train_X, train_y)
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("\nValidation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))

from sklearn.ensemble import RandomForestRegressor

# Define the model and set random_state to 1
rf_model = RandomForestRegressor(random_state=1)

# Fit model
rf_model.fit(train_X, train_y)

# Calculate the mean absolute error of Random Forest model on the validation data
rf_preds = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(val_y, rf_preds)

print("\nValidation MAE for Random Forest Model: {}\n".format(rf_val_mae))