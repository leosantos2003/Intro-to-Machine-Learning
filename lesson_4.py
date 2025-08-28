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

from sklearn.model_selection import train_test_split

# Split up the data:
# exclude some data from the model-building process and
# use it to test the model's accuracy on data it hasn't seen before
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Specify the model
iowa_model = DecisionTreeRegressor(random_state=1)

# Fit iowa_model with the training data
iowa_model.fit(train_X, train_y)

# Predict with all validation observations
val_predictions = iowa_model.predict(val_X)

# Print the top few validation predictions
print("\nTop few validation predictions after data split:\n",val_predictions[:5])
# Print the top few actual prices from validation data
print("\nTop few actual prices from validation data:\n",val_y[:5])

from sklearn.metrics import mean_absolute_error

# Calculate the mean absolute error in validation data
val_mae = mean_absolute_error(val_y, val_predictions)

print("\nMean absolute error:", val_mae, "\n")
