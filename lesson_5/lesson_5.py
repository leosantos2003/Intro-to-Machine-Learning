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

# Specify model
iowa_model = DecisionTreeRegressor(random_state=1)
# Fit model
iowa_model.fit(train_X, train_y)

# Make validation predictions and calculate mean absolute error
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("\nValidation MAE: {:,.0f}\n".format(val_mae))

# Function to calculate the mean absolute error of a given model
def get_mae(max_leaf_nodes, train_X, val_X, tain_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]

# Loop to find the ideal tree size from cadidate_max_leaf_nodes
for max_leaf_nodes in [5, 25, 50, 100, 250, 500]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d \t Mean absolute error: %d" %(max_leaf_nodes, my_mae))

# Obtain the minimum value of MAE:
# the most accurate model
scores = {
    leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes
    }
best_tree_size = min(scores, key=scores.get)

# Final model with optimal size
final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=0)
final_model.fit(X, y)
final_predictions = final_model.predict(X)

# Print the top few predictions with minimum MAE
print("\nTop few predictions with minimum MAE:\n",final_predictions[:5])
# Print the top few actual prices
print("\nTop few actual prices:\n",y[:5])

print("\n")