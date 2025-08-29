import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import lesson_5_graphics

# Terminal: cd .\lesson_5\

iowa_file_path = '../train.csv'
home_data = pd.read_csv(iowa_file_path)

# Create target object and call it y
y = home_data.SalePrice
# Create X
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[features]
# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Function to calculate the mean absolute error of a given model
def get_mae(max_leaf_nodes, train_X, val_X, tain_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
# Obtain the minimum value of MAE:
# the most accurate model
scores = {
    leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes
    }
# Prints results
for leaf_size, mae in scores.items():
    print(f"Max leaf nodes: {leaf_size:d} \t\t Mean absolute error: {mae:,.0f}")

best_tree_size = min(scores, key=scores.get)
print(f"\nBest tree size found: {best_tree_size} leaves")

# --- Graphic 1: MAE vs. Tree Size ---
lesson_5_graphics.plot_mae_vs_leaf_nodes(scores)

# --- Final Model and Predictions ---
# Trains the final model with the best size and with all the data
final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=0)
final_model.fit(X, y)

# Gets the 5 first predictions and real values for the comparison graphic
top_predictions = final_model.predict(X.head())
top_actuals = y.head().tolist()

# --- Graphic 2: Predictions Comparison ---
lesson_5_graphics.plot_prediction_comparison(
    predictions=top_predictions,
    actuals=top_actuals,
    title=f'Comparison: Prediction vs. Real Value (Model with {best_tree_size} Leaves)',
    file_name='final_model_comparison.png'
)