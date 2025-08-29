import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import lesson_6_graphics

# Terminal: cd .\lesson_6\

iowa_file_path = '../train.csv'
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

# --- 1. Simple model without specifying max_leaf_nodes ---

# Make validation predictions and calculate mean absolute error
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("\nValidation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))

# --- 2. Model with optimized max_leaf_nodes ---

# Using best value for max_leaf_nodes
iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
iowa_model.fit(train_X, train_y)
val_predictions = iowa_model.predict(val_X)
val_mae_best = mean_absolute_error(val_predictions, val_y)
print("\nValidation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae_best))

# --- 3. Model with Random Forest ---

# Define the model and set random_state to 1
rf_model = RandomForestRegressor(random_state=1)
# Fit model
rf_model.fit(train_X, train_y)
# Calculate the mean absolute error of Random Forest model on the validation data
rf_preds = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(val_y, rf_preds)

print("\nValidation MAE for Random Forest Model: {:,.0f}\n".format(rf_val_mae))

# --- 4. Graphic comparison generation ---

model_names = [
    'Decision Tree (standard)',
    'Decision Tree (max_leaf_nodes=100)',
    'Random Forest'
]
mae_scores = [
    val_mae,
    val_mae_best,
    rf_val_mae
]

# Graphic
lesson_6_graphics.plot_mae_comparison(
    mae_scores=mae_scores,
    model_names=model_names,
    title='Mean Absolute Error (MAE) Comparison between Models',
    file_name='mae_models_comparison.png'
)