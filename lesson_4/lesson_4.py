import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import lesson_4_graphics

# Terminal: cd .\lesson_4\

iowa_file_path = '../train.csv'
home_data = pd.read_csv(iowa_file_path)

y = home_data.SalePrice
feature_columns = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[feature_columns]

# --- 1. Model trained with full DataSet ---

# Specify model and fit model
iowa_model = DecisionTreeRegressor()
iowa_model.fit(X, y)

# First in-sample predictions and actual values
in_sample_preds = iowa_model.predict(X.head())
in_sample_actuals = y.head().tolist()

# --- 2. Model trained with split DataSet ---

# Split up the data:
# exclude some data from the model-building process and
# use it to test the model's accuracy on data it hasn't seen before
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Specify the model and fit with the training data
iowa_model_split = DecisionTreeRegressor(random_state=1)
iowa_model_split.fit(train_X, train_y)

# Predict with all validation observations
val_predictions = iowa_model_split.predict(val_X.head())
val_actuals = val_y.head().tolist()

# Calculate the mean absolute error in validation data
val_mae = mean_absolute_error(val_y, iowa_model_split.predict(val_X))
print("\nMean absolute error:", val_mae)

print("\nCreating comparison graphics...")

# Chama a função do outro arquivo para criar o primeiro gráfico
lesson_4_graphics.plot_comparison_bars(
    predictions=in_sample_preds,
    actual_values=in_sample_actuals,
    title='Comparison: Prediction vs. Actual Value (training data)',
    file_name='comparison_in_sample.png'
)

# Chama a mesma função para criar o segundo gráfico
lesson_4_graphics.plot_comparison_bars(
    predictions=val_predictions,
    actual_values=val_actuals,
    title='Comparison: Prediction vs. Actual Value (validation data)',
    file_name='comparison_validation.png'
)

print("\n")