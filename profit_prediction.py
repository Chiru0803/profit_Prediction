import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import seaborn as sb

# Load your dataset (replace 'dataset.csv' with your actual dataset filename)
data = pd.read_csv('50_Startups.csv')

# Split the dataset into features (X) and target (y)
X = data[['R&D Spend', 'Administration', 'Marketing Spend']]
y = data['Profit']

# Step ii) Divide the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step i) Construct Different Regression Algorithms
models = {

    'Random Forest Regression': RandomForestRegressor(),
    'Gradient Boosting Regression': GradientBoostingRegressor()
}

# Train and evaluate each model
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results[name] = {'RMSE': rmse,'MSE':mse, 'MAE': mae, 'R-squared': r2}

# Step iv) Choose the best model
best_model = min(results, key=lambda x: results[x]['RMSE'])

print("Regression Metrics:\n")
for name, metrics in results.items():
    print(f"{name}:")
    print(f"MSE: {metrics['MSE']}")
    print(f"RMSE: {metrics['RMSE']}")
    print(f"MAE: {metrics['MAE']}")
    print(f"R-squared: {metrics['R-squared']}")
    print()

print(f"The best modelx is: {best_model}")
print(y_pred )