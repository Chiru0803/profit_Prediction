
# Profit Prediction Project

## Overview

This project aims to predict the profit of startups based on their spending in various areas using regression algorithms. The dataset used contains information about startup expenditures in R&D, Administration, and Marketing, and the target variable is the profit. The project involves data preprocessing, model training, evaluation, and selection of the best-performing regression model.

## File Structure

- `profit_prediction.py`: The main Python script for data processing, model training, and evaluation.

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- seaborn

You can install the required packages using pip:

```bash
pip install pandas numpy scikit-learn seaborn
```

## Usage

1. **Dataset**: Ensure you have the dataset file `50_Startups.csv` in the same directory as the script. The dataset should contain the following columns:
   - `R&D Spend`
   - `Administration`
   - `Marketing Spend`
   - `Profit`

2. **Run the Script**: Execute the script to train and evaluate the regression models.

   ```bash
   python profit_prediction.py
   ```

## Script Description

1. **Data Loading**: The script loads the dataset from `50_Startups.csv` into a pandas DataFrame.

2. **Feature and Target Separation**: 
   - Features (X): `R&D Spend`, `Administration`, `Marketing Spend`
   - Target (y): `Profit`

3. **Data Splitting**: The dataset is divided into training and testing sets with an 80-20 split.

4. **Model Construction**:
   - Random Forest Regressor
   - Gradient Boosting Regressor

5. **Model Training and Evaluation**:
   - Each model is trained on the training set and evaluated on the test set.
   - Metrics computed include Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared.

6. **Best Model Selection**: The model with the lowest RMSE is selected as the best model.

7. **Output**: The script prints the performance metrics for each model and the best model's prediction results.

## Results

The script outputs the following metrics for each regression model:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R-squared

It also identifies the best model based on RMSE and displays its predictions.

## Example Output

```
Regression Metrics:

Random Forest Regression:
MSE: 12345.67
RMSE: 111.11
MAE: 85.55
R-squared: 0.87

Gradient Boosting Regression:
MSE: 9876.54
RMSE: 99.38
MAE: 77.23
R-squared: 0.89

The best model is: Gradient Boosting Regression
[Predicted values...]
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

 
