import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

# Training file
file_path = './train.csv'

# Read the file into a DataFrame
home_data = pd.read_csv(file_path)

# Extract features
y = home_data["SalePrice"]
features_names = ['MSSubClass', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']
X = home_data[features_names].dropna()

# Split data into training and validation sets
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

# Initialize model
forest_model = RandomForestRegressor(random_state=1)

# Hyperparameters to try
param_dist = {
    'n_estimators': [int(x) for x in np.linspace(start=100, stop=1000, num=10)],
    # 'max_depth': [int(x) for x in np.linspace(10, 110, num=11)] + [None],
    'max_depth': [50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
}

# Random search through hyperparameters to find best combination
random_search = RandomizedSearchCV(
    estimator = forest_model,
    param_distributions = param_dist,
    n_iter = 100,  # Number of iterations (number of hyperparameters to try)
    cv = 5,  # Number of folds for cross validation
    verbose = 0, # Outputs
    random_state = 42,
    n_jobs = -1  # Use all available processors
)

# Find best model on all data
random_search.fit(train_X, train_y)

# Save best parameters and model
best_params = random_search.best_params_
best_model = random_search.best_estimator_

# Fit default model to training data
forest_model.fit(train_X, train_y)

# Compare optimized model and default model
best_house_prices_preds = best_model.predict(val_X)
house_prices_preds = forest_model.predict(val_X)
print("Best MAE")
print(mean_absolute_error(val_y, best_house_prices_preds))
print("Default MAE")
print(mean_absolute_error(val_y, house_prices_preds))
