import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Training file
file_path = './train.csv'

# Read the file into a DataFrame
home_data = pd.read_csv(file_path)

# Extract features
y = home_data["SalePrice"]
features_names = ["LotArea","YearBuilt","1stFlrSF","2ndFlrSF","FullBath","BedroomAbvGr","TotRmsAbvGrd"]
X = home_data[features_names]

# Split data into training and validation sets
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

# Initialize model
forest_model = RandomForestRegressor(random_state=1)

# Fit model to training data
forest_model.fit(train_X, train_y)

# Predict prices for 5 homes
print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(forest_model.predict(X.head()))
print("Real prices were")
print(y.head())

# Show MAE
predicted_home_prices= forest_model.predict(val_X)
print("MAE is")
print(mean_absolute_error(val_y, predicted_home_prices))
