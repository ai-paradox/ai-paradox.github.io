import pandas as pd
from datetime import datetime
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Training file
file_path = './train.csv'

# Read the file into a DataFrame
home_data = pd.read_csv(file_path)

# Shows statistics extracted from data
statistics = home_data.describe()
avg_lot_size = home_data["LotArea"].mean()
avg_price = home_data["SalePrice"].mean()
newest_home_age = datetime.now().year - home_data["YearBuilt"].max()
print("Average price is")
print(avg_price)

# Extract features
y = home_data["SalePrice"]
features_names = ["LotArea","YearBuilt","1stFlrSF","2ndFlrSF","FullBath","BedroomAbvGr","TotRmsAbvGrd"]
X = home_data[features_names]

# Split data into training and validation sets
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

# Mean average error
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

# Finds best tree size between candidates
best_tree_size = -1
min_mae = 10000000
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500, 1000, 2000]
for max_leaf_nodes in candidate_max_leaf_nodes:
    mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    if mae < min_mae:
        best_tree_size = max_leaf_nodes
        min_mae = mae
print("Best tree size is")
print(best_tree_size)

# Initialize model with best tree size
price_model = DecisionTreeRegressor(max_leaf_nodes = best_tree_size, random_state = 1)

# Fit model with all data
price_model.fit(X, y)

# Predict prices for 5 homes
print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(price_model.predict(X.head()))
print("Real prices were")
print(y.head())

# Show MAE
predicted_home_prices= price_model.predict(X)
print("MAE is")
print(mean_absolute_error(y, predicted_home_prices))
