import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_parquet("data/raw/stock_data.parquet")  # Read the DataFrame from the Parquet file
df = df.sort_values(["ticker","date"])  # Sort the DataFrame by date and ticker

print(df.head(50))  # Print the first few rows of the sorted DataFrame

df["past_5d_return"] = (df.groupby("ticker")["close"].pct_change(5))  # Calculate the past 5-day average closing price for each ticker

print(df[["ticker","date","close","past_5d_return"]].head(50))  # Print the first few rows of the DataFrame with the new column


df["past_10d_return"] = (df.groupby("ticker")["close"].pct_change(-10)/df["close"])-1  # Calculate the past 10-day average closing price for each ticker

print(df[["ticker","date","close","past_10d_return"]].head(50))  # Print the first few rows of the DataFrame with the new column

df["future_10d_return"] = (
    df.groupby("ticker")["close"].shift(-10)
    / df["close"]
    - 1
)

#Clean the missing values and split the data into train and test sets (periods)
df = df.dropna(subset=["past_5d_return", "past_10d_return", "future_10d_return"])  # Drop rows with missing values in the specified columns

train_df = df[df["date"] < "2023-01-01"]  # Create a training set with data before 2023
test_df = df[df["date"] >= "2023-01-01"]  # Create a test set with data from 2023 onwards

features = ["past_5d_return", "past_10d_return", "future_10d_return"]  # Define the features to be used for modeling

X_train = train_df[features]  # Extract the features for the training set
X_test = test_df[features]  # Extract the features for the test set
y_train = train_df["future_10d_return"]  # Extract the target variable for the training set
y_test = test_df["future_10d_return"]  # Extract the target variable for the test set



print("X TRAIN:")
print(X_train.head())

print("\ny TRAIN:")
print(y_train.head())

print("\nTraining observations:", len(X_train))
print("Test observations:", len(X_test))



model = LinearRegression()
model.fit(X_train, y_train)

print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)