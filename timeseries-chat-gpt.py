## pip install numpy pandas matplotlib scikit-learn keras tensorflow statsmodels

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from statsmodels.tsa.arima.model import ARIMA

# Load the dataset (example: AirPassengers)
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
df = pd.read_csv(url, header=0, parse_dates=[0], index_col=0, squeeze=True)

# Check the data
print(df.head())

# Rescale the data between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df.values.reshape(-1, 1))

# Create a function to process time series data into features and labels
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# Reshape the data into a format suitable for LSTM input
time_step = 12  # Use 12 months (1 year) as input sequence
X, y = create_dataset(scaled_data, time_step)

# Reshape X to be 3D as required by LSTM [samples, time steps, features]
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split data into training and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))  # Dropout for regularization
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))  # Output layer

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32)

# Predict on test data
predictions = model.predict(X_test)

# Inverse transform the predictions and true values to get the original scale
predictions = scaler.inverse_transform(predictions)
y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot the results
plt.figure(figsize=(10,6))
plt.plot(df.index[-len(y_test_original):], y_test_original, label='True Values')
plt.plot(df.index[-len(y_test_original):], predictions, label='Predictions')
plt.legend()
plt.show()

from sklearn.metrics import mean_squared_error, mean_absolute_error

# Calculate MSE and MAE
mse = mean_squared_error(y_test_original, predictions)
mae = mean_absolute_error(y_test_original, predictions)

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')

# Fit ARIMA model
model = ARIMA(df, order=(5, 1, 0))  # ARIMA(5,1,0) is an example, adjust order accordingly
model_fit = model.fit()

# Make predictions
predictions_arima = model_fit.forecast(steps=12)  # Predict next 12 months

# Plot the predictions
plt.plot(df.index, df.values, label='Historical Data')
plt.plot(pd.date_range(df.index[-1], periods=13, freq='M')[1:], predictions_arima, label='ARIMA Predictions')
plt.legend()
plt.show()

