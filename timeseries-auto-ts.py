import pandas as pd
import auto_ts as ats
import matplotlib.pyplot as plt


# Load your dataset
data = pd.read_csv('data/test.csv', delimiter=";", decimal=",", parse_dates=['Datum'])

# Load your dataset
# data = pd.read_csv('electricity_load.csv', parse_dates=['timestamp'])

# Check the first few rows to understand the data structure
print(data.head())

# Set the timestamp as the index (important for time series tasks)
data.set_index('Datum', inplace=True)

# Define your target variable (the load column you want to predict)
target_column = 'load'

# Initialize the AutoTS model
model = ats(forecast_length=24,  # Forecast the next 24 time steps (e.g., hours)
                   frequency='H',  # Frequency of the data (e.g., hourly)
                   prediction_interval=0.95)  # Prediction interval (95% confidence)

# Fit the model to your data
model = model.fit(data, date_col='Datum', value_col=target_column)

# Make predictions for the next 24 periods
prediction = model.predict()

# Show the predictions
print(prediction.forecast)

# Optionally, visualize the actual vs. predicted values
plt.figure(figsize=(10, 6))

# Plot the historical data
plt.plot(data.index, data[target_column], label='kWh')

# Plot the predictions
forecast_dates = pd.date_range(start=data.index[-1], periods=25, freq='H')[1:]  # Generate forecast dates
plt.plot(forecast_dates, prediction.forecast, label='Predicted Load', linestyle='--')

plt.title('Electrical Load Prediction')
plt.xlabel('Time')
plt.ylabel('Load')
plt.legend()
plt.show()
