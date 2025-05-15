# Exp.no: 10 IMPLEMENTATION OF SARIMA MODEL

## AIM:
To implement SARIMA model using Python.

## PROCEDURE:
1. Explore the dataset
2. Check for stationarity
3. Determine SARIMA model parameters using auto_arima
4. Fit the SARIMA model
5. Make predictions
6. Evaluate model predictions

## PROGRAM:
```
Developed by: Naveenkumar M
Reg No: 212224230182

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('/content/AirPassengers (1).csv')

# Rename columns if needed (adjust based on actual header)
data.columns = ['Month', 'Passengers']

# Convert 'Month' column to datetime and set as index
data['Month'] = pd.to_datetime(data['Month'])
data.set_index('Month', inplace=True)
data = data.sort_index()

# Plot the Passenger Time Series
plt.figure(figsize=(10, 4))
plt.plot(data.index, data['Passengers'], label='Monthly Air Passengers')
plt.xlabel('Date')
plt.ylabel('Number of Passengers')
plt.title('Monthly Air Passengers Time Series')
plt.legend()
plt.tight_layout()
plt.show()

# Check for stationarity using Dickey-Fuller test
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')

check_stationarity(data['Passengers'])

# Plot ACF and PACF
plot_acf(data['Passengers'])
plt.show()
plot_pacf(data['Passengers'])
plt.show()

# Split data into train and test sets
train_size = int(len(data) * 0.8)
train, test = data['Passengers'][:train_size], data['Passengers'][train_size:]

# Fit the SARIMA model (initial guess; you can tune these parameters)
sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit(disp=False)

# Forecast
predictions = sarima_result.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

# Calculate RMSE
mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print('RMSE:', rmse)

# Plot actual vs predicted values
plt.figure(figsize=(10, 4))
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, color='red', label='Predicted')
plt.xlabel('Date')
plt.ylabel('Number of Passengers')
plt.title('SARIMA Model Predictions')
plt.legend()
plt.tight_layout()
plt.show()
```
## OUTPUT: 
![image](https://github.com/user-attachments/assets/68ce6ead-c75c-47d8-8f69-c273b8c50b76)
![image](https://github.com/user-attachments/assets/cfbca22c-5a0f-467d-a00b-3590c32c61d6)
![image](https://github.com/user-attachments/assets/d61ffa84-4dd2-4b50-b678-c3084f17c867)
![image](https://github.com/user-attachments/assets/851461af-a558-4c17-b3b4-38526d3ad96b)

## RESULT:
Thus the program run successfully based on the SARIMA model.
