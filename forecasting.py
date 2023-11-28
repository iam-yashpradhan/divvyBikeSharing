import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose  # Seasonal decomposition of time series
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # ACF and PACF plots
from itertools import product  # Cartesian product of input iterables
import statsmodels.api as sm

data01 = pd.read_csv('/Users/yashpradhan/Desktop/UNC Charlotte/Sem Fall 2023/Big Data Analytics/Divvy Project/divvyData/202210-divvy-tripdata.csv')
data02 = pd.read_csv('/Users/yashpradhan/Desktop/UNC Charlotte/Sem Fall 2023/Big Data Analytics/Divvy Project/divvyData/202211-divvy-tripdata.csv')
data03 = pd.read_csv('/Users/yashpradhan/Desktop/UNC Charlotte/Sem Fall 2023/Big Data Analytics/Divvy Project/divvyData/202212-divvy-tripdata.csv')
data04 = pd.read_csv('/Users/yashpradhan/Desktop/UNC Charlotte/Sem Fall 2023/Big Data Analytics/Divvy Project/divvyData/202301-divvy-tripdata.csv')
data05 = pd.read_csv('/Users/yashpradhan/Desktop/UNC Charlotte/Sem Fall 2023/Big Data Analytics/Divvy Project/divvyData/202302-divvy-tripdata.csv')
data06 = pd.read_csv('/Users/yashpradhan/Desktop/UNC Charlotte/Sem Fall 2023/Big Data Analytics/Divvy Project/divvyData/202303-divvy-tripdata.csv')
data07 = pd.read_csv('/Users/yashpradhan/Desktop/UNC Charlotte/Sem Fall 2023/Big Data Analytics/Divvy Project/divvyData/202304-divvy-tripdata.csv')
data08 = pd.read_csv('/Users/yashpradhan/Desktop/UNC Charlotte/Sem Fall 2023/Big Data Analytics/Divvy Project/divvyData/202304-divvy-tripdata.csv')
data09 = pd.read_csv('/Users/yashpradhan/Desktop/UNC Charlotte/Sem Fall 2023/Big Data Analytics/Divvy Project/divvyData/202306-divvy-tripdata.csv')
data10 = pd.read_csv('/Users/yashpradhan/Desktop/UNC Charlotte/Sem Fall 2023/Big Data Analytics/Divvy Project/divvyData/202307-divvy-tripdata.csv')
data11 = pd.read_csv('/Users/yashpradhan/Desktop/UNC Charlotte/Sem Fall 2023/Big Data Analytics/Divvy Project/divvyData/202308-divvy-tripdata.csv')
data12 = pd.read_csv('/Users/yashpradhan/Desktop/UNC Charlotte/Sem Fall 2023/Big Data Analytics/Divvy Project/divvyData/202309-divvy-tripdata.csv')

dataframes = [data01, data02, data03, data04, data05,
              data06, data07, data08, data09, data10, data11, data12]
data = pd.concat(dataframes, ignore_index=True)
data.info()

data['started_at'] = pd.to_datetime(data['started_at'])
data['ended_at'] = pd.to_datetime(data['ended_at'])
data['duration'] = data['ended_at'] - data['started_at']
data['duration'] = pd.to_timedelta(data['duration'])
data['duration'] = data['duration'].dt.total_seconds()

ridesData = data.groupby(data['started_at'].dt.date).size().reset_index(name='count')

plt.figure(figsize=(10, 6))
ax = sns.lineplot(data=ridesData, x='started_at', y='count')
ax.set_title('Distribution of Rides Over Time')
ax.set_xlabel('Date')
ax.set_ylabel('Number of Rides')
plt.tight_layout()
plt.show()

# Convert the 'started_at' column to the index
ridesData.set_index('started_at', inplace=True)
# Resample the dataframe to daily frequency
ridesData = ridesData.asfreq('D')

plt.figure(figsize=(10, 6))
plt.plot(ridesData.index, ridesData['count'])
plt.xlabel('Date')
plt.ylabel('Count')
plt.show()

# Compute rolling statistics
rolling_mean = ridesData['count'].rolling(window=12).mean()
rolling_std = ridesData['count'].rolling(window=12).std()

# Plot the original data and rolling statistics
plt.figure(figsize=(10, 6))
plt.plot(ridesData.index, ridesData['count'], label='Original Data')
plt.plot(ridesData.index, rolling_mean, label='Rolling Mean', color='red')
plt.plot(ridesData.index, rolling_std, label='Rolling Std', color='green')
plt.xlabel('Date')
plt.ylabel('Count')
plt.title('Rolling Statistics')
plt.legend()
plt.show()

ridesData.isna().sum()
ridesData = ridesData.dropna()
# Stationary Test
result = adfuller(ridesData['count'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])
print('Critical Values:', result[4])

ridesData['first_diff'] = ridesData['count'] - ridesData['count'].shift(1)
plt.figure(figsize=(10, 6))
plt.plot(ridesData.index, ridesData['count'], label='Original Data')
plt.plot(ridesData.index, ridesData['first_diff'], label='Differenced Data', color='red')
plt.xlabel('Date')
plt.ylabel('Count')
plt.legend()
plt.grid(False)
plt.show()

# Stationary Test
result01 = adfuller(ridesData['first_diff'])
print('ADF Statistic:', result01[0])
print('p-value:', result01[1])
print('Critical Values:', result01[4])

result = seasonal_decompose(ridesData['count'], model='additive', period=12)

# Plot the original data, trend, seasonal, and residual components
plt.figure(figsize=(12, 8))
plt.subplot(4, 1, 1)
plt.plot(ridesData.index, ridesData['count'])
plt.title('Original Data')
plt.subplot(4, 1, 2)
plt.plot(ridesData.index, result.trend)
plt.title('Trend Component')
plt.subplot(4, 1, 3)
plt.plot(ridesData.index, result.seasonal)
plt.title('Seasonal Component')
plt.subplot(4, 1, 4)
plt.plot(ridesData.index, result.resid)
plt.title('Residual Component')
plt.tight_layout()
plt.show()

residuals = result.resid.dropna()
adf_result = adfuller(residuals)
# Print ADF test results
print('ADF Statistic:', adf_result[0])
print('p-value:', adf_result[1])
print('Critical Values:', adf_result[4])

# Plot ACF and PACF
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plot_acf(ridesData['first_diff'].dropna(), lags=40, ax=plt.gca(), title='Autocorrelation Function (ACF)')
plt.subplot(2, 1, 2)
plot_pacf(ridesData['first_diff'].dropna(), lags=40, method='ywm', ax=plt.gca(), title='Partial Autocorrelation Function (PACF)')
plt.tight_layout()
plt.show()

# HyperParameter Optimization
# Define the parameter combinations to search over
p_values = [0, 1, 2]
d_values = [0, 1]
q_values = [0, 1, 2]
seasonal_p_values = [0, 1]
seasonal_d_values = [0, 1]
seasonal_q_values = [0, 1]
seasonal_m_values = [12]

# Initialize variables to store the best model and AIC score
best_model = None
best_aic = float('inf')

# Iterate through and fit a seaosonal ARIMA model with all combinations of parameters
for p, d, q, P, D, Q, m in product(p_values, d_values, q_values, seasonal_p_values, seasonal_d_values,
                                   seasonal_q_values, seasonal_m_values):
    try:
        model = sm.tsa.SARIMAX(ridesData['count'], order=(p, d, q), seasonal_order=(P, D, Q, m))
        results = model.fit()

        aic = results.aic

        if aic < best_aic:
            best_aic = aic
            best_model = model
            best_params = (p, d, q, P, D, Q, m)

    except Exception as e:
        continue

# Print the optimal parameter values
print('Optimal Parameters:', best_params)
print('AIC Score:', best_aic)

# Fit the best model with the optimal parameters
finalModel = sm.tsa.SARIMAX(ridesData['count'], order=best_params[:3], seasonal_order=best_params[3:])
finalResults = finalModel.fit()
# Print the summary of the final model
print(finalResults.summary())
finalResults.plot_diagnostics(figsize=(15, 12))
plt.show()

# Forecasting
# Get predictions for 60 future time points using the fitted model (with 95% confidence level)
forecastSteps = 60
startDate = ridesData.index[0]
endDate = ridesData.index[-1] + pd.Timedelta(days=forecastSteps - 1)
# Ensure endDate is within the range of the DataFrame index
if endDate not in ridesData.index:
    endDate = ridesData.index[-1]
forecast = finalResults.get_prediction(
    start=startDate,
    end=endDate,
    alpha=0.05)

# Extract the forecasted values and confidence intervals
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()
# Add the forecasted values to the dataframe
ridesData['sarimax_pred'] = forecast_mean
# Add the upper and lower limit columns for the confidence intervals
ridesData['sarimax_ci_upper'] = forecast_ci['upper count']
ridesData['sarimax_ci_lower'] = forecast_ci['lower count']
# Print the dataframe with the SARIMAX predictions and confidence intervals
print(ridesData)


# Plot the original time series, predicted values and the 95% confidence interval
plt.figure(figsize=(10, 6))
plt.plot(ridesData.index, ridesData['count'], label='Original Time Series', color='blue')
plt.plot(forecast_mean.index, forecast_mean, label='Predicted Values', color='red')
plt.fill_between(forecast_ci.index, forecast_ci['lower count'],
                 forecast_ci['upper count'], color='pink', alpha=0.3,
                 label='95% Confidence Interval')
plt.xlabel('Date')
plt.ylabel('Count')
plt.title('SARIMAX Predicted Values with 95% Confidence Interval')
plt.legend()
plt.show()