# Time-Series-Forecasting-NTPC-Stock-Data-
## Contents
### Overview :
This project describes different time series and machine learning forecasting models applied to a real stock close price dataset. For this project we will start with a general idea of the stock price, including dataset analysis. Followed by a general description and analysis of the dataset, our objective is to apply different forecasting predictive models for “S&P500” stock daily close price. The models will be evaluated, analyzed and compared, following the main course project directions. The data will be prepared to predict the next 30 days’ close price from today.
### Objective :
Historically, there has been a continuous interest in trying to analyze market tendencies, behavior and random reactions. This continuous concern to understand what happens before it really happens motivates us to continue with this study. Some great market traders and economists says that is almost impossible to predict stock returns or prices referring to, independence between each other, the past movements or trends cannot be used to predict future values, explained by random walk theory, skewness, kurtosis and big random component. With the new different advanced models, we will try to go against the current, because, why not? As this is a data science project this forecasting models are not considered as oracles, but are really useful for analyzing the movements of stock prices with a statistical approach. The main objective of this research is to show the models fitted, compare them and encourage the use of them.
### Time Series:
A time series is a sequence of data points collected, recorded, or measured at successive, evenly-spaced time intervals. Each data point represents observations or measurements taken over time, such as stock prices, temperature readings, or sales figures. Time series data is commonly represented graphically with time on the horizontal axis and the variable of interest on the vertical axis, allowing analysts to identify trends, patterns, and changes over time. Time series data is often represented graphically as a line plot, with time depicted on the horizontal x-axis and the variable’s values displayed on the vertical y-axis. This graphical representation facilitates the visualization of trends, patterns, and fluctuations in the variable over time, aiding in the analysis and interpretation of the data.
### Time Series Forecasting:
A forecasting algorithm is an information process that seeks to predict future values based on past and present data. This historical data points are extracted and prepared trying to predict future values for a selected variable of the dataset. Time series forecasting is the process of analyzing time series data using statistics and modeling to make predictions and inform strategic decision-making. It’s not always an exact prediction, and likelihood of forecasts can vary wildly—especially when dealing with the commonly fluctuating variables in time series data as well as factors outside our control. However, forecasting insight about which outcomes are more likely—or less likely— to occur than other potential outcomes. Often, the more comprehensive the data we have, the more accurate the forecasts can be. While forecasting and “prediction” generally mean the same thing, there is a notable distinction. In some industries, forecasting might refer to data at a specific future point in time, while prediction refers to future data in general. Series forecasting is often used in conjunction with time series analysis.
<br>
Time series analysis involves developing models to gain an understanding of the data to understand the underlying causes. Analysis can provide the “why” behind the outcomes you are seeing. Forecasting then takes the next step of what to do with that knowledge and the predictable extrapolations of what might happen in the future. 
### Stationarity:
A stationary time series is one whose statistical properties (mean, variance, autocorrelation) remain constant over time. In simpler terms, the data doesn't exhibit trends, seasonality, or other patterns that change systematically over time.
<br>
Thus, time series with trends, or with seasonality, are not stationary — the trend and seasonality will affect the value of the time series at different times. On the other hand, a white noise series is stationary — it does not matter when you observe it, it should look much the same at any point in time.

#### Key characteristics of a stationary time series:
##### Constant mean:
    The average value of the series is constant over time.
##### Constant variance:
    The variability of the series around the mean is constant over time.
##### Constant autocorrelation:
    The relationship between observations at different time points remains constant over time.
#### Why is stationarity important?
Most statistical time series models assume stationarity as a fundamental condition. Non-stationary data often needs to be transformed into stationary data before applying time series models. A stationary time series will have no predictable patterns in the long-term. Time plots will show the series to be roughly horizontal (although some cyclic behavior is possible), with constant variance.
#### Common methods to achieve stationarity:
##### 1.	Differencing:
    Subtracting the previous value from the current value to remove trends.
##### 2.	Log transformation:
    Can stabilize variance in some cases.
##### 3.	Other transformations:
    E.g., square root, power transformations.
#### Methods to Check Stationarity:
##### 1.	Visual Inspection
###### Time Series Plot:
Look for trends, seasonality, or other patterns. A stationary series should fluctuate around a constant mean without clear trends.
###### ACF (Autocorrelation Function) and PACF (Partial Autocorrelation Function) Plots:
These plots measure the correlation of a time series with its past values. For a stationary series, the ACF and PACF should decay rapidly to zero.
##### 2.	Statistical Tests:
###### Augmented Dickey-Fuller (ADF) Test:
Tests for the presence of a unit root, indicating non-stationarity. A low p-value suggests stationarity.
###### Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test:
Tests for stationarity around a mean or trend. A high p-value suggests stationarity.
#### Autoregressive models:
In a multiple regression model, we forecast the variable of interest using a linear combination of predictors. In an auto regression model, we forecast the variable of interest using a linear combination of past values of the variable. The model assumes that the current value is a linear combination of past values plus a random error term. The term auto regression indicates that it is a regression of the variable against itself.
<br>
Thus, an autoregressive model of order p can be written as:
