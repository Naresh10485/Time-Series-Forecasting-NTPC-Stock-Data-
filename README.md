# Time-Series-Forecasting-NTPC-Stock-Data-
## Contents
### Overview :
This project investigates the application of various time series and machine learning forecasting models to a real-world stock closing price dataset. We begin by exploring the characteristics of stock prices and conducting a comprehensive analysis of the dataset. Subsequently, we focus on predicting the daily closing price of the S&P500 index using a range of forecasting models. These models will undergo rigorous evaluation, comparison, and analysis aligned with the project guidelines. Ultimately, the project aims to build a model capable of forecasting the S&P500 closing price for the next 30 days.
### Objective :
Predicting stock market behaviour has long been a subject of interest, with many believing it to be a complex, unpredictable process. The random walk theory, emphasizing the independence of past and future price movements, supports this view. However, recent advancements in data science and modelling techniques encourage a re-examination of this perspective.
<br>
This research aims to explore the potential of various forecasting models in predicting stock prices, specifically the S&P500 closing price. By comparing and contrasting these models, we seek to demonstrate their utility as tools for analysing stock price movements rather than infallible prediction instruments. Ultimately, this study aims to contribute to the understanding of stock price dynamics and the application of data-driven forecasting methods in financial analysis.

### Time Series:
A time series is a sequence of data points collected, recorded, or measured at successive, evenly-spaced time intervals. Each data point represents observations or measurements taken over time, such as stock prices, temperature readings, or sales figures. Time series data is commonly represented graphically with time on the horizontal axis and the variable of interest on the vertical axis, allowing analysts to identify trends, patterns, and changes over time. Time series data is often represented graphically as a line plot, with time depicted on the horizontal x-axis and the variable’s values displayed on the vertical y-axis. This graphical representation facilitates the visualization of trends, patterns, and fluctuations in the variable over time, aiding in the analysis and interpretation of the data. 
### Time Series Forecasting:
A forecasting algorithm is a computational process that predicts future values based on historical data. It involves collecting, processing, and analysing past data points to create a model that can estimate future outcomes for a specific variable. Time series forecasting is the process of analysing time series data using statistics and modelling to make predictions and inform strategic decision-making. It’s not always an exact prediction, and likelihood of forecasts can vary wildly—especially when dealing with the commonly fluctuating variables in time series data as well as factors outside our control. However, forecasting insight about which outcomes are more likely—or less likely— to occur than other potential outcomes. Often, the more comprehensive the data we have, the more accurate the forecasts can be. While forecasting and “prediction” generally mean the same thing, there is a notable distinction. In some industries, forecasting might refer to data at a specific future point in time, while prediction refers to future data in general. Series forecasting is often used in conjunction with time series analysis.
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
 ![AR_equation](Visualization/ARmodel.png)

where ε_t is white noise. This is like a multiple regression but with lagged values of y_t as predictors. We refer to this as an AR(p) model, an autoregressive model of order p.
Autoregressive models are remarkably flexible at handling a wide range of different time series patterns. Changing the parameters ϕ_1  ,…,ϕ_p results in different time series patterns. The variance of the error term εt will only change the scale of the series, not the patterns.

#### Moving Average Models: 
Rather than using past values of the forecast variable in a regression, a moving average model uses past forecast errors in a regression-like model.
 ![MA_equation](Visualization/MAmodel.png)

where εt is white noise. We refer to this as an MA(q) model, a moving average model of order q. Of course, we do not observe the values of εt, so it is not really a regression in the usual sense. The model assumes that the current value is a linear combination of past error terms plus a random error term.

#### Arima : Autoregressive Integrating Moving Average
ARIMA is a statistical model used for time series data. It stands for Auto Regressive Integrated Moving Average. This method is often referred to as the Box-Jenkins approach. Box and Jenkins introduced the idea of using differencing to convert data that doesn't have a constant mean or variance (non-stationary) into data that does (stationary).
The full model can be written as,
 ![Arima_equation](Visualization/Arima.png)
Where, Yt= y_t‘, is the differenced time series value, ϕ and θ are unknown parameters and e are independent identically distributed error terms with zero mean. Here, Yt is expressed in terms of its past values and the current and past values of error terms.

<br>
The ARIMA model combines three key elements:

##### Auto Regression (AR):
    In auto-regression the values of a given time series data are regressed on their own lagged values, which is indicated by the “p” value in the model.
##### Differencing (I for Integrated):
    To handle data with trends or seasonality, differencing is applied. This involves subtracting previous values from current values to stabilize the data. The 'd' parameter specifies the order of differencing. If d = 1, it looks at the difference between two time-series entries, if d = 2 it looks at the differences of the differences obtained at d =1, and so forth.
##### Moving Average (MA):
    This component incorporates past error terms into the prediction. The 'q' parameter determines the number of error terms included.
<br>
Together, these components form the ARIMA(p, d, q) model, where p, d, and q represent the order of autoregression, differencing, and moving average, respectively.

