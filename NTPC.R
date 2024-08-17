library(quantmod)
# Download the data from yahoo finance for desired date
getSymbols("NTPC.NS",src="yahoo",from="2022-01-01",to = "2024-07-01")
# Remove rows with missing values
NTPC.NS <- na.omit(NTPC.NS)
library(lubridate)
# setting date format
date = index(NTPC.NS)
date = as.Date(date)
# Access the downloaded data
head(NTPC.NS)
# plot the downloaded data
chartSeries(NTPC.NS,type="line",TA = NULL)
chartSeries(NTPC.NS,TA=c(addVo(),addBBands(),addMACD()))
# # test for stationarity
library(tseries)
#ADF TEST 
print(adf.test(NTPC.NS$NTPC.NS.Close))
#KPSS TEST
print(kpss.test(NTPC.NS$NTPC.NS.Close))
#Plot ACF and PACF
par(mfrow = c(1, 2))
acf(NTPC.NS$NTPC.NS.Close)
pacf(NTPC.NS$NTPC.NS.Close)
par(mfrow = c(1, 1))

# transforming non-stationary data into stationary
ts_data=diff(NTPC.NS$NTPC.NS.Close)
ts_data=na.omit(ts_data)
chartSeries(ts_data, type = "line")
adf.test(ts_data)
#apply auto arima to the stationary dataset 
library(forecast)
arima_model<- auto.arima(NTPC.NS$NTPC.NS.Close)
arima_model
summary(arima_model)
## Arima Results
# Diagnostics on Residuals
et<- residuals(arima_model)
plot(et,ylab="Residuals",main="Residuals(Arima(1,1,0)) vs. Time")

print(kpss.test(et))
print(adf.test(et))      
acf(et)
# Histogram of Residuals & Normality Assumption
hist(et,freq = F,main="Histogram of Residuals")
curve(dnorm(x, mean=mean(et), sd=sd(et)), add=TRUE, col="red")
# Box test for lag=1
Box.test(et, lag= 1, type="Ljung-Box",fitdf = 2)
#fitdf= (p+d+q) from ARIMA(p,d,q)
Box.test(et, type="Ljung-Box")
# Diagnostics for Arima
tsdiag(arima_model)

plot(as.ts(NTPC.NS$NTPC.NS.Close))

#Dataset forecasting  for the  next  30  days
pforecast <- forecast(arima_model,h=30)
print(pforecast)
plot(pforecast)
head(pforecast$mean)
head(pforecast$upper)
head(pforecast$lower)

#printing future line with the dates
last_date <- tail(index(NTPC.NS+1), 1);last_date
# Creating a sequence of future dates
forecast_dates <- seq(last_date, by = "day", length.out = 30);forecast_dates
forecast=data.frame(forecast_dates,pforecast);forecast
library(ggplot2)
# Creating a data frame with date and forecast values
result <- data.frame(date = forecast_dates, value = pforecast$mean);result
# Create the plot using ggplot2
ggplot(result, aes(x = date, y = value)) +
  geom_line() +
  labs(x = "Date", y = "Forecast Value")

#Dividing the data into train & test sets , applying the model
N = length (NTPC.NS$NTPC.NS.Close)
n = 0.8*N
train = NTPC.NS$NTPC.NS.Close[1:n, ]
test = NTPC.NS$NTPC.NS.Close[(n+1):N,]


trainarimafit <- auto.arima(train$NTPC.NS.Close)
summary(trainarimafit)
predlen= length(test)
trainarima_fit <- forecast(trainarimafit, h= predlen)

#Plotting mean predicted  values vs real data
meanvalues<- as.vector(trainarima_fit$mean)
precios <- as.vector(test$NTPC.NS.Close)
plot(meanvalues, type = "l",col="red")
lines(precios, type = "l",col="blue")

