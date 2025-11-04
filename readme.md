# Time Series Forecasting using ARIMA Model

## 1. Overview

This test applies **Autoregressive Integrated Moving Average (ARIMA)** models to forecast passenger counts for various transit routes. The goal is to predict short-term demand patterns using historical ridership data for routes such as **Local Route**, **Light Rail**, **Peak Service**, **Rapid Route**, and **School**.

The ARIMA model is a classical yet powerful statistical method for **univariate time series forecasting** that captures autocorrelations within the data without relying on external predictors.

---

## 2. Chosen Algorithm: ARIMA

### What is ARIMA?

ARIMA stands for **AutoRegressive Integrated Moving Average**, represented as **ARIMA(p, d, q)**, where:

- **p** → Number of autoregressive (AR) terms.
- **d** → Degree of differencing needed to make the series stationary.
- **q** → Number of moving average (MA) terms.

The ARIMA model combines:

- **AR (AutoRegressive)**: uses the dependency between current and past values.
- **I (Integrated)**: removes trends and seasonality through differencing.
- **MA (Moving Average)**: models the error term as a linear combination of past forecast errors.

---

## 3. Model Parameters Used

| Parameter                      | Value | Description                                                                           |
| :----------------------------- | :---- | :------------------------------------------------------------------------------------ |
| **p (AR order)**               | 1     | Captures one period of autocorrelation, meaning today's value depends on yesterday's. |
| **d (Degree of differencing)** | 0     | Data is already stationary, so differencing is not needed.                            |
| **q (MA order)**               | 1     | Captures one lag of the moving average component, smoothing short-term noise.         |

Thus, the final model applied for each route is:

> **ARIMA(1, 0, 1)**

---

## 4. Why These Parameters Were Chosen

1. **Stationarity Check:**  
   The time series was found to be approximately stationary through visualization and statistical tests (ADF test), allowing `d = 0`.
2. **Autocorrelation & Partial Autocorrelation Analysis:**  
   The ACF and PACF plots indicated significant lags at 1, supporting `p = 1` and `q = 1`.
3. **Simplicity and Interpretability:**  
   ARIMA(1,0,1) is a balanced model that captures short-term dependencies while remaining computationally efficient.
4. **Empirical Performance:**  
   The model produced stable forecasts across all route types without overfitting.

---

## 5. Forecasting Setup

For each route:

- The model was trained on the complete time series.
- A **7-day forecast horizon** was generated using the fitted ARIMA model.
- Visualizations include:
  - The last 30 days of observed data.
  - The predicted 7-day forecast with trend visualization.

```python
model = ARIMA(df_news[col], order=(1, 0, 1))
results = model.fit()
forecast = results.forecast(steps=7)
```
