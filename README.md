# # 🕒 Time Series Forecasting Project

This project demonstrates a full pipeline for time series forecasting of **weekly sales**, covering everything from data analysis to deploying a GRU deep learning model using **FastAPI**.

---

## ✅ Project Checklist

- [x] **Time Series Analysis**
- [x] **Exponential Smoothing Models (Holt, Holt-Winters)**
- [x] **ARIMA Family (ARIMA, SARIMA, SARIMAX)**
- [x] **Machine Learning Regressors (Random Forest, XGBoost)**
- [x] **Deep Learning Models (ANN, CNN, RNN, LSTM, GRU)**
- [x] **Prophet Model**
- [x] **Model Evaluation (RMSE, MAE, MAPE)**
- [x] **Deployment via FastAPI**

---

## 📊 1. Time Series Analysis

- **Visualization** of weekly sales over time.
- **Trend and seasonality** decomposition using `seasonal_decompose`.
- Analysis of **external regressors**: `Temperature`, `Fuel_Price`, `CPI`, `Unemployment`, `IsHoliday`.
- Performed **ACF/PACF** plots to determine autocorrelations and lags.

---

## 📉 2. Exponential Smoothing Family

Implemented:
- **Holt’s Linear Trend**
- **Holt-Winters Seasonal Smoothing**

Forecast accuracy was moderate. Better at capturing long-term trends than sudden changes.

---

## 📈 3. ARIMA Family Models

- **ARIMA** for baseline auto-regression.
- **SARIMA** for seasonal patterns.
- **SARIMAX** to incorporate external regressors (e.g., holidays, temperature).

Tuned using `auto_arima` and evaluated using RMSE.

---

## 🤖 4. Machine Learning Regressors

Trained traditional regressors with time-based feature engineering:
- Lag features, rolling averages, date features
- Models used:  
  - **Random Forest Regressor**  
  - **XGBoost Regressor**

Performance was solid but struggled with temporal patterns compared to deep models.

---

## 🧠 5. Deep Learning Models

Applied various neural networks:
- **ANN**  
- **CNN**  
- **RNN**  
- **LSTM**  
- **GRU**  

### 🔢 RMSE Results:
| Model     | RMSE    |
|-----------|---------|
| ANN       | 630.51  |
| RNN       | 610.98  |
| LSTM      | 573.62  |
| GRU       | ✅ **545.87** (Best) |
| CNN       | 577.55  |
| Deep CNN  | 581.94  |
| Deep LSTM | 556.80  |

---

## 📅 6. Prophet Model

Used **Facebook Prophet** for additive time series forecasting.  
- Captured trends and holidays well  
- Easy interpretability and modularity  
- RMSE was higher than GRU but interpretability was useful

---

## 📏 7. Model Evaluation

Used the following metrics:
- **RMSE (Root Mean Squared Error)**
- **MAE (Mean Absolute Error)**
- **MAPE (Mean Absolute Percentage Error)**

### 🔍 Model Comparison
- Traditional ML models were stable but limited in sequential learning.
- Prophet was interpretable but underperformed on recent trends.
- Deep learning (GRU) had the best forecasting accuracy.

---

## 🚀 8. Deployment (FastAPI)

The best model (**GRU**) was deployed via **FastAPI** and served using **Uvicorn**.

### 🛠️ Run the API

```bash
uvicorn main:app --reload
