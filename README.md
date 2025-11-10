# 📈 Day 14 — Stock Price Prediction using Machine Learning

This project focuses on predicting future stock prices based on historical data using **Linear Regression** and **Random Forest Regressor**. It’s a comparison of classical ML models applied to time-series financial data — simple yet powerful in demonstrating trend analysis and forecasting.

---

## 🚀 Overview
- Built a **supervised regression pipeline** to forecast stock closing prices  
- Engineered time-lag features for short-term trend learning  
- Compared **Linear Regression** and **Random Forest** models for accuracy and stability  
- Visualized model performance and feature importance  

---

## 🧠 Workflow
1. **Data Loading & Feature Engineering** — Load stock dataset and generate lag-based predictors  
2. **Model Training** — Fit both Linear Regression and Random Forest models  
3. **Evaluation** — Compare performance using MAE, RMSE, and R² metrics  
4. **Visualization** — Plot predicted vs. actual values and feature importances  

---

## 💡 Results
| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Linear Regression | 0.90 | 1.13 | -0.41 |
| Random Forest | 1.60 | 1.82 | -2.66 |

📊 Both models were tested and evaluated for overfitting and interpretability.  
Visuals were saved as:
- `stock_actual_vs_pred.png`
- `stock_feature_importance.png`

---

## 🧩 Tech Stack
Python | Pandas | Scikit-learn | Matplotlib | NumPy

---

---

**Update:** Added dependencies, changelog, and license for version 1.0.0
