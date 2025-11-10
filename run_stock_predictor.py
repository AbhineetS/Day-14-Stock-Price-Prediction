# run_stock_predictor.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

CSV_PATH = "stock_data.csv"
PLOT_COMP = "stock_actual_vs_pred.png"
PLOT_IMP  = "stock_feature_importance.png"
MODEL_OUT = "stock_random_forest.pkl"

# ✅ fixed version: manual RMSE
def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5

def build_features(df):
    if 'Date' not in df.columns or 'Close' not in df.columns:
        raise SystemExit("CSV must have at least 'Date' and 'Close' columns.")
    df = df[['Date','Close']].copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    df['ret_1d']   = df['Close'].pct_change(1)
    df['lag1']     = df['Close'].shift(1)
    df['lag2']     = df['Close'].shift(2)
    df['lag3']     = df['Close'].shift(3)
    df['roll_mean_3'] = df['Close'].rolling(3).mean()
    df['roll_mean_5'] = df['Close'].rolling(5).mean()
    df['roll_std_5']  = df['Close'].rolling(5).std()

    df['target'] = df['Close'].shift(-1)
    df = df.dropna().reset_index(drop=True)
    return df

def train_and_eval(df_feat):
    features = ['ret_1d','lag1','lag2','lag3','roll_mean_3','roll_mean_5','roll_std_5']
    X = df_feat[features].values
    y = df_feat['target'].values
    dates = df_feat['Date'].values

    split_idx = int(len(df_feat) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_test = dates[split_idx:]

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    lin = LinearRegression()
    rf  = RandomForestRegressor(n_estimators=400, random_state=42)

    lin.fit(X_train_sc, y_train)
    rf.fit(X_train, y_train)

    lin_pred = lin.predict(X_test_sc)
    rf_pred  = rf.predict(X_test)

    def report(name, y_true, y_hat):
        print(f"{name} -> MAE: {mean_absolute_error(y_true, y_hat):.2f} | "
              f"RMSE: {rmse(y_true, y_hat):.2f} | R²: {r2_score(y_true, y_hat):.3f}")

    print("\n=== Evaluation (Test Set) ===")
    report("Linear Regression", y_test, lin_pred)
    report("Random Forest   ", y_test, rf_pred)

    # Plot actual vs predicted (RF)
    plt.figure(figsize=(10,5))
    plt.plot(dates_test, y_test, label="Actual")
    plt.plot(dates_test, rf_pred, label="Predicted (RF)")
    plt.title("Actual vs Predicted Close (Test Set)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_COMP, dpi=150)
    plt.close()
    print(f"Saved: {PLOT_COMP}")

    # Feature importance
    importances = rf.feature_importances_
    order = np.argsort(importances)[::-1]
    labels = np.array(features)[order]

    plt.figure(figsize=(8,5))
    plt.barh(range(len(order)), importances[order])
    plt.yticks(range(len(order)), labels)
    plt.gca().invert_yaxis()
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()
    plt.savefig(PLOT_IMP, dpi=150)
    plt.close()
    print(f"Saved: {PLOT_IMP}")

    joblib.dump({'model': rf, 'features': features}, MODEL_OUT)
    print(f"Saved model: {MODEL_OUT}")

def main():
    if not os.path.exists(CSV_PATH):
        raise SystemExit(f"Missing '{CSV_PATH}'. Put your CSV with Date, Close columns in the folder.")
    df = pd.read_csv(CSV_PATH)
    if len(df) < 20:
        raise SystemExit("Need at least ~20 rows to run. Add more historical rows to 'stock_data.csv'.")
    df_feat = build_features(df)
    if len(df_feat) < 15:
        raise SystemExit("After feature creation there are too few rows. Add more data.")
    train_and_eval(df_feat)

if __name__ == "__main__":
    main()