import numpy as np
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# --- Kalman Filter Function ---
def apply_kalman_filter(preds, actuals, Q=0.01, R=1000):
    n = len(preds)
    x_est = 0   # initial bias estimate
    P = 1       # initial uncertainty
    filtered_preds = []

    for t in range(n):
        # Prediction
        x_pred = x_est
        P_pred = P + Q

        # Update
        y_tilde = actuals[t] - (preds[t] + x_pred)
        K = P_pred / (P_pred + R)
        x_est = x_pred + K * y_tilde
        P = (1 - K) * P_pred

        filtered_preds.append(preds[t] + x_est)

    return np.array(filtered_preds)

# --- Apply to MLP ---
y_kf_mlp = apply_kalman_filter(mlp_preds, y_test_mlp)
mae_mlp_kf = mean_absolute_error(y_test_mlp, y_kf_mlp)
print(f"MLP MAE after Kalman Filter: {mae_mlp_kf:.2f} MW")

# --- Apply to LSTM ---
y_kf_lstm = apply_kalman_filter(lstm_preds, y_test_lstm)
mae_lstm_kf = mean_absolute_error(y_test_lstm, y_kf_lstm)
print(f"LSTM MAE after Kalman Filter: {mae_lstm_kf:.2f} MW")

# --- Plot (compare raw and filtered MLP) ---
plt.figure(figsize=(14, 6))
plt.plot(y_test_mlp[:500], label='Actual', color='black')
plt.plot(mlp_preds[:500], label='MLP', linestyle='--', color='orange')
plt.plot(y_kf_mlp[:500], label='MLP + Kalman', linestyle=':', color='blue')
plt.title(f"MLP Kalman Forecast (First 500 Hours) - MAE: {mae_mlp_kf:.2f} MW")
plt.xlabel("Hour")
plt.ylabel("Ontario Demand (MW)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot (compare raw and filtered LSTM) ---
plt.figure(figsize=(14, 6))
plt.plot(y_test_lstm[:500], label='Actual', color='black')
plt.plot(lstm_preds[:500], label='LSTM', linestyle='--', color='green')
plt.plot(y_kf_lstm[:500], label='LSTM + Kalman', linestyle=':', color='blue')
plt.title(f"LSTM Kalman Forecast (First 500 Hours) - MAE: {mae_lstm_kf:.2f} MW")
plt.xlabel("Hour")
plt.ylabel("Ontario Demand (MW)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
