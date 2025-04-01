import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras import layers, models

# ----------------------------
# 1. Load and preprocess data
# ----------------------------
df = pd.read_csv("PUB_DemandZonal_2025.csv", skiprows=3)
df.columns = ['Date', 'Hour', 'Ontario Demand', 'Northwest', 'Northeast', 'Ottawa', 'East',
              'Toronto', 'Essa', 'Bruce', 'Southwest', 'Niagara', 'West', 'Zone Total', 'Diff']
df['Hour'] = df['Hour'].astype(int) - 1
df['Timestamp'] = pd.to_datetime(df['Date']) + pd.to_timedelta(df['Hour'], unit='h')
df.set_index('Timestamp', inplace=True)
df.drop(columns=['Date', 'Hour', 'Zone Total', 'Diff'], inplace=True)
df = df.apply(pd.to_numeric, errors='coerce').asfreq('H').dropna()

# Feature engineering
df['trend'] = np.linspace(0, 1, len(df))
df['Toy'] = df.index.dayofyear / 365.25
df['weekday'] = df.index.dayofweek
df['LoadD'] = df['Ontario Demand'].shift(24)
df['LoadW'] = df['Ontario Demand'].shift(24 * 7)
df['Temps95'] = df['Ontario Demand'].ewm(alpha=1-0.95).mean()
df = pd.concat([df, pd.get_dummies(df['weekday'], prefix='day')], axis=1)
df.dropna(inplace=True)

# Split and scale
X = df.drop(columns=['Ontario Demand', 'weekday'])
y = df['Ontario Demand'].values.reshape(-1, 1)
X_scaler = StandardScaler()
y_scaler = StandardScaler()
X_scaled = X_scaler.fit_transform(X)
y_scaled = y_scaler.fit_transform(y)

# Sequence creation
def create_sequences(X, y, window_size):
    Xs, ys = [], []
    for i in range(len(X) - window_size):
        Xs.append(X[i:i+window_size])
        ys.append(y[i+window_size])
    return np.array(Xs), np.array(ys)

window_size = 168
X_seq, y_seq = create_sequences(X_scaled, y_scaled, window_size)

# Train/test split
split_idx = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

# ----------------------------
# 2. Transformer model
# ----------------------------
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def build_transformer_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = transformer_encoder(inputs, head_size=64, num_heads=4, ff_dim=128, dropout=0.1)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(1)(x)
    return models.Model(inputs, x)

model = build_transformer_model(input_shape=X_train.shape[1:])
model.compile(optimizer="adam", loss="mae")
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_split=0.1)

# Prediction and inverse scaling
y_pred_scaled = model.predict(X_test)
y_pred = y_scaler.inverse_transform(y_pred_scaled)
y_true = y_scaler.inverse_transform(y_test)

# ----------------------------
# 3. Kalman Filter adaptation
# ----------------------------
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

# Apply Kalman filter on Transformer predictions
y_kalman = apply_kalman_filter(y_pred.ravel(), y_true.ravel())
mae_kalman = mean_absolute_error(y_true, y_kalman)
print("MAE after Kalman Filter:", mae_kalman)

# ----------------------------
# 4. Plot
# ----------------------------
plt.figure(figsize=(14, 6))
plt.plot(y_true[:500], label='Actual', color='black')
plt.plot(y_pred[:500], label='Transformer', linestyle='--', color='green')
plt.plot(y_kalman[:500], label='Kalman-Filtered', linestyle=':', color='blue')
plt.title(f"Transformer Forecast (First 500 Test Hours)\nKalman MAE: {mae_kalman:.2f} MW")
plt.xlabel("Hour")
plt.ylabel("Ontario Demand (MW)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
