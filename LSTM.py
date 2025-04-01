import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

df = pd.read_csv("PUB_DemandZonal_2024.csv", skiprows=3)
df.columns = ['Date', 'Hour', 'Ontario Demand', 'Northwest', 'Northeast', 'Ottawa', 'East',
              'Toronto', 'Essa', 'Bruce', 'Southwest', 'Niagara', 'West', 'Zone Total', 'Diff']
df['Hour'] = df['Hour'].astype(int) - 1
df['Timestamp'] = pd.to_datetime(df['Date']) + pd.to_timedelta(df['Hour'], unit='h')
df.set_index('Timestamp', inplace=True)
df.drop(columns=['Date', 'Hour', 'Zone Total', 'Diff'], inplace=True)
df = df.apply(pd.to_numeric, errors='coerce').asfreq('H').dropna()

df['trend'] = np.linspace(0, 1, len(df))
df['Toy'] = df.index.dayofyear / 365.25
df['weekday'] = df.index.dayofweek
df['LoadD'] = df['Ontario Demand'].shift(24)
df['LoadW'] = df['Ontario Demand'].shift(24 * 7)
df['Temps95'] = df['Ontario Demand'].ewm(alpha=1-0.95).mean()
df = pd.concat([df, pd.get_dummies(df['weekday'], prefix='day')], axis=1)
df.dropna(inplace=True)

X = df.drop(columns=['Ontario Demand', 'weekday'])
y = df['Ontario Demand'].values.reshape(-1, 1)

X_scaler = StandardScaler()
y_scaler = StandardScaler()
X_scaled = X_scaler.fit_transform(X)
y_scaled = y_scaler.fit_transform(y)

def create_sequences(X, y, window_size):
    Xs, ys = [], []
    for i in range(len(X) - window_size):
        Xs.append(X[i:i+window_size])
        ys.append(y[i+window_size])
    return np.array(Xs), np.array(ys)

window_size = 168  # one week of hourly data
X_seq, y_seq = create_sequences(X_scaled, y_scaled, window_size)
split_index = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split_index], X_seq[split_index:]
y_train, y_test = y_seq[:split_index], y_seq[split_index:]

model = Sequential([
    LSTM(64, activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(1)
])
model.compile(optimizer='adam', loss='mae')
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# --- 5. Predict and Evaluate ---
y_pred_scaled = model.predict(X_test)
y_pred = y_scaler.inverse_transform(y_pred_scaled)
y_true = y_scaler.inverse_transform(y_test.reshape(-1, 1))
mae_lstm = mean_absolute_error(y_true, y_pred)
print("LSTM MAE:", mae_lstm)

# Plot
plt.figure(figsize=(14, 6))
plt.plot(y_true[:500], label='Actual', color='black')
plt.plot(y_pred[:500], label='LSTM Prediction', linestyle='--', color='blue')
plt.title(f"LSTM Forecast (First 500 Test Hours) - MAE: {mae_lstm:.2f} MW")
plt.xlabel("Hour")
plt.ylabel("Ontario Demand (MW)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
