import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
import warnings

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

#Common Functions
def plot2(df, y_actual, y_predict, split_index, pred_period, zone, modelname):

    mae = mean_absolute_error(y_actual, y_predict)
    test_datetimes = df['datetime'][split_index:split_index+int(pred_period)*24]

    plt.figure(figsize=(12, 5))
    plt.plot(test_datetimes, y_actual, label="Actual", linewidth=1)
    plt.plot(test_datetimes, y_predict, label="Predicted " + modelname, linewidth=1)
    plt.title(f"{zone} Demand Forecast {modelname} - {pred_period} Days")
    plt.xlabel("Hour")
    plt.ylabel("Demand")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.hist(y_actual - y_predict, bins=50, density=True)
    plt.title(f"Prediction Error Distribution {modelname}")
    plt.xlabel("Error")
    plt.ylabel("Density")
    plt.grid(True)
    plt.show()

    print("MAE:", mae)

def construct_features(df,zone,lags_morning,lags_day,exog_features,split_morning = 8):
    X, y = [], []
    for i in range(504, len(df)):
        hour = df['hour'].iloc[i]
        lags = lags_morning if hour <= split_morning else lags_day
        try:
            x_lags = [df[zone].iloc[i - lag] for lag in lags]
            x_exog = [df[feat].iloc[i] for feat in exog_features]
            X.append(x_lags + x_exog)
            y.append(df[zone].iloc[i])
        except:
            continue
    return np.array(X), np.array(y)

# Model Construction Functions
#-----------------------------------------------------------------------------------------
#KalmanAR
def kalman_ar(X, y, Q=1e-2, R=1):
    n, d = X.shape
    theta = np.zeros(d)
    P = np.eye(d)
    Q = Q * np.eye(d)

    theta_hist = []
    y_pred = []

    for t in range(n):
        x_t = X[t]
        y_t = y[t]

        if np.isnan(x_t).any() or np.isnan(y_t):
            theta_hist.append(np.full(d, np.nan))
            y_pred.append(np.nan)
            continue

        y_hat = np.dot(x_t, theta)
        S = np.dot(np.dot(x_t, P), x_t.T) + R
        K = np.dot(P, x_t) / S

        theta = theta + K * (y_t - y_hat)
        P = P - np.outer(K, np.dot(x_t, P)) + Q

        theta_hist.append(theta.copy())
        y_pred.append(y_hat)

    return np.array(y_pred), np.array(theta_hist)

def train_kalman_ar(df, zone, lags_morning, lags_day, exog_features, pred_period='14'):
    X, y = construct_features(df,zone,lags_morning,lags_day,exog_features)

    # Clean and impute
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    pred_period = int(pred_period)
    split_index = len(y) - 24 * pred_period

    X_train, y_train = X[:split_index], y[:split_index]
    X_test, y_test = X[split_index:], y[split_index:]

    # Fit Kalman Filter AR
    y_train_pred, theta_hist = kalman_ar(X_train, y_train)
    final_theta = theta_hist[-1]

    # Predict on test data
    y_test_pred = X_test @ final_theta

    return y_test, y_test_pred, final_theta

#-----------------------------------------------------------------------------------------
#VIKING
def viking_ar(X, y, rho_a=1e-3, rho_b=1e-3):
    n, d = X.shape
    theta = np.zeros(d)
    P = np.eye(d)
    a = np.log(1.0)   # log observation variance
    b = np.log(1e-2)  # log process variance

    theta_hist = []
    y_pred = []

    for t in range(n):
        x_t = X[t]
        y_t = y[t]

        Q = np.exp(b) * np.eye(d)
        R = np.exp(a)

        y_hat = np.dot(x_t, theta)
        S = x_t @ P @ x_t.T + R
        K = P @ x_t / S

        theta = theta + K * (y_t - y_hat)
        P = P - np.outer(K, x_t @ P) + Q

        err = y_t - y_hat
        grad_a = 0.5 * ((err ** 2) / R - 1)
        grad_b = 0.5 * (np.trace(np.linalg.inv(Q) @ (P - Q)) - d)

        a += rho_a * grad_a
        b += rho_b * grad_b

        theta_hist.append(theta.copy())
        y_pred.append(y_hat)

    return np.array(y_pred), np.array(theta_hist)

def train_viking_ar(df, zone, lags_morning, lags_day, exog_features, pred_period='14'):
    X, y = construct_features(df,zone,lags_morning,lags_day,exog_features)

    # Clean and impute
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    pred_period = int(pred_period)
    split_index = len(y) - 24 * pred_period

    X_train, y_train = X[:split_index], y[:split_index]
    X_test, y_test = X[split_index:], y[split_index:]

    # Fit Viking AR
    y_train_pred, theta_hist = viking_ar(X_train, y_train)
    final_theta = theta_hist[-1]

    # Predict on test data
    y_test_pred = X_test @ final_theta

    return y_test, y_test_pred, final_theta

#-----------------------------------------------------------------------------------------
#LSTM

def build_lstm_model_manual(input_shape):
    model = Sequential()
    model.add(LSTM(units=64, return_sequences=False, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

def train_lstm_manual(df, zone, lags_morning, lags_day, exog_features, pred_period='14'):
    X, y = construct_features(df, zone, lags_morning, lags_day, exog_features)

    # Remove any NaNs just in case
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X, y = X[mask], y[mask]

    # Scale the features
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    pred_period = int(pred_period)
    split_index = len(y) - 24 * pred_period

    X_train, y_train = X[:split_index], y[:split_index]
    X_test, y_test = X[split_index:], y[split_index:]

    # Reshape for LSTM: (samples, time_steps=1, features)
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    model = build_lstm_model_manual((X_train.shape[1], X_train.shape[2]))

    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

    # Predictions
    y_pred = model.predict(X_test)

    # Reverse scaling
    y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_inv = scaler_y.inverse_transform(y_pred).flatten()

    return y_test_inv, y_pred_inv, model, history

#-----------------------------------------------------------------------------------------
#MLP

def build_mlp_model(input_dim):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=input_dim))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

def train_mlp_manual(df, zone, lags_morning, lags_day, exog_features, pred_period='14'):
    X, y = construct_features(df, zone, lags_morning, lags_day, exog_features)

    # Remove NaNs
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X, y = X[mask], y[mask]

    # Scale
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    pred_period = int(pred_period)
    split_index = len(y) - 24 * pred_period

    X_train, y_train = X[:split_index], y[:split_index]
    X_test, y_test = X[split_index:], y[split_index:]

    model = build_mlp_model(X_train.shape[1])

    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

    # Predictions
    y_pred = model.predict(X_test)

    # Reverse scaling
    y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_inv = scaler_y.inverse_transform(y_pred).flatten()

    return y_test_inv, y_pred_inv, model, history

#-----------------------------------------------------------------------------------------
#Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)

class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, model_dim=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=nhead, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(model_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.decoder(x[:, -1, :])
        return x.squeeze()

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X[:, None, :], dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_transformer(df, zone, lags_morning, lags_day, exog_features, pred_period='14'):
    X, y = construct_features(df, zone, lags_morning, lags_day, exog_features)

    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    pred_period = int(pred_period)
    split_index = len(y) - 24 * pred_period

    X_train, y_train = X[:split_index], y[:split_index]
    X_test, y_test = X[split_index:], y[split_index:]

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    train_ds = TimeSeriesDataset(X_train, y_train_scaled)
    test_ds = TimeSeriesDataset(X_test, y_test_scaled)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64)

    model = TransformerRegressor(input_dim=X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(100):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}, Train Loss: {total_loss/len(train_loader):.4f}")

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            pred = model(xb)
            preds.append(pred.numpy())
            trues.append(yb.numpy())

    y_pred_scaled = np.concatenate(preds)
    y_true_scaled = np.concatenate(trues)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = scaler_y.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()

    return y_true, y_pred, model
