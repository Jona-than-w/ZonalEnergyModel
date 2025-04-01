import numpy as np
import pandas as pd

file_path = "PUB_DemandZonal_2024.csv"
data = pd.read_csv(file_path)

# Convert the format of the data
df_cleaned = data.iloc[3:].reset_index(drop=True)
df_cleaned.columns = ["Date", "Hour", "Ontario Demand", "Northwest", "Northeast", "Ottawa",
                      "East", "Toronto", "Essa", "Bruce", "Southwest", "Niagara", "West", "Zone Total", "Diff"]
df_cleaned["Date"] = pd.to_datetime(df_cleaned["Date"])
df_cleaned["Hour"] = pd.to_numeric(df_cleaned["Hour"])

# Create a timestamp column for better time series analysis
df_cleaned["Timestamp"] = df_cleaned.apply(lambda row: row["Date"] + pd.to_timedelta(row["Hour"] - 1, unit="h"), axis=1)


demand_columns = ["Ontario Demand", "Northwest", "Northeast", "Ottawa", "East", "Toronto",
                  "Essa", "Bruce", "Southwest", "Niagara", "West", "Zone Total", "Diff"]
df_cleaned[demand_columns] = df_cleaned[demand_columns].apply(pd.to_numeric, errors="coerce")

# Firstly analyze the Ontario Demand
df_entire = df_cleaned[["Timestamp", "Ontario Demand", "Zone Total"]]
df_entire.to_csv("Ontario_Demand.csv", index=False)

# Plot hourly, daily and weekly demand trends
import matplotlib.pyplot as plt

#only using 30% of the data for visualization
df_entire = df_entire.head(int(len(df_entire) * 0.3))
df_daily = df_entire.resample("D", on="Timestamp").mean()
df_weekly = df_entire.resample("W", on="Timestamp").mean()

fig, axes = plt.subplots(3, 1, figsize=(12, 8))

# Daily trend
axes[0].plot(df_daily.index, df_daily["Ontario Demand"], label="Ontario Demand", linewidth=1.5)
axes[0].set_title("Daily Average Electricity Demand")
axes[0].set_ylabel("Demand (MW)")
axes[0].legend()

# Weekly trend
axes[1].plot(df_weekly.index, df_weekly["Ontario Demand"], label="Ontario Demand", linewidth=1.5, color="purple")
axes[1].set_title("Weekly Average Electricity Demand")
axes[1].set_ylabel("Demand (MW)")
axes[1].legend()

#Hourly trend
axes[2].plot(df_entire.index, df_entire["Ontario Demand"], label="Ontario Demand", linewidth=1.5, color="green")
axes[2].set_title("Hourly Average Electricity Demand")
axes[2].set_ylabel("Demand (MW)")
axes[2].legend()

plt.tight_layout()
plt.show()

# See the seasonal trend 
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf

# Perform seasonal decomposition
decomposition = seasonal_decompose(df_cleaned.set_index("Timestamp")["Ontario Demand"], model="additive", period=7*24)

# Plot seasonal decomposition
fig, axes = plt.subplots(4, 1, figsize=(12, 10))

decomposition.observed.plot(ax=axes[0], title="Observed")
decomposition.trend.plot(ax=axes[1], title="Trend")
decomposition.seasonal.plot(ax=axes[2], title="Seasonality")
decomposition.resid.plot(ax=axes[3], title="Residuals")

plt.tight_layout()
plt.show()

# Plot autocorrelation function to check seasonality
plot_acf(df_cleaned["Ontario Demand"], lags=168)  # Checking for weekly seasonality (7 days * 24 hours)
plt.title("Autocorrelation of Electricity Demand")
plt.show()

#AR model

def get_Lh(hour):
    if hour <= 7:
        return [24, 48, 72]
    else:
        return [48, 72, 96]

def seasonal_ar_forecast(y_history, t, hour, alpha_lags, alpha_weekly):
    """
    y_history: array-like, time series of past load values
    t: int, current time index
    hour: int, hour of the day for current t (0 to 23)
    alpha_lags: dict, coefficients for daily lags (e.g., {24: a1, 48: a2, ...})
    alpha_weekly: dict, coefficients for weekly lags (e.g., {168: b1, ...})
    """
    Lh = get_Lh(hour)
    
    # Daily lags
    daily_sum = sum(alpha_lags[l] * y_history[t - l] for l in Lh) 
    # Weekly seasonal lags (1 to 6 weeks)
    weekly_sum = sum(alpha_weekly[168 * l] * y_history[t - 168 * l] for l in range(1, 7))
    
    return daily_sum + weekly_sum  # ε_t is not modeled directly here

