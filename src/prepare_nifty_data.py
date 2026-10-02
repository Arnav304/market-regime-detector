import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. LOAD DATA
# ============================================================

# Change this path according to where your CSV is stored
FILE_PATH = "data/nifty50.csv"

df = pd.read_csv(FILE_PATH)
print(df.columns.tolist())

print("\nOriginal Data:")
print(df.head())

print("\nColumns:")
print(df.columns)

print("\nShape:")
print(df.shape)

print("\nMissing values:")
print(df.isnull().sum())


# ============================================================
# 2. CLEAN BASIC DATA
# ============================================================

# Remove accidental spaces from column names
df.columns = df.columns.str.strip()

# Convert Date column from strings like 01-OCT-2026
# into actual datetime values
df["Date"] = pd.to_datetime(
    df["Date"],
    format="%d-%b-%Y"
)

# Sort from oldest date to newest date
df = df.sort_values("Date")

# Make Date the dataframe index
df.set_index("Date", inplace=True)

# Make sure Close contains numbers
df["Close"] = pd.to_numeric(
    df["Close"],
    errors="coerce"
)

# Remove rows where Close is missing
df = df.dropna(subset=["Close"])

print("\nCleaned Data:")
print(df.head())

print("\nLast rows:")
print(df.tail())

# ============================================================
# 3. PLOT NIFTY 50 PRICE
# ============================================================

plt.figure(figsize=(12, 5))

plt.plot(df.index, df["Close"])

plt.title("Nifty 50 Price")
plt.xlabel("Date")
plt.ylabel("Close Price")

plt.grid()

plt.show()


# ============================================================
# 4. DAILY RETURNS
# ============================================================

# Percentage change compared with previous trading day
df["return"] = df["Close"].pct_change()

print("\nDaily Returns:")
print(df[["Close", "return"]].head(10))


# Plot daily returns
plt.figure(figsize=(12, 4))

plt.plot(df.index, df["return"])

plt.axhline(0)

plt.title("Nifty 50 Daily Returns")
plt.xlabel("Date")
plt.ylabel("Daily Return")

plt.grid()

plt.show()


# ============================================================
# 5. 20-DAY ROLLING VOLATILITY
# ============================================================

# Standard deviation of previous 20 trading-day returns.
#
# sqrt(252) converts daily volatility into approximately
# annualized volatility because a year has about
# 252 trading days.

df["volatility_20"] = (
    df["return"]
    .rolling(window=20)
    .std()
    * np.sqrt(252)
)


plt.figure(figsize=(12, 4))

plt.plot(df.index, df["volatility_20"])

plt.title("20-Day Rolling Annualized Volatility")
plt.xlabel("Date")
plt.ylabel("Volatility")

plt.grid()

plt.show()


# ============================================================
# 6. 20-DAY MOMENTUM
# ============================================================

# Compare today's closing price with the price
# approximately 20 trading days ago.
#
# Positive momentum:
# Market has gone up over the last 20 days.
#
# Negative momentum:
# Market has gone down over the last 20 days.

df["momentum_20"] = df["Close"].pct_change(periods=20)


plt.figure(figsize=(12, 4))

plt.plot(df.index, df["momentum_20"])

plt.axhline(0)

plt.title("20-Day Momentum")
plt.xlabel("Date")
plt.ylabel("Momentum")

plt.grid()

plt.show()


# ============================================================
# 7. DRAWDOWN
# ============================================================

# Highest price observed up to each date
df["peak"] = df["Close"].cummax()

# Percentage decline from the previous peak
df["drawdown"] = (df["Close"] / df["peak"]) - 1


plt.figure(figsize=(12, 4))

plt.plot(df.index, df["drawdown"])

plt.axhline(0)

plt.title("Nifty 50 Drawdown")
plt.xlabel("Date")
plt.ylabel("Drawdown")

plt.grid()

plt.show()


# ============================================================
# 8. OPTIONAL MOVING AVERAGES
# ============================================================

# Useful later for regime detection

df["ma_20"] = df["Close"].rolling(window=20).mean()
df["ma_50"] = df["Close"].rolling(window=50).mean()


plt.figure(figsize=(12, 5))

plt.plot(
    df.index,
    df["Close"],
    label="Nifty 50"
)

plt.plot(
    df.index,
    df["ma_20"],
    label="20-Day MA"
)

plt.plot(
    df.index,
    df["ma_50"],
    label="50-Day MA"
)

plt.title("Nifty 50 with Moving Averages")
plt.xlabel("Date")
plt.ylabel("Price")

plt.legend()
plt.grid()

plt.show()


# ============================================================
# 9. VIEW ALL FEATURES
# ============================================================

feature_columns = [
    "Close",
    "return",
    "volatility_20",
    "momentum_20",
    "drawdown",
    "ma_20",
    "ma_50"
]

print("\nDataset with Features:")
print(df[feature_columns].tail(20))


# ============================================================
# 10. REMOVE NaN VALUES CREATED BY ROLLING WINDOWS
# ============================================================

# The first few rows will contain NaN because:
#
# 20-day volatility needs 20 days of previous data.
# 50-day MA needs 50 days of previous data.

df_clean = df.dropna().copy()

print("\nClean Dataset Shape:")
print(df_clean.shape)

print("\nFinal Dataset:")
print(df_clean[feature_columns].head())


# ============================================================
# 11. BASIC STATISTICS
# ============================================================

print("\nFeature Statistics:")

print(
    df_clean[
        [
            "return",
            "volatility_20",
            "momentum_20",
            "drawdown"
        ]
    ].describe()
)


# ============================================================
# 12. SAVE PROCESSED DATA
# ============================================================

OUTPUT_PATH = "data/nifty50_features.csv"

df_clean.to_csv(OUTPUT_PATH)

print(
    f"\nProcessed data successfully saved to: "
    f"{OUTPUT_PATH}"
)


# ============================================================
# 13. LATEST MARKET INFORMATION
# ============================================================

latest = df_clean.iloc[-1]

print("\n==============================")
print("LATEST MARKET DATA")
print("==============================")

print(
    f"Close: "
    f"{latest['Close']:.2f}"
)

print(
    f"Daily Return: "
    f"{latest['return'] * 100:.2f}%"
)

print(
    f"20-Day Volatility: "
    f"{latest['volatility_20'] * 100:.2f}%"
)

print(
    f"20-Day Momentum: "
    f"{latest['momentum_20'] * 100:.2f}%"
)

print(
    f"Drawdown: "
    f"{latest['drawdown'] * 100:.2f}%"
)

print("==============================")