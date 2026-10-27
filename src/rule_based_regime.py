import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("data/nifty50_features.csv", parse_dates=["Date"])

df=df.sort_values("Date")

volatility_threshold=df["volatility_20"].median()
print(volatility_threshold)

def detect_regime(row):
    momentum=row["momentum_20"]
    volatility=row["volatility_20"]

    if momentum>0.01 and volatility<volatility_threshold:
        return "bullish calm"

    elif momentum>0.01 and volatility>volatility_threshold:
        return "bullish volatile"

    elif momentum<-0.01 and volatility>volatility_threshold:
        return "bear/crisis"

    elif momentum<-0.01:
        return "bear"

    else:
        return "sideways"

df['regime'] = df.apply(detect_regime, axis=1)

print(df["regime"].value_counts())

stats=df.groupby("regime").agg(
    avg_return=("return", "mean"),
    avg_volatility=("volatility_20", "mean"),
    avg_momentum=("momentum_20", "mean"),
    avg_drawdown=("drawdown", "mean"),
    days=("regime", "size")

)


print(stats)



plt.figure(figsize=(14, 6))

for regime in df["regime"].unique():

    temp = df[df["regime"] == regime]

    plt.scatter(
        temp["Date"],
        temp["Close"],
        label=regime,
        s=15
    )

plt.title("Nifty 50 - Rule Based Market Regimes")
plt.xlabel("Date")
plt.ylabel("Nifty 50")
plt.legend()
plt.grid()
plt.show()

df.to_csv("data/nifty50_rule_regimes.csv", index=False)