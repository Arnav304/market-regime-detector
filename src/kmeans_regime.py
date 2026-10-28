import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


df = pd.read_csv("data/nifty50_features.csv")

features = [
    "momentum_20",
    "volatility_20",
    "drawdown"
]

X = df[features]

print(X.head())



Scaler = StandardScaler()

X_scaled = Scaler.fit_transform(X)

print("Original Data: ")
print(X.head())

print("\nScaled Data: ")
print(X_scaled[:5])



kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

kmeans.fit(X_scaled)

print("K-Means has finished clustering!")

df["cluster"] = kmeans.labels_

print(df[["Date", "momentum_20", "volatility_20", "drawdown", "cluster"]].head(10))



cluster_summary = df.groupby("cluster")[
    ["momentum_20", "volatility_20", "drawdown"]
].mean()

print(cluster_summary)

print(df["cluster"].value_counts().sort_index())



regime_names = {
    0: "calm/neutral",
    1: "bearish",
    2: "high volatility Bullish"
}

df["kmeans_regime"] = df["cluster"].map(regime_names)

print(df[["Date", "cluster", "kmeans_regime"]].head(10))


rule_df = pd.read_csv("data/nifty50_rule_regimes.csv")


comparison = df[["Date", "kmeans_regime"]].merge(
    rule_df[["Date", "regime"]],
    on="Date",
    how="inner"
)


print(comparison.head(10))

print("\nComparison table:")
print(pd.crosstab(
    comparison["kmeans_regime"],
    comparison["regime"]
))


