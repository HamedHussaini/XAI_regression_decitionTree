import pandas as pd

df = pd.read_csv("brazil.csv")
if "index" in df.columns:
    df = df.drop(columns=["index"])

df["time_period"] = pd.PeriodIndex(df["time_period"], freq="M")
df = df.sort_values(["location", "time_period"]).reset_index(drop=True)

# global 1..N
df.insert(0, "index", df.index + 1)

split_period = pd.Period("2016-12", freq="M")
historic = df[df["time_period"] <= split_period].copy()
future   = df[df["time_period"]  > split_period].copy()

historic.to_csv("historic_brazil.csv", index=False)
future.to_csv("future_brazil.csv", index=False)
