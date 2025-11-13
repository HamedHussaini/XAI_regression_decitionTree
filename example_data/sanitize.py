import pandas as pd

# === 1. Les filen ===
df = pd.read_csv("./future_brazil.csv")

# === 2. Sjekk for manglende verdier ===
print("Missing values per column:")
print(df.isna().sum())

# === 3. Fjern rader med manglende verdier i nøkkelfelt ===
df = df.dropna(subset=["rainfall", "mean_temperature", "location", "time_period"])

# === 4. Fjern duplikater (hvis noen) ===
df = df.drop_duplicates(subset=["location", "time_period"])

# === 5. Sjekk at alle lokasjoner har like mange tidsperioder ===
counts = df.groupby("location")["time_period"].nunique()
invalid_locs = counts[counts < counts.max()].index.tolist()

if invalid_locs:
    print(f"⚠️ Fjerner ufullstendige lokasjoner: {invalid_locs}")
    df = df[~df["location"].isin(invalid_locs)]

# === 6. Sorter og lagre ===
df = df.sort_values(["location", "time_period"])
df.to_csv("./future_brazil.csv", index=False)
print("✅ Saved cleaned file to example_data/future_brazil_clean.csv")
