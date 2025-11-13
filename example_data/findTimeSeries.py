import pandas as pd

df = pd.read_csv("./future_brazil.csv")

counts = df.groupby("location")["time_period"].nunique().sort_values()
print("📊 Antall tidsperioder per lokasjon:\n")
print(counts)

min_count = counts.min()
max_count = counts.max()
print(f"\n➡️ Min periods per location: {min_count}, Max: {max_count}")

if min_count < max_count:
    bad_locs = counts[counts < max_count].index.tolist()
    print(f"\n⚠️ Lokasjoner med færre tidsperioder enn resten:")
    for loc in bad_locs:
        print(f"   - {loc} ({counts[loc]} perioder)")
else:
    print("\n✅ Alle lokasjoner har konsistent antall tidsperioder!")
