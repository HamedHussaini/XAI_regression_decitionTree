import pandas as pd

# === 1. Les datasettet ===
df = pd.read_csv("./future_brazil.csv")

print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns\n")

# === 2. Sjekk alle typer manglende verdier (NaN, tomme strenger, None) ===
missing_mask = df.isna() | df.eq('') | df.eq(' ')

# === 3. Finn alle rader som har minst én manglende verdi ===
rows_with_missing = missing_mask.any(axis=1)

if rows_with_missing.sum() == 0:
    print("🎉 Ingen manglende verdier funnet i noen kolonner eller rader!")
else:
    print(f"⚠️ Fant {rows_with_missing.sum()} rader med manglende verdier.\n")

    # === 4. Vis detaljert oversikt per rad ===
    for idx, row in df[rows_with_missing].iterrows():
        missing_cols = [col for col in df.columns if pd.isna(row[col]) or str(row[col]).strip() == ""]
        print(f"🔸 Rad {idx}: mangler verdier i kolonner → {missing_cols}")

# === 5. (Valgfritt) Lagre alle rader med problemer til en CSV for inspeksjon ===
# df[rows_with_missing].to_csv("rows_with_missing.csv", index=False)
# print("\n💾 Lagret rader med manglende verdier til 'rows_with_missing.csv'")
