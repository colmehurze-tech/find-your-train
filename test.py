import pandas as pd

df = pd.read_csv(
    "data/new_delhi_rajdhani_express_synthetic_dataset.csv"
)

print("Rows:", len(df))
print("\nDelay causes:")
print(df["Cause of Delay (if any)"].value_counts(dropna=False))
print("\nNumber of unique causes:")
print(df["Cause of Delay (if any)"].nunique(dropna=False))