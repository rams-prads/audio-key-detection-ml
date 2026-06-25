import pandas as pd

df = pd.read_csv(
    "labels/labels_8000.csv"
)

print(df["key"].value_counts())

print()

print(df["mode"].value_counts())