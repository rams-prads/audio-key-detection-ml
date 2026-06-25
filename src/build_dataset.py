from pathlib import Path
import pandas as pd
import numpy as np

from feature_extraction import extract_features

labels = pd.read_csv(
    "labels/labels_8000.csv"
)

X = []
y = []

labels = labels[labels["confidence"] > 0.7]

for _, row in labels.iterrows():

    try:

        filename = row["file"]

        file_path = next(
            Path("data/fma_small").rglob(filename)
        )

        features = extract_features(file_path)

        X.append(features)

        y.append(
            row["key"] + "_" + row["mode"]
        )

    except Exception as e:

        print(f"Skipping {filename}")
        print(e)

X = np.array(X)
y = np.array(y)

print("X Shape:", X.shape)

print("y Shape:", y.shape)

np.save("data/X.npy", X)

np.save("data/y.npy", y)

print("Dataset Saved")