from pathlib import Path
from feature_extraction import extract_features

import numpy as np
from tqdm import tqdm

audio_files = list(
    Path("data/fma_small").rglob("*.mp3")
)

print(f"Found {len(audio_files)} files")

# Only first 100 for now
audio_files = audio_files[:100]

X = []

for file in tqdm(audio_files):

    try:
        features = extract_features(file)
        X.append(features)

    except Exception as e:
        print(f"Error: {file}")
        print(e)

X = np.array(X)

print("Dataset Shape:", X.shape)

np.save(
    "data/X_100.npy",
    X
)

print("Saved!")