from pathlib import Path

from feature_extraction import extract_features

file = next(
    Path("data/fma_small").rglob("*.mp3")
)

features = extract_features(file)

print(features.shape)
print(features)