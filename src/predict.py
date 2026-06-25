import sys
import joblib
import numpy as np
import sys

from feature_extraction import extract_features

model = joblib.load(
    "models/key_detector.pkl"
)


if len(sys.argv) > 1:
    audio_file = sys.argv[1]
else:
    audio_file = "data/21 Savage - redrum (Official Audio).mp3"

features = extract_features(
    audio_file
)

features = np.array(
    features
).reshape(1, -1)

prediction = model.predict(
    features
)[0]
confidence = np.max(
    model.predict_proba(features)
)

print()
print("Predicted Key:")
print(prediction)

print()
print(f"Confidence: {confidence:.2%}")