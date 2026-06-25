import numpy as np
import joblib 

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report
)

X = np.load(
    "data/X.npy"
)

y = np.load(
    "data/y.npy",
    allow_pickle=True
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

preds = model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    preds
)

print(
    classification_report(
        y_test,
        preds
    )
)

print(
    f"Accuracy: {accuracy:.3f}"
)
joblib.dump(
    model,
    "models/key_detector.pkl"
)

print("Model saved successfully!")