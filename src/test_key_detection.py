from pathlib import Path

from key_detection import (
    get_chroma,
    detect_key
)

file = next(
    Path("data/fma_small").rglob("*.mp3")
)

chroma = get_chroma(file)

key, mode, score = detect_key(chroma)

print(
    f"Predicted Key: {key} {mode}"
)

print(
    f"Confidence: {score:.3f}"
)