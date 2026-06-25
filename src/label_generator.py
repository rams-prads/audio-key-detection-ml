from pathlib import Path
import pandas as pd
from tqdm import tqdm

from key_detection import (
    get_chroma,
    detect_key
)

audio_files = list(
    Path("data/fma_small").rglob("*.mp3")
)

print(f"Found {len(audio_files)} songs")

audio_files = audio_files[:8000]

rows = []

for file in tqdm(audio_files):

    try:

        chroma = get_chroma(file)

        key, mode, confidence = detect_key(chroma)

        rows.append({
            "file": file.name,
            "key": key,
            "mode": mode,
            "confidence": confidence
        })

    except Exception as e:

        print(f"Error: {file}")
        print(e)

df = pd.DataFrame(rows)

df.to_csv(
    "labels/labels_8000.csv",
    index=False
)

print(df.head())

print("\nSaved labels!")