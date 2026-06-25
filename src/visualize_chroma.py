from pathlib import Path
import librosa
import librosa.display
import matplotlib.pyplot as plt

file = next(
    Path("data/fma_small").rglob("*.mp3")
)

y, sr = librosa.load(
    file,
    duration=10
)

chroma = librosa.feature.chroma_stft(
    y=y,
    sr=sr
)

plt.figure(figsize=(12,4))

librosa.display.specshow(
    chroma,
    x_axis="time",
    y_axis="chroma"
)

plt.colorbar()

plt.title("Chroma Features")

plt.show()