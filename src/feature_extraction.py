import librosa
import numpy as np

def extract_features(audio_path):

    y, sr = librosa.load(
        audio_path,
        duration=10,
        mono=True
    )

    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=sr
    )

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13
    )

    centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr
    )

    contrast = librosa.feature.spectral_contrast(
        y=y,
        sr=sr
    )

    tonnetz = librosa.feature.tonnetz(
        y=librosa.effects.harmonic(y),
        sr=sr
    )

    zcr = librosa.feature.zero_crossing_rate(
        y
    )

    rolloff = librosa.feature.spectral_rolloff(
        y=y,
        sr=sr
    )

    bandwidth = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr
    )

    features = np.concatenate([
        chroma.mean(axis=1),      # 12
        mfcc.mean(axis=1),        # 13
        contrast.mean(axis=1),    # 7
        tonnetz.mean(axis=1),     # 6
        centroid.mean(axis=1),    # 1
        zcr.mean(axis=1),         # 1
        rolloff.mean(axis=1),     # 1
        bandwidth.mean(axis=1)    # 1
    ])

    return features