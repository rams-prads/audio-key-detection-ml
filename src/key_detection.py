import numpy as np

MAJOR_PROFILE = np.array([
    6.35,
    2.23,
    3.48,
    2.33,
    4.38,
    4.09,
    2.52,
    5.19,
    2.39,
    3.66,
    2.29,
    2.88
])

MINOR_PROFILE = np.array([
    6.33,
    2.68,
    3.52,
    5.38,
    2.60,
    3.53,
    2.54,
    4.75,
    3.98,
    2.69,
    3.34,
    3.17
])

NOTES = [
    "C","C#","D","D#",
    "E","F","F#","G",
    "G#","A","A#","B"
]

import librosa

def get_chroma(audio_path):

    y, sr = librosa.load(
        audio_path,
        duration=10
    )

    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=sr
    )

    return chroma.mean(axis=1)

def detect_key(chroma):

    best_score = -999

    best_key = None

    best_mode = None

    for shift in range(12):

        major_profile = np.roll(
            MAJOR_PROFILE,
            shift
        )

        major_corr = np.corrcoef(
            chroma,
            major_profile
        )[0,1]

        if major_corr > best_score:

            best_score = major_corr

            best_key = NOTES[shift]

            best_mode = "Major"

        minor_profile = np.roll(
            MINOR_PROFILE,
            shift
        )

        minor_corr = np.corrcoef(
            chroma,
            minor_profile
        )[0,1]

        if minor_corr > best_score:

            best_score = minor_corr

            best_key = NOTES[shift]

            best_mode = "Minor"

    return best_key, best_mode, best_score

