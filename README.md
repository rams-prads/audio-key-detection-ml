# Audio Key Detection using Machine Learning

## Overview

This project predicts the musical key of an audio file using classical machine learning techniques and audio feature extraction.

The idea behind the project was to explore Music Information Retrieval (MIR) and understand how machine learning can be applied to real-world audio analysis. Given an MP3 or WAV file, the system extracts a set of harmonic and spectral features from a 10-second audio clip and predicts one of the 24 musical keys (12 major and 12 minor).

The project was built from scratch using Python, Librosa and Scikit-learn.

---

## Motivation

As someone interested in both machine learning and music production, I wanted to build something that combined the two. Most beginner machine learning projects involve tabular datasets or images, but audio processing introduces a completely different workflow involving signal processing, feature engineering and Music Information Retrieval.

This project helped me understand how musical information can be represented numerically and used by a machine learning model.

---

## Dataset

The model was trained using the **FMA Small** dataset, which contains approximately 8,000 songs across multiple genres.

The dataset itself is not included in this repository because of GitHub size limitations.

---

## Feature Extraction

Each audio file is trimmed to a 10-second segment before feature extraction.

The following features are extracted using Librosa:

* MFCCs
* Chroma Features
* Tonnetz Features
* Spectral Contrast
* Spectral Centroid
* Spectral Bandwidth
* Spectral Rolloff
* Zero Crossing Rate

These features are combined into a 42-dimensional feature vector which is used for training and prediction.

---

## Model

The current implementation uses a Random Forest classifier.

Training is performed using an 80-20 train-test split.

After experimenting with different feature combinations and increasing the dataset size to approximately 8,000 songs, the final model achieved an accuracy of around **77%** on the test set.

---

## Example

```
Input:
21 Savage - redrum.mp3

Prediction:
D Major

Confidence:
48%
```

---

## Project Structure

```
audio-key-detection-ml
│
├── src
│   ├── feature_extraction.py
│   ├── build_dataset.py
│   ├── label_generator.py
│   ├── train_model.py
│   ├── predict.py
│   └── key_detection.py
│
├── data
├── models
├── requirements.txt
└── README.md
```

---

## Running the Project

Install the required packages

```bash
pip install -r requirements.txt
```

Train the model

```bash
python src/train_model.py
```

Predict the key of an audio file

```bash
python src/predict.py path/to/audio.mp3
```

---

## Future Improvements

Some ideas I plan to work on next are:

* BPM detection
* Chord progression detection
* Improving accuracy using deep learning models
* A simple web interface for uploading audio files

---

## What I Learned

Working on this project gave me practical experience with:

* Audio signal processing
* Feature engineering
* Music Information Retrieval
* Building datasets for machine learning
* Training and evaluating classification models
* Organizing a machine learning project from data collection to deployment

Although there are more accurate approaches based on deep learning, my goal for this project was to first build a complete end-to-end pipeline using classical machine learning and understand every step involved.
