# Speech Emotion Recognition (SER)

## Project Overview
Speech Emotion Recognition (SER) is a machine learning project that identifies human emotions from speech recordings using audio signal processing and deep learning.

The project uses the RAVDESS speech dataset and extracts MFCC-based audio features to classify speech into eight emotion categories.

## Objectives
- Analyze speech recordings and extract meaningful audio features.
- Identify emotions from voice recordings.
- Train and compare ANN and CNN models.
- Evaluate model performance on unseen speakers.
- Predict emotions from new audio recordings.

## Dataset
**RAVDESS — Ryerson Audio-Visual Database of Emotional Speech and Song**

This project uses the speech audio dataset containing 1,440 recordings from 24 actors.

Emotion categories:
1. Neutral
2. Calm
3. Happy
4. Sad
5. Angry
6. Fearful
7. Disgust
8. Surprised

Dataset source: https://zenodo.org/records/11063852

## Technologies Used
- Python
- TensorFlow / Keras
- Librosa
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

## Methodology

### 1. Audio Preprocessing
- Load audio recordings.
- Resample audio to 16 kHz.
- Extract audio features using Librosa.

### 2. Feature Extraction
The improved feature extraction approach uses:
- MFCC mean (40 features)
- MFCC standard deviation (40 features)
- Delta MFCC mean (40 features)
- Delta-Delta MFCC mean (40 features)

Total: 160 features per audio recording.

### 3. Model Development
The project experiments with:
- Artificial Neural Network (ANN)
- Convolutional Neural Network (CNN)

### 4. Model Training
- Feature scaling using StandardScaler.
- Training using the Adam optimizer.
- Sparse Categorical Crossentropy loss.
- Dropout regularization.
- EarlyStopping to control overfitting.

### 5. Evaluation
Models are evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

An actor-independent evaluation is also performed to test performance on speakers excluded from training.

## Experimental Results

| Model | Evaluation | Accuracy |
|---|---|---:|
| Original ANN | Random split | 54.86% |
| Original CNN | Random split | 57.64% |
| CNN | Actor-independent | 36% |
| Improved ANN | Actor-independent | 41% |

The improved ANN achieved 41% accuracy on the actor-independent test set of 300 recordings.

Note: Random-split and actor-independent results use different evaluation protocols and should not be directly compared as equivalent tests.

## Model Prediction
The trained model can process a new speech recording and return:
- Predicted emotion
- Prediction score

Saved model artifacts:
- `emotion_model.keras`
- `emotion_scaler.pkl`
- `label_encoder.pkl`

## Installation

Clone the repository:

```bash
git clone https://github.com/Karthikaraja-ai/Speech-Emotion-Recognition.git
cd Speech-Emotion-Recognition
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open `emotion_recognition.ipynb` and run the notebook cells.

## Future Improvements
- Improve accuracy on unseen speakers.
- Experiment with LSTM and hybrid CNN-LSTM architectures.
- Explore additional speech emotion datasets.
- Improve robustness across different recording conditions.
- Develop a user-friendly interface for audio prediction.

## Project Status
Core model development, evaluation and saved-model prediction pipeline completed.

This project was developed as part of a Machine Learning Internship.

## Author
Karthika Raja