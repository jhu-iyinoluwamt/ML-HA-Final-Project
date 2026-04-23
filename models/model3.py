#%% md
# ## Model 3: multimodal fusion
#%%
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, recall_score, f1_score, precision_score, \
    precision_recall_curve
import tensorflow as tf
from sklearn.utils import resample
from tensorflow import keras
layers = keras.layers
models = keras.models

import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


BASE_PATH = '1.0.3/training_data'
SAMPLE_RATE = 4000
N_MELS = 128
HOP_LENGTH = 256
N_FFT = 1024
DURATION = 5  # seconds
TARGET_SHAPE = (N_MELS, int(SAMPLE_RATE * DURATION / HOP_LENGTH) + 1)
#%%
cohort = pd.read_csv('cohort.csv')
print(f"Total patients: {len(cohort)}")
print(f"Murmur distribution:\n{cohort['Murmur'].value_counts()}")
print(f"Outcome distribution:\n{cohort['Outcome'].value_counts()}")