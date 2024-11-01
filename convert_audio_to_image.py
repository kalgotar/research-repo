import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Function to convert audio to image
def audio_to_image(audio_path, img_path):
    y, sr = librosa.load(audio_path, sr=None)
    chromagram = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=512, cmap='coolwarm')
    plt.axis('off')
    plt.savefig(img_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# Convert all audio files to images
audio_dir = r'\\audio_data'
image_dir = r'\\image_data'
os.makedirs(image_dir, exist_ok=True)

file_paths = []
labels = []

for label in ['good', 'bad']:
    audio_folder = os.path.join(audio_dir, label)
    for audio_file in os.listdir(audio_folder):
        if audio_file.endswith('.wav'):
            audio_path = os.path.join(audio_folder, audio_file)
            img_path = os.path.join(image_dir, f"{label}_{audio_file.replace('.wav', '.png')}")
            audio_to_image(audio_path, img_path)
            file_paths.append(img_path)
            labels.append(label)

# Encode labels
le = LabelEncoder()
labels = le.fit_transform(labels)