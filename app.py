import os 
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import streamlit as st
import torch
import torch.nn as nn
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

# The architecture must match your AudioCNN class exactly
class AudioCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(AudioCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4) 
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), 
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = nn.functional.adaptive_avg_pool2d(x, (4, 4)) 
        x = self.fc_layers(x)
        return x

# Helper to process audio identical to your GTZANDataset class
def process_audio(audio_path):
    num_samples = 22050 * 30
    signal, sr = librosa.load(audio_path, sr=22050)
    if len(signal) > num_samples:
        signal = signal[:num_samples]
    else:
        padding = num_samples - len(signal)
        signal = np.pad(signal, (0, padding), 'constant')
    mel_spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    input_tensor = torch.tensor(mel_spec_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return signal, mel_spec_db, input_tensor

# UI Layout
st.title("🎵 Music Genre Classifier")
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file)
    signal, mel_spec, input_tensor = process_audio(uploaded_file)
    
    # Load model
    model = AudioCNN(num_classes=10)
    model.load_state_dict(torch.load('music_genre_cnn.pth', map_location='cpu'))
    model.eval()
    
    # Predict
    output = model(input_tensor)
    probs = torch.nn.functional.softmax(output, dim=1)
    score, index = torch.max(probs, 1)
    
    st.subheader(f"Prediction: {GENRES[index].upper()}")
    st.write(f"Confidence: {score.item()*100:.2f}%")
    
    # Visualizations
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    librosa.display.waveshow(signal, sr=22050, ax=ax[0])
    librosa.display.specshow(mel_spec, sr=22050, x_axis='time', y_axis='mel', ax=ax[1])
    st.pyplot(fig)