#Music Genre Classification using Deep 

A robust Deep Learning pipeline that classifies audio tracks into 10 distinct genres using Convolutional Neural Networks (CNNs) and Mel-spectrogram analysis.

🚀 Project Overview
This project transforms raw audio files into visual representations (Mel-spectrograms) to leverage the power of Convolutional Neural Networks (CNNs) for genre classification.
By extracting Mel-Frequency Cepstral Coefficients (MFCCs), the model effectively captures both timbral and rhythmic characteristics of different music genres.

📊 Performance Metrics

The model achieved an 85% overall accuracy on the test set

  Metric                       Value
Test Accuracy                   85%
Best Performing         ClassClassical (0.97 F1-score)
Frameworks              PyTorch, Librosa, Scikit-Learn
       
Confusion Matrix:
<img width="500" height="500" alt="confusion_matrix" src="https://github.com/user-attachments/assets/b45a0329-0a2c-487c-98f1-ec26d3fa326b" />


🛠 Technical Pipeline
Preprocessing: Converted raw .wav files into Mel-spectrograms using librosa to visualize audio frequencies over time.
Feature Engineering: Extracted MFCCs to highlight the most discriminative features for genre recognition.
Model Architecture: Implemented a CNN in PyTorch, optimized through hyperparameter tuning and dropout layers to mitigate overfitting.

🚀 How to RunBash# Clone the repository
git clone https://github.com/ishaanisingh/music-genre-classification.git

# Install requirements
pip install -r requirements.txt

# Run inference on a new audio file
python predict.py --audio_path "path/to/your/song.wav"
