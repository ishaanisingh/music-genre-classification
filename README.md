Music Genre Classification using Deep LearningA robust Deep Learning pipeline that classifies audio tracks into 10 distinct genres using Convolutional Neural Networks (CNNs) and Mel-spectrogram analysis.🚀 Project OverviewThis project transforms raw audio files into visual representations (Mel-spectrograms) to leverage the power of Convolutional Neural Networks (CNNs) for genre classification. By extracting Mel-Frequency Cepstral Coefficients (MFCCs), the model effectively captures both timbral and rhythmic characteristics of different music genres.📊 Performance MetricsThe model achieved an 85% overall accuracy on the test set.MetricValueTest Accuracy85%Best Performing ClassClassical (0.97 F1-score)FrameworksPyTorch, Librosa, Scikit-LearnConfusion Matrix:<img width="1000" height="800" alt="confusion_matrix" src="https://github.com/user-attachments/assets/381d214a-5c19-4d10-99f5-ff2452ac0fd8" />
🛠 Technical PipelinePreprocessing: Converted raw .wav files into Mel-spectrograms using librosa to visualize audio frequencies over time.Feature Engineering: Extracted MFCCs to highlight the most discriminative features for genre recognition.Model Architecture: Implemented a CNN in PyTorch, optimized through hyperparameter tuning and dropout layers to mitigate overfitting.🚀 How to RunBash# Clone the repository
git clone https://github.com/ishaanisingh/music-genre-classification.git

# Install requirements
pip install -r requirements.txt

# Run inference on a new audio file
python predict.py --audio_path "path/to/your/song.wav"
