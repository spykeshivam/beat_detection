import numpy as np
import matplotlib.pyplot as plt
import librosa
import torchaudio

# Load the data from data.npy
data = np.load('../Dataset/BallroomAnnotations-master/audio/Albums-Latin_Jam2-12.npy')

# Print the data
#print(data)
def plot_log_mel(log_mel, title="Log-Mel Spectrogram", sr=44100, hop_length=441):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(log_mel, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()

plot_log_mel(data)