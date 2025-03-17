import os
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from torchaudio.transforms import MelSpectrogram
import torchaudio
import torch

def extract_features(audio_path, sr=44100, n_fft=2048, hop_length=441, n_mels=81):
    """
    Extract log-mel spectrogram from an audio file.
    """
    # Load audio
    trim_duration=29
    """
    Help of ai(Deepseek R1) was taken in selecting the parameters of the MelSpectrogram
    center=True,
    pad_mode="reflect",
    power=2.0,
    norm="slaney"
    Purpose : Enhance the energy of the spectrogram and appropriate padding in the beginning and end
    """
    melspec=MelSpectrogram(
            sample_rate=44100,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm="slaney",
            n_mels=n_mels,
        )
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != sr:
        waveform = torchaudio.functional.resample(waveform, sample_rate, sr)
    
    trim_samples = int(trim_duration * sr)
    if waveform.shape[1] > trim_samples:
        waveform = waveform[:, :trim_samples]
    
    mel_spectrogram = melspec(waveform)
    log_mel_spectrogram = torch.log10(mel_spectrogram + 1e-10)
    log_mel_np = log_mel_spectrogram.squeeze().numpy()
    print(log_mel_np.shape)
    
    
    return log_mel_np

def plot_log_mel(log_mel, title="Log-Mel Spectrogram", sr=44100, hop_length=441):
    """
    Plot the log-mel spectrogram.
    """
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(log_mel, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def preprocess_audio_folder(audio_folder, output_folder, sr=44100, n_mels=81, hop_length=441):
    """
    Preprocess all audio files in a folder and save the features.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    features_dict = {}
    print('E###########',audio_folder)
    for root, _, files in os.walk(audio_folder):
        #print(files,root,_,'files')
        #genre = os.path.basename(root)

        #Checking each folder for .wav files
        for file in files:
            #print(file)
            if file.endswith('.wav'):
                audio_path = os.path.join(root, file)
                basename = os.path.splitext(file)[0]
                print(f"Processing: {basename}")
                #print(file)
                
                # Extract features
                features = extract_features(audio_path, sr=sr, n_mels=n_mels, hop_length=hop_length)
                
                # Save features
                relative_path = os.path.relpath(root, audio_folder)
                save_folder = os.path.join(output_folder, relative_path)
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                
                output_path = os.path.join(save_folder, file.replace('.wav', '.npy'))
                np.save(output_path, features)
                features_dict[basename] = features
                #Plot the spectogram to manually check it is correct
                if file == "Albums-Ballroom_Magic-03.wav":
                    plot_log_mel(features, title=f"Log-Mel Spectrogram: {file}")

if __name__ == "__main__":
    # Paths
    audio_folder = "../Dataset/BallroomData"
    output_folder = "../Dataset/BallroomAnnotations-master"
    
    preprocess_audio_folder(audio_folder, output_folder)