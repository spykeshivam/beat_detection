import os
import librosa
import numpy as np
from pathlib import Path


"""
Parts of the functions text_label_to_float,get_quantised_ground_truth,load_spectrogram_and_labels in this code were taken from :
https://github.com/ben-hayes/beat-tracking-tcn/blob/master/beat_tracking_tcn/datasets/ballroom_dataset.py
"""
sr=44100
n_fft=2048
hop_length=441
n_mels=81

def text_label_to_float(text):
        """Exracts beat time from a text line and converts to a float"""
        allowed = '1234567890. \t'
        filtered = ''.join([c for c in text if c in allowed])
        if '\t' in filtered:
            t = filtered.rstrip('\n').split('\t')
        else:
            t = filtered.rstrip('\n').split(' ')
        return float(t[0]), float(t[1])

def get_quantised_ground_truth(annotation_path, hop_size, get_downbeats=False):
        with open(annotation_path, "r") as f:
            
            beat_times_secs = []
            downbeat_times_secs = []
            beat_times = []
            downbeat_times = []
            
            for line in f:
                time, beat_position = text_label_to_float(line)
                beat_times.append(time * sr)
                beat_times_secs.append(time)
                
                # Check if this is a downbeat (position 1)
                if beat_position is not None and beat_position == 1:
                    downbeat_times.append(time * sr)
                    downbeat_times_secs.append(time)

        quantised_times = []
        quantised_downbeat_times = []
        
        # Quantize beat times
        for time in beat_times:
            spec_frame = int(time / hop_size)
            quantised_time = spec_frame * hop_size / sr
            quantised_times.append(quantised_time)
        
        # Quantize downbeat times if requested
        if get_downbeats:
            for time in downbeat_times:
                spec_frame = int(time / hop_size)
                quantised_time = spec_frame * hop_size / sr
                quantised_downbeat_times.append(quantised_time)
            
            return np.array(quantised_times), np.array(quantised_downbeat_times)
        
        return np.array(quantised_times)

def load_annotations_beat(audio_dir, annotation_dir, sr=44100, hop_size_in_seconds =0.01):
    """
    Load and process beat annotations from .beats files.
    """
    import time
    hop_size = int(np.floor(hop_size_in_seconds * 44100))
    # Get list of audio files
    audio_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith(".wav")]
    file_names = [Path(f).stem for f in audio_files]

    # Initialize dictionaries
    beat_times = {}

    #print("n_spec_frames",n_spec_frames)

    # Process each file
    for filename in file_names:
        annotation_path = Path(annotation_dir) / (filename + ".beats")
        #print(filename)
        
        # Load beat times from annotation file
        quantised_ground_truth=get_quantised_ground_truth(annotation_path,hop_size)
        #print(quantised_ground_truth,len(quantised_ground_truth))
    
        beat_times[filename] = quantised_ground_truth
    return beat_times

def load_annotations_downbeat(audio_dir, annotation_dir, sr=44100, hop_size_in_seconds=0.01):
    """
    Load and process beat and downbeat annotations from .beats files.
    """
    hop_size = int(np.floor(hop_size_in_seconds * 44100))
    # Get list of audio files
    audio_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith(".wav")]
    file_names = [Path(f).stem for f in audio_files]

    # Initialize dictionaries
    downbeat_times = {}

    # Process each file
    for filename in file_names:
        annotation_path = Path(annotation_dir) / (filename + ".beats")
        
        # Load beat and downbeat times from annotation file
        quantised_beat_times, quantised_downbeat_times = get_quantised_ground_truth(annotation_path, hop_size, get_downbeats=True)
    
        downbeat_times[filename] = quantised_downbeat_times
        
    return downbeat_times
        

audio_dir = "../Dataset/BallroomData/audio/"  # Path to the Ballroom dataset audio npy files
annotation_dir = "../Dataset/BallroomAnnotations-master/annotations/" # Path to the Ballroom dataset annotation files
#print(beat_times)

def load_spectrogram_and_labels(beat_times, sr=44100, hop_size_in_seconds=0.01,spectrogram_dir="../Dataset/BallroomAnnotations-master/audio/"):
    """
    Load spectrograms and beat annotations for all files in the dataset.
    """
    spectrogram_dir=spectrogram_dir
    hop_size = int(np.floor(hop_size_in_seconds * sr))  # Convert hop size to samples

    spectrograms = {}  # Dictionary to store spectrograms
    beat_vectors = {}  # Dictionary to store beat vectors

    for basename, beat_times_list in beat_times.items():
        spectrogram_path = os.path.join(spectrogram_dir, f"{basename}.npy")
        spectrogram = np.load(spectrogram_path)
        spectrograms[basename] = spectrogram

        # Initialize beat vector
        beat_vector = np.zeros(spectrogram.shape[-1])
        for time in beat_times_list:
            spec_frame = int(time * sr / hop_size)
            if spec_frame < beat_vector.shape[0] - 3:

                #if basename=='Media-106119':
                    #print(spec_frame)
                for n in range(-2, 3):  # 2 frames around the beat
                    if 0 <= spec_frame + n < beat_vector.shape[0]+2:# So that last frame is not assigned as beat
                        beat_vector[spec_frame + n] = 1.0 if n == 0 else 0.5
        

        beat_vectors[basename] = beat_vector
    return spectrograms, beat_vectors

#spectrograms, beat_vectors=load_spectrogram_and_labels(beat_times, sr=44100, hop_size_in_seconds=0.01,spectrogram_dir="../Dataset/BallroomAnnotations-master/audio/")