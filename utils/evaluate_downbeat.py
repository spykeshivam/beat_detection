import torch
import numpy as np
import os
from scipy.signal import find_peaks
import librosa
import sys
import mir_eval
import matplotlib.pyplot as plt
from pathlib import Path

"""
Parts of the functions load_ground_truth in this code was taken from :
https://github.com/ben-hayes/beat-tracking-tcn/blob/master/beat_tracking_tcn/datasets/ballroom_dataset.py
"""

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.tcn import BeatNet

def evaluate_downbeat_detection_model(model, input_np_path, hop_length=441, sr=44100, prominence=0.1, width=1, device='cpu'):
    """
    Evaluate a downbeat detection model by running inference and detecting peaks.
    """
    input_data = np.load(input_np_path)
    
    # Ensure input is properly shaped for the model
    if len(input_data.shape) == 2:
        input_data = np.expand_dims(input_data, axis=0)
    
    input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)
    
    # Predict probability
    with torch.no_grad():
        probabilities = model(input_tensor)
        
    if isinstance(probabilities, torch.Tensor):
        probabilities = probabilities.cpu().numpy()
    
    probabilities = probabilities.flatten()
    
    # Find peaks in the probability array
    downbeat_frames, _ = find_peaks(probabilities, prominence=prominence, width=width)
    
    # Convert frames to timestamps
    downbeat_timestamps = librosa.frames_to_time(downbeat_frames, sr=sr, hop_length=hop_length)
    
    return downbeat_timestamps, downbeat_frames, probabilities

def text_label_to_float(text):
    """Extracts beat time from a text line and converts to a float"""
    allowed = '1234567890. \t'
    filtered = ''.join([c for c in text if c in allowed])
    if '\t' in filtered:
        t = filtered.rstrip('\n').split('\t')
    else:
        t = filtered.rstrip('\n').split(' ')
    return float(t[0]), float(t[1]) if len(t) > 1 else (float(t[0]), None)

def load_ground_truth_downbeats(annotation_path):
    """
    Load ground truth downbeat annotations from a .beats file.
    """
    downbeat_times = []
    
    with open(annotation_path, "r") as f:
        for line in f:
            time, beat_position = text_label_to_float(line)
            # Check if this is a downbeat (position 1)
            if beat_position is not None and beat_position == 1:
                downbeat_times.append(time)
    
    return np.array(downbeat_times)

def load_annotations_downbeat(audio_dir, annotation_dir, sr=44100, hop_size_in_seconds=0.01):
    """
    Load and process downbeat annotations from .beats files.
    """
    hop_size = int(np.floor(hop_size_in_seconds * sr))
    # Get list of audio files
    audio_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith(".wav")]
    file_names = [Path(f).stem for f in audio_files]

    # Initialize dictionary
    downbeat_times = {}

    # Process each file
    for filename in file_names:
        annotation_path = Path(annotation_dir) / (filename + ".beats")
        if os.path.exists(annotation_path):
            downbeat_times[filename] = load_ground_truth_downbeats(annotation_path)
        
    return downbeat_times

def run_batch_evaluation(model_path, audio_dir, annotations_dir, output_dir="downbeat_evaluation_results", max_duration=29):
    """
    Run downbeat evaluation on all files in the dataset.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = BeatNet(channels=16, tcn_layers=11, kernel_size=5, dropout=0.1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded from {model_path}")
    
    # Store results
    results = {}
    all_f_scores = []
    
    # Process all files
    total_files = len([f for f in os.listdir(audio_dir) if f.endswith('.npy')])
    print(f"Found {total_files} .npy files to process")
    
    for i, filename in enumerate(os.listdir(audio_dir)):
        if filename.endswith('.npy'):
            base_name = os.path.splitext(filename)[0]
            beat_file = os.path.join(annotations_dir, f"{base_name}.beats")
            audio_file = os.path.join(audio_dir, filename)
            
            # Check if both files exist
            if not os.path.exists(beat_file):
                print(f"Annotation file not found for {base_name}, skipping...")
                continue
            
            print(f"Processing file {i+1}/{total_files}: {base_name}")
            
            try:
                downbeat_timestamps, downbeat_frames, probabilities = evaluate_downbeat_detection_model(
                    model, audio_file, hop_length=441, sr=44100, device=device
                )
                print(downbeat_timestamps)
                
                
                # Load ground truth downbeats for comparison
                ground_truth_downbeats = load_ground_truth_downbeats(beat_file)
                
                # Apply maximum duration constraint
                ground_truth_downbeats = ground_truth_downbeats[ground_truth_downbeats <= max_duration]
                downbeat_timestamps = downbeat_timestamps[downbeat_timestamps <= max_duration]
                
                # Calculate metrics using mir_eval
                if len(ground_truth_downbeats) > 0 and len(downbeat_timestamps) > 0:
                    scores = mir_eval.beat.evaluate(ground_truth_downbeats, downbeat_timestamps)
                    
                    # Store results
                    results[base_name] = scores
                    all_f_scores.append(scores['F-measure'])
                    
                    print(f"  Downbeat F-measure: {scores['F-measure']:.4f}")
                
                
            except Exception as e:
                print(f"  Error processing {base_name}: {str(e)}")
    
    # Calculate average F-score
    average_f_score = np.mean(all_f_scores) if all_f_scores else 0
    print(f"Average Downbeat F-measure across all files: {average_f_score:.4f}")
    
    # Create F-score distribution histogram
    plt.figure(figsize=(10, 6))
    plt.hist(all_f_scores, bins=20, alpha=0.7)
    plt.axvline(average_f_score, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {average_f_score:.4f}')
    plt.xlabel('F-measure')
    plt.ylabel('Number of Files')
    plt.title('Distribution of Downbeat F-measures Across Dataset')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'downbeat_f_measure_distribution.png'))
    


if __name__ == "__main__":
    model_path = "../model_save/down_beat_tracker_0.0457.pt"
    audio_dir = "../Dataset/BallroomAnnotations-master/audio"
    annotations_dir = "../Dataset/BallroomAnnotations-master/annotations"
    
    run_batch_evaluation(
        model_path, audio_dir, annotations_dir
    )