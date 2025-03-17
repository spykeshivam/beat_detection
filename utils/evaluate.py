import torch
import numpy as np
import os
from scipy.signal import find_peaks
import librosa
import sys
import mir_eval
import matplotlib.pyplot as plt

"""
Parts of the functions load_ground_truth in this code was taken from :
https://github.com/ben-hayes/beat-tracking-tcn/blob/master/beat_tracking_tcn/datasets/ballroom_dataset.py
"""

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.tcn import BeatNet

def evaluate_beat_detection_model(model, input_np_path, hop_length=441, sr=44100, prominence=0.1, width=1, device='cpu'):
    """
    Evaluate a beat detection model by running inference and detecting peaks.
    """
    input_data = np.load(input_np_path)
    
    # Ensure input is properly shaped for the model
    if len(input_data.shape) == 2:
        input_data = np.expand_dims(input_data, axis=0)
    
    input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)
    
    # Predict probablity
    with torch.no_grad():
        probabilities = model(input_tensor)
        
    if isinstance(probabilities, torch.Tensor):
        probabilities = probabilities.cpu().numpy()
    
    probabilities = probabilities.flatten()
    
    # Find peaks in the probability array
    beat_frames, _ = find_peaks(probabilities, prominence=prominence, width=width)
    
    # Convert frames to timestamps
    beat_timestamps = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
    
    return beat_timestamps, beat_frames, probabilities

def load_ground_truth(annotation_path):
    """
    Load ground truth beat annotations from a .beats file.
    """
    def text_label_to_float(text):
        """Extracts beat time from a text line and converts to a float"""
        allowed = '1234567890. \t'
        filtered = ''.join([c for c in text if c in allowed])
        if '\t' in filtered:
            t = filtered.rstrip('\n').split('\t')
        else:
            t = filtered.rstrip('\n').split(' ')
        return float(t[0])
    
    with open(annotation_path, "r") as f:
        beat_times = []
        for line in f:
            time = text_label_to_float(line)
            beat_times.append(time)
    
    return np.array(beat_times)

def run_batch_evaluation(model_path, audio_dir, annotations_dir, output_dir="evaluation_results", max_duration=29):
    """
    Run evaluation on all files in the dataset.
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
                # Run inference and get beat timestamps
                # beat_frames, probabilities for debugging
                beat_timestamps, beat_frames, probabilities = evaluate_beat_detection_model(
                    model, audio_file, hop_length=441, sr=44100, device=device
                )
                
                # Load ground truth for comparison
                ground_truth_beats = load_ground_truth(beat_file)
                
                # Apply maximum duration constraint to check for beats within 29 secs
                ground_truth_beats = ground_truth_beats[ground_truth_beats <= max_duration]
                beat_timestamps = beat_timestamps[beat_timestamps <= max_duration]
                
                # Calculate metrics using mir_eval
                if len(ground_truth_beats) > 0 and len(beat_timestamps) > 0:
                    scores = mir_eval.beat.evaluate(ground_truth_beats, beat_timestamps)
                    
                    # Store results
                    results[base_name] = scores
                    all_f_scores.append(scores['F-measure'])
                    
                    print(f"  F-measure: {scores['F-measure']:.4f}")
                else:
                    print(f"Empty ground truth or predictions for {base_name}")
                
            except Exception as e:
                print(f"  Error processing {base_name}: {str(e)}")
    
    # Calculate average F-score
    average_f_score = np.mean(all_f_scores) if all_f_scores else 0
    print(f"Average F-measure across all files: {average_f_score:.4f}")
    
    # Create F-score distribution histogram
    plt.figure(figsize=(10, 6))
    plt.hist(all_f_scores, bins=20, alpha=0.7)
    plt.axvline(average_f_score, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {average_f_score:.4f}')
    plt.xlabel('F-measure')
    plt.ylabel('Number of Files')
    plt.title('Distribution of F-measures Across Dataset')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'f_measure_distribution.png'))
    
    return average_f_score, results

if __name__ == "__main__":
    # Configuration
    model_path = "../model_save/beat_tracker_0.1065.pt"
    audio_dir = "../Dataset/BallroomAnnotations-master/audio"
    annotations_dir = "../Dataset/BallroomAnnotations-master/annotations"
    
    # Run batch evaluation
    average_f_score, all_results = run_batch_evaluation(
        model_path, audio_dir, annotations_dir
    )
    
    print(f"Average F-measure: {average_f_score:.4f}")
    