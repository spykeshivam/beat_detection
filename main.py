import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from preprocessing.load_annotations import load_annotations_beat, load_spectrogram_and_labels, load_annotations_downbeat
from preprocessing.dataset import BallroomDataset
from models.tcn import BeatNet
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from preprocessing.audio_to_features import extract_features
from utils.evaluate import evaluate_beat_detection_model

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=30, early_stopping_patience=20):
    """
    Train the beat tracking model with early stopping and learning rate scheduling
    """
    # Initialize tracking variables
    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_losses = []
    val_losses = []
    
    # Create models directory if it doesn't exist
    Path("model_save").mkdir(exist_ok=True)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        #Progress bar for training
        train_pbar = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        train_pbar.set_description(f"Epoch {epoch+1}")
        
        for i, (spectrogram, beat_vector) in train_pbar:
            spectrogram = spectrogram.to(device)
            beat_vector = beat_vector.to(device)
            optimizer.zero_grad()
            
            outputs = model(spectrogram)
            
            # Calculate loss
            loss = criterion(outputs, beat_vector)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Update running loss
            running_loss += loss.item()
            train_pbar.set_description(f"Epoch {epoch+1}   , Loss:     {running_loss/(i+1):.4f}")
        
        # Calculate average training loss for this epoch
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            val_pbar = tqdm(enumerate(val_loader), total=len(val_loader), leave=False)
            val_pbar.set_description(f"Validation")
            
            for i, (spectrogram, beat_vector) in val_pbar:
                # Move data to device
                spectrogram = spectrogram.to(device)
                beat_vector = beat_vector.to(device)
                
                # Forward pass
                outputs = model(spectrogram)
                
                # Calculate loss
                loss = criterion(outputs, beat_vector)
                
                # Update validation loss
                val_loss += loss.item()
                val_pbar.set_description(f"Validation, Loss: {val_loss/(i+1):.4f}")
        
        # average validation loss
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Training Loss: {avg_train_loss:.4f}, "
              f"Validation Loss: {avg_val_loss:.4f}, "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Update learning rate based on validation loss
        scheduler.step(avg_val_loss)
        
        # Early stopping check. If loss is not decreasing. Stop training.
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            #torch.save(model.state_dict(), Path("model_save") / f"beat_tracker_{avg_val_loss:.4f}.pt")
            torch.save(model.state_dict(), Path("model_save") / f"down_beat_tracker_{avg_val_loss:.4f}.pt")
            print(f"Model saved with val_loss: {avg_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs")
            
        if epochs_no_improve >= early_stopping_patience:
            print("Early stopping")
            break
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.close()
    
    return train_losses, val_losses


import tempfile
def beatTracker(inputFile):
    device='cpu'
    device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    beat_model = loadModel('model_save/beat_tracker_0.1065.pt', device)
    down_beat_model = loadModel('model_save/down_beat_tracker_0.0457.pt', device)
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as temp_file:
        temp_path = temp_file.name
            # Extract features
        features = extract_features(inputFile)
        np.save(temp_path, features)
        beat_timestamps=evaluate_beat_detection_model(beat_model, temp_path, hop_length=441, sr=44100, prominence=0.1, width=1)
        downbeat_timestamps=evaluate_beat_detection_model(down_beat_model, temp_path, hop_length=441, sr=44100, prominence=0.1, width=1)
    
        
    
    return beat_timestamps[0], downbeat_timestamps[0]

def loadModel(model_path, device):
    """
    Load the beat/downbeat detection model.
    """
    from models.tcn import BeatNet

    
    # Initialize model
    model = BeatNet(channels=16, tcn_layers=11, kernel_size=5, dropout=0.1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model




if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    audio_dir = "Dataset/BallroomData/audio"
    annotation_dir = "Dataset/BallroomAnnotations-master/annotations/"
    spectrogram_dir = "Dataset/BallroomAnnotations-master/audio/"
    beats,downbeats=beatTracker("Dataset/BallroomData/audio/Albums-AnaBelen_Veneo-02.wav")
    print(beats,downbeats)
    """
    UNCOMMENT THIS CODE FOR TRAINING

    # Load beat annotations
    beat_times = load_annotations_beat(audio_dir, annotation_dir, sr=44100, hop_size_in_seconds=0.01)
    beat_times, downbeat_times = load_annotations_beat_and_downbeat(audio_dir, annotation_dir, sr=44100, hop_size_in_seconds=0.01)
    #print(beat_times)
    # Load spectrograms and beat vectors
    spectrograms, beat_vectors = load_spectrogram_and_labels(downbeat_times, sr=44100, hop_size_in_seconds=0.01, spectrogram_dir=spectrogram_dir)
    
    # Create the dataset
    full_dataset = BallroomDataset(spectrograms, beat_vectors)
    
    # Split dataset into train, validation, and test sets (70%, 15%, 15%)
    dataset_size = len(full_dataset)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    batch_size = 16  
    # pin_memory for better gpu processing
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Print dataset sizes
    print(f"Total dataset size: {dataset_size}")
    print(f"Training set size: {train_size}")
    print(f"Validation set size: {val_size}")
    print(f"Test set size: {test_size}")
    
    # Initialize model
    model = BeatNet(channels=16, tcn_layers=11, kernel_size=5, dropout=0.1).to(device)
    
    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Added weight decay for regularization
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=5, min_lr=1e-6, verbose=True)
    
    # Train the model
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=300,
        early_stopping_patience=10
    )
    
    # Evaluate on test set
    model.eval()
    test_loss = 0.0
    
    with torch.no_grad():
        for spectrogram, beat_vector in tqdm(test_loader, desc="Testing"):
            spectrogram = spectrogram.to(device)
            beat_vector = beat_vector.to(device)
            
            outputs = model(spectrogram)
            loss = criterion(outputs, beat_vector)
            
            test_loss += loss.item()
    
    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4f}")
    
    print("Training and evaluation complete!")"""

#check utils/evaluate.py for evaluation of the model
