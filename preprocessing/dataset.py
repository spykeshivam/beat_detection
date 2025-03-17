import torch
from torch.utils.data import Dataset

class BallroomDataset(Dataset):
    def __init__(self, spectrograms, beat_vectors):
        """
        Initialize the dataset.
        """
        self.spectrograms = spectrograms
        self.beat_vectors = beat_vectors
        self.basenames = list(spectrograms.keys())

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.basenames)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        """
        basename = self.basenames[idx]
        spectrogram = self.spectrograms[basename]
        beat_vector = self.beat_vectors[basename]

        #Convert to PyTorch tensors
        spectrogram = torch.from_numpy(spectrogram).float()
        beat_vector = torch.from_numpy(beat_vector).float()

        #Add channel dimension to spectrogram. Helps during creating the network.
        #spectrogram = spectrogram.squeeze(0).T
        return spectrogram, beat_vector