import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

class SinogramDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (str): Path to the folder (e.g., "train" or "test") containing the sinogram npy files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.transform = transform

        # Define the range of indices for i and j.
        # Adjust these ranges as needed for your dataset.
        # For example, training data: i in 1...170, test data: i in 1...36.
        # Here we assume the folder itself distinguishes the mode.
        # You can customize this if your folder structure is different.
        if "train" in data_dir:
            self.i_range = range(1, 171)  # For training
        else:
            self.i_range = range(1, 37)   # For testing

        # The j index range is assumed constant (e.g., 1 to 1764)
        self.j_range = range(1, 1765)

        # Create all possible (i, j) pairs
        self.pairs = [(i, j) for i in self.i_range for j in self.j_range]

        # Preload all sinogram pairs into memory
        print(f"Preloading data from {self.data_dir} ...")
        self.incomplete_data = {}
        self.complete_data = {}
        for i, j in tqdm(self.pairs):
            inc_path = os.path.join(self.data_dir, f"incomplete_{i}_{j}.npy")
            comp_path = os.path.join(self.data_dir, f"complete_{i}_{j}.npy")
            # Load data as float16 to save memory, then cast to float32 when retrieving.
            self.incomplete_data[(i, j)] = np.load(inc_path).astype(np.float16)
            self.complete_data[(i, j)] = np.load(comp_path).astype(np.float16)
        print(f"Successfully preloaded {len(self.pairs)} sinogram pairs.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        # Retrieve the preloaded data and convert to float32
        incomplete = self.incomplete_data[(i, j)].astype(np.float32)
        complete = self.complete_data[(i, j)].astype(np.float32)
        # Convert numpy arrays to torch tensors
        incomplete = torch.from_numpy(incomplete)
        complete = torch.from_numpy(complete)
        # Ensure the tensors have a channel dimension (if 2D, add one)
        if incomplete.dim() == 2:
            incomplete = incomplete.unsqueeze(0)
        if complete.dim() == 2:
            complete = complete.unsqueeze(0)
            
        # Convert single-channel to 3-channel by repeating the channel
        if incomplete.size(0) == 1:
            incomplete = incomplete.repeat(3, 1, 1)
        if complete.size(0) == 1:
            complete = complete.repeat(3, 1, 1)
            
        # Resize to 256x256
        resize_transform = transforms.Resize((256, 256), antialias=True)
        incomplete = resize_transform(incomplete)
        complete = resize_transform(complete)

        # Apply transforms if any
        if self.transform:
            incomplete = self.transform(incomplete)
            complete = self.transform(complete)
        return incomplete, complete

# Helper functions to get training and test datasets

def get_training_data(root_dir, transform=None):
    train_dir = os.path.join(root_dir, 'train')
    assert os.path.exists(train_dir), f"Training directory {train_dir} does not exist."
    return SinogramDataset(train_dir, transform=transform)

def get_test_data(root_dir, transform=None):
    test_dir = os.path.join(root_dir, 'test')
    assert os.path.exists(test_dir), f"Test directory {test_dir} does not exist."
    return SinogramDataset(test_dir, transform=transform)

# Optionally, a helper to create dataloaders for train and test

def create_dataloaders(root_dir, batch_size=8, num_workers=4, transform=None):
    train_dataset = get_training_data(root_dir, transform=transform)
    test_dataset = get_test_data(root_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader
