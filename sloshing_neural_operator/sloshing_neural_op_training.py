import torch
import scipy.io as sio
import glob
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.nn.utils import clip_grad_norm_
from neuralop.models import FNO
from neuralop import Trainer
from neuralop.training import AdamW
from neuralop import LpLoss, H1Loss
from neuralop.utils import count_model_params
from neuralop.data.transforms.normalizers import UnitGaussianNormalizer
from abc import abstractmethod
import time

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on device: {device}")

# --- 1. Find and Load All Data Files ---
data_path = './FNO_Dataset_PT/'
file_paths = glob.glob(f"{data_path}/FNO_dataset_run_*.pt")
file_paths.sort()

if not file_paths:
    raise FileNotFoundError(f"No .mat files found in {data_path}")

print(f"Found {len(file_paths)} data files.")

# --- 2. Split File Paths into Train and Test ---
train_fraction = 0.6
validation_fraction = 0.2

train_split = int(train_fraction * len(file_paths))
valid_split = int((train_fraction + validation_fraction) * len(file_paths))
train_paths = file_paths[:train_split]
valid_paths = file_paths[train_split:valid_split]
test_paths = file_paths[valid_split:]

# --- 3. Load ALL Data into RAM for Normalization ---
# This is the workflow you prefer. It requires
# loading all training data into memory first.

def load_data_from_paths(paths): # <-- Removed data_key
    all_tensors = []
    for p in paths:
        try:
            # Load the .pt file directly as a tensor
            tensor_data = torch.load(p).float() # <-- Changed loading function
            all_tensors.append(tensor_data)
        except Exception as e:
            print(f"Warning: Error loading {p}: {e}")
    # Concatenate all runs along the time dimension (dim=0)
    return torch.cat(all_tensors, dim=0)

print("Loading training data into memory...")
train_data_sequence = load_data_from_paths(train_paths)
print(f"Full training sequence shape: {train_data_sequence.shape}")

print("Loading validation data into memory...")
validation_data_sequence = load_data_from_paths(valid_paths)
print(f"Full validation sequence shape: {validation_data_sequence.shape}")

print("Loading test data into memory...")
test_data_sequence = load_data_from_paths(test_paths)
print(f"Full test sequence shape: {test_data_sequence.shape}")

# --- 4. Fit and Transform (Your Method) ---
# Create the normalizer
normalizer = UnitGaussianNormalizer(dim=[0, 2, 3, 4]) 

# Fit ONLY on the training data
print("Fitting normalizer on training data...")
normalizer.fit(train_data_sequence)
print("Fit complete.")

# Transform both sets
print("Normalizing data...")
train_data = normalizer.transform(train_data_sequence)
valid_data = normalizer.transform(validation_data_sequence)
test_data = normalizer.transform(test_data_sequence)

# --- ADD THIS SANITY CHECK ---
print(f"Normalized train data mean: {train_data.mean()}")
print(f"Normalized train data std: {train_data.std()}")
# -----------------------------

# Free up memory
del train_data_sequence
del validation_data_sequence
del test_data_sequence
print("Normalization complete. Raw data cleared from RAM.")

# --- 5. Define Simple Dataset Class ---
class TimeSteppingDataset(Dataset):
    """
    A simple dataset that just returns the (t, t+1) pairs
    from a pre-normalized data sequence.
    
    This version is "run-aware" to prevent mixing
    data from different simulation runs.
    """
    def __init__(self, data_sequence, steps_per_run):
        """
        Args:
            data_sequence (torch.Tensor): The giant tensor of all runs
            steps_per_run (int): The number of time steps in EACH run
                                 (e.g., 100)
        """
        self.data = data_sequence
        self.steps_per_run = steps_per_run
        
        # We can't use the last step of *any* run as an input 'x'
        self.valid_pairs_per_run = self.steps_per_run - 1
        
        # Calculate how many runs are in this tensor
        self.num_runs = self.data.shape[0] // self.steps_per_run
        
        if self.data.shape[0] % self.steps_per_run != 0:
            print(f"Warning: Data shape {self.data.shape[0]} is not "
                  f"perfectly divisible by steps_per_run {self.steps_per_run}")

    def __len__(self):
        """
        Returns the total number of *valid* (t, t+1) pairs.
        """
        return self.num_runs * self.valid_pairs_per_run

    def __getitem__(self, idx):
        """
        Gets the N-th *valid* pair, skipping boundaries.
        'idx' will be from 0 to (total_valid_pairs - 1)
        """
        
        # 1. Which run is this pair in?
        # e.g., if valid_pairs_per_run=99, idx=100 -> run_index=1
        run_index = idx // self.valid_pairs_per_run
        
        # 2. What is the time-index *within* that run?
        # e.g., if valid_pairs_per_run=99, idx=100 -> time_index=1
        time_index_in_run = idx % self.valid_pairs_per_run
        
        # 3. What is the *actual* index in the giant data tensor?
        # This calculation skips the boundary indices.
        # e.g., run_index=1, time_index=1 -> (1 * 100) + 1 = 101
        global_start_idx = (run_index * self.steps_per_run) + time_index_in_run
        
        # This will now correctly get (e.g.) data[101] and data[102]
        # and will *never* ask for (data[99], data[100])
        x = self.data[global_start_idx]
        y = self.data[global_start_idx + 1]
        
        return {'x': x, 'y': y}

# --- 6. Create Datasets and DataLoaders ---

# You must know this value from your data generation
# For example, if each .pt file had 100 time steps:
STEPS_PER_RUN = 101 

# Create the datasets from your NEW normalized tensors
train_dataset = TimeSteppingDataset(train_data, steps_per_run=STEPS_PER_RUN)
valid_dataset = TimeSteppingDataset(valid_data, steps_per_run=STEPS_PER_RUN)
test_dataset = TimeSteppingDataset(test_data, steps_per_run=STEPS_PER_RUN)

# Create the DataLoaders
# Try a small batch size first due to memory
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=100, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

# --- 7. Define Model, Optimizer, Loss ---
model = FNO(
    n_modes=(8, 8, 8),
    hidden_channels=4,
    in_channels=3,
    out_channels=3,
    n_layers=2
).to(device) 

print(f"Model has {count_model_params(model)} parameters.")

n_epochs = 1000
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3) # Using the lower 1e-4 lr
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
l2loss = LpLoss(d=3, p=2)
h1loss = H1Loss(d=3)

# --- 8. Create Trainer (No Processor) ---
trainer = Trainer(model=model, n_epochs=n_epochs,
                  device=device,
                  wandb_log=False,
                  eval_interval=1,
                  use_distributed=False,
                  verbose=True)

# --- 9. Start Training ---
print("Starting training on full, normalized dataset...")
# Use the shape of one test sample as the key
valid_key = valid_data[0].shape[1]
trainer.train(train_loader=train_loader,
              test_loaders={valid_key: valid_loader},
              optimizer=optimizer,
              scheduler=scheduler,
              training_loss=l2loss,
              eval_losses={'h1': h1loss, 'l2': l2loss},
                save_best=f'{valid_key}_l2',
                save_dir='./checkpoints/')