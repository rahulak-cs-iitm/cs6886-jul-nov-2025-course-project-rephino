# Fault Injecting an FNO trained on Darcy-Flow
import torch
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
from neuralop.models import FNO
from neuralop import Trainer
from neuralop.training import AdamW
from neuralop.data.datasets import load_darcy_flow_small
from neuralop.utils import count_model_params
from neuralop import LpLoss, H1Loss
from neuralop.data.datasets.darcy import DarcyDataset
from torch.utils.data import DataLoader
import time

# For Fault injection
from pytorchfi.core import fault_injection

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Running on device: {device}')

#==================
# Get-Set Data
#------------------

# Create the main dataset object
dataset = DarcyDataset(root_dir="../Data/Darcy",
                       n_train=1000,
                       n_tests=[100, 50],
                       batch_size=32,
                       test_batch_sizes=[32, 32],
                       train_resolution=32,
                       test_resolutions=[32, 64],
                       download=True 
)

# Get the data processor from the dataset object
data_processor = dataset.data_processor

# Create the training DataLoader
train_loader = DataLoader(dataset.train_db,
                          batch_size=32,
                          shuffle=True,
                          num_workers=0,
                          pin_memory=True)

# Create a dictionary for the test DataLoaders
test_loaders = {}
test_resolutions = [32, 64]
test_batch_sizes = [32, 32]

for res, bsize in zip(test_resolutions, test_batch_sizes):
    test_loaders[res] = DataLoader(dataset.test_dbs[res],
                                   batch_size=bsize,
                                   shuffle=False,
                                   num_workers=0,
                                   pin_memory=True)

#===================
# Simple FNO Model
#-------------------
model = FNO(n_modes=(32, 32),
             in_channels=1, 
             out_channels=1,
             hidden_channels=32, 
             projection_channel_ratio=2)
model = model.to(device)

# Load the model with already trained parameters
model.load_state_dict(torch.load('../Checkpoints/Darcy/darcy_fno_state_dict.pt', map_location=torch.device(device), weights_only=False) )

n_params = count_model_params(model)
print(f'\nThe model has {n_params} parameters.')
sys.stdout.flush()
print('\n### MODEL ###\n', model)

#===================
# Training Setup 
#-------------------

# Optimizer and Scheduler
optimizer = AdamW(model.parameters(), 
                                lr=8e-3, 
                                weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

# Losses
l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)

train_loss = h1loss
eval_losses={'h1': h1loss, 'l2': l2loss}

# Training
print('\nSKIPPING TRAINING - Using pretrained weights ! \n')

# print('\n### MODEL ###\n', model)
# print('\n### OPTIMIZER ###\n', optimizer)
# print('\n### SCHEDULER ###\n', scheduler)
# print('\n### LOSSES ###')
# print(f'\n * Train: {train_loss}')
# print(f'\n * Test: {eval_losses}')
# # sys.stdout.flush()
# 
# trainer = Trainer(model=model, n_epochs=60,
#                   device=device,
#                   data_processor=data_processor,
#                   wandb_log=False,
#                   eval_interval=3,
#                   use_distributed=False,
#                   verbose=True)
# 
# trainer.train(train_loader=train_loader,
#               test_loaders=test_loaders,
#               optimizer=optimizer,
#               scheduler=scheduler, 
#               regularizer=False, 
#               training_loss=train_loss,
#               eval_losses=eval_losses)

#==========================
# Visualizing Predictions 
#--------------------------

test_samples = test_loaders[32].dataset

# --- Create the output directory if it doesn't exist ---
output_dir = 'plots'
os.makedirs(output_dir, exist_ok=True)
model.to(device)

# #----------------------------------------------------
# # === INFERENCE TIME MEASUREMENT FOR CLEAN MODEL ===
# print("\n" + "="*50)
# print("MEASURING CLEAN MODEL INFERENCE TIME")
# print("="*50)
# 
# # Warmup runs
# print("Warming up...")
# with torch.no_grad():
#     for _ in range(10):
#         data = test_samples[0]
#         data = data_processor.preprocess(data, batched=False)
#         x = data['x'].to(device)
#         _ = model(x.unsqueeze(0))
# 
# # Measure inference time over multiple runs
# num_runs = 100
# inference_times = []
# 
# print(f"Running {num_runs} inference iterations...")
# with torch.no_grad():
#     for i in range(num_runs):
#         data = test_samples[i % len(test_samples)]
#         data = data_processor.preprocess(data, batched=False)
#         x = data['x'].to(device)
#         
#         start_time = time.perf_counter()
#         out = model(x.unsqueeze(0))
#         end_time = time.perf_counter()
#         
#         inference_times.append((end_time - start_time) * 1000)  # Convert to milliseconds
# 
# # Calculate statistics
# avg_time = np.mean(inference_times)
# std_time = np.std(inference_times)
# min_time = np.min(inference_times)
# max_time = np.max(inference_times)
# 
# print(f"\nClean Model Inference Time Statistics:")
# print(f"  Average: {avg_time:.2f} ms")
# print(f"  Std Dev: {std_time:.2f} ms")
# print(f"  Min: {min_time:.2f} ms")
# print(f"  Max: {max_time:.2f} ms")
# print(f"  FPS: {1000/avg_time:.2f}")
# 
# # Save results to file
# results_file = os.path.join(output_dir, 'inference_times.txt')
# with open(results_file, 'w') as f:
#     f.write("="*50 + "\n")
#     f.write("CLEAN MODEL INFERENCE TIME\n")
#     f.write("="*50 + "\n")
#     f.write(f"Average: {avg_time:.2f} ms\n")
#     f.write(f"Std Dev: {std_time:.2f} ms\n")
#     f.write(f"Min: {min_time:.2f} ms\n")
#     f.write(f"Max: {max_time:.2f} ms\n")
#     f.write(f"FPS: {1000/avg_time:.2f}\n\n")
# #----------------------------------------------------

fig = plt.figure(figsize=(7, 7))
for index in range(3):
    data = test_samples[index]
    data = data_processor.preprocess(data, batched=False)
    # Input x
    x = data['x'].to(device)
    # Ground-truth
    y = data['y'].to(device)
    # Model prediction
    out = model(x.unsqueeze(0))

    ax = fig.add_subplot(3, 3, index*3 + 1)
    ax.imshow(x[0].cpu(), cmap='gray')
    if index == 0: 
        ax.set_title('Input x')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 3, index*3 + 2)
    ax.imshow(y.cpu().squeeze())
    if index == 0: 
        ax.set_title('Ground-truth y')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 3, index*3 + 3)
    ax.imshow(out.squeeze().cpu().detach().numpy())
    if index == 0: 
        ax.set_title('Model prediction')
    plt.xticks([], [])
    plt.yticks([], [])

fig.suptitle('Inputs, ground-truth output and prediction (32x32).', y=0.98)
plt.tight_layout()

# --- Save the figure to a file ---
# Create a path to save the image in the 'plots' folder
file_path = os.path.join(output_dir, 'fno_predictions.png')
plt.savefig(file_path)
print(f"Figure saved to: {file_path}")

# --- Display the figure and wait until it is closed ---
# plt.show()

#====================================
# Fault Injection - Single Bit Flip
#-----------------------------------
import struct
import random

# Set Seeds for Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
print(f"Seeds set to {SEED} for reproducibility.")

# Bit-Flipping Helper Functions
def float_to_int(f):
    """Packs a float into a 32-bit integer."""
    return struct.unpack('<I', struct.pack('<f', f))[0]

def int_to_float(i):
    """Unpacks a 32-bit integer into a float."""
    return struct.unpack('<f', struct.pack('<I', i))[0]

def single_bit_flip_random(original_float):
    """Performs a single, random bit-flip on a 32-bit float."""
    if not isinstance(original_float, float):
        original_float = float(original_float)
    
    original_int = float_to_int(original_float)
    
    # --- Choose a random bit to flip ---
    bit_to_flip = random.randint(0, 31)
    
    flipper_mask = 1 << bit_to_flip
    flipped_int = original_int ^ flipper_mask
    
    flipped_float = int_to_float(flipped_int)
    
    return flipped_float

# Custom Fault Injector Class
class RadiationUpsetInjector(fault_injection):
    def __init__(self, model, batch_size, **kwargs):
        super().__init__(model, batch_size, **kwargs)

    def custom_single_bit_flip(self, module, input, output):
        """Custom hook function to inject the fault."""
        shape = output.shape
        rand_neuron_idx = [random.randint(0, dim - 1) for dim in shape[1:]]
        full_idx = (0, *rand_neuron_idx)
        original_val = output[full_idx].item()
        
        # Use the updated random bit-flip function
        flipped_val = single_bit_flip_random(original_val)
        
        with torch.no_grad():
            output[full_idx] = flipped_val
        
        self.updateLayer()
        if self.get_current_layer() >= self.get_total_layers():
            self.reset_current_layer()

# Main Execution
batch_size = 1
c = 1 
h = 1
w = 32
use_cuda = True if torch.cuda.is_available() else False

pfi_model = RadiationUpsetInjector(model, 
                     batch_size,
                     input_shape=[c,h,w],
                     layer_types=[torch.nn.Conv1d],
                     use_cuda=use_cuda,
                     )

# Declare the fault injection with your custom function
inj = pfi_model.declare_neuron_fi(function=pfi_model.custom_single_bit_flip)

# #----------------------------------------------------
# # === INFERENCE TIME MEASUREMENT FOR SINGLE BIT FLIP ===
# print("\n" + "="*50)
# print("MEASURING SINGLE BIT FLIP INFERENCE TIME")
# print("="*50)
# 
# # Warmup
# print("Warming up fault injection...")
# for _ in range(10):
#     _ = inj(x.unsqueeze(0))
# 
# # Measure
# num_runs = 100
# fi_single_times = []
# 
# print(f"Running {num_runs} fault injection iterations...")
# for i in range(num_runs):
#     start_time = time.perf_counter()
#     inj_output = inj(x.unsqueeze(0))
#     end_time = time.perf_counter()
#     fi_single_times.append((end_time - start_time) * 1000)
# 
# # Calculate statistics
# avg_fi_single = np.mean(fi_single_times)
# std_fi_single = np.std(fi_single_times)
# 
# print(f"\nSingle Bit Flip Inference Time:")
# print(f"  Average: {avg_fi_single:.2f} ms")
# print(f"  Std Dev: {std_fi_single:.2f} ms")
# print(f"  Overhead vs Clean: {avg_fi_single - avg_time:.2f} ms ({((avg_fi_single/avg_time - 1) * 100):.1f}%)")
# 
# # Append to results file
# with open(results_file, 'a') as f:
#     f.write("="*50 + "\n")
#     f.write("SINGLE BIT FLIP INFERENCE TIME\n")
#     f.write("="*50 + "\n")
#     f.write(f"Average: {avg_fi_single:.2f} ms\n")
#     f.write(f"Std Dev: {std_fi_single:.2f} ms\n")
#     f.write(f"Overhead vs Clean: {avg_fi_single - avg_time:.2f} ms ({((avg_fi_single/avg_time - 1) * 100):.1f}%)\n\n")
# #----------------------------------------------------

# Inject error and get faulty output
inj_output = inj(x.unsqueeze(0))

# --- Create the output directory if it doesn't exist ---
output_dir = 'plots'
os.makedirs(output_dir, exist_ok=True)

# --- Visualize Original vs. Injected Output ---
fig1, ax1 = plt.subplots(1, 2, figsize=(8, 4))
ax1[0].imshow(out.squeeze().cpu().detach().numpy())
ax1[0].set_title("Original (Fault-Free) Output")
ax1[0].set_xticks([])
ax1[0].set_yticks([])

ax1[1].imshow(inj_output.squeeze().cpu().detach().numpy())
ax1[1].set_title("Injected Output (1-bit flip)")
ax1[1].set_xticks([])
ax1[1].set_yticks([])
fig1.tight_layout()

# --- Visualize Error Heatmap ---
# Convert tensors to numpy arrays
out_np = out.squeeze().cpu().detach().numpy()
inj_output_np = inj_output.squeeze().cpu().detach().numpy()

# Calculate the absolute difference
error_map = np.abs(out_np - inj_output_np)

# Create a new figure for the heatmap
fig2, ax2 = plt.subplots(figsize=(6, 5))

heatmap = ax2.imshow(error_map, cmap='YlOrRd')
plt.colorbar(heatmap, ax=ax2, label='Absolute Error')
ax2.set_title('Error Heatmap')
ax2.set_xticks([])
ax2.set_yticks([])

# --- Save the heatmap figure ---
file_path = os.path.join(output_dir, 'single_bit_flip.png')
plt.savefig(file_path)
print(f"Error heatmap saved to: {file_path}")


# plt.show()

#====================================
# Fault Injection - Multi Bit Flip
#-----------------------------------

# Set Seeds for Reproducibility
SEED = 99090 
# Perfect SEED = 99090 
# Okay SEED = 12334
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
print(f"Seeds set to {SEED} for reproducibility.")

# Bit-Flipping Helper Functions
def float_to_int(f):
    return struct.unpack('<I', struct.pack('<f', f))[0]

def int_to_float(i):
    return struct.unpack('<f', struct.pack('<I', i))[0]

def multiple_bit_flip(original_float, n_bitflips=1):
    original_int = float_to_int(float(original_float))
    # Flip N unique random bits
    bit_positions = random.sample(range(32), min(n_bitflips, 32))
    for bit in bit_positions:
        original_int ^= (1 << bit)
    return int_to_float(original_int)

# Custom Multi-Bit Fault Injector Class
class RadiationUpsetInjectorMulti(fault_injection):
    def __init__(self, model, batch_size, n_bitflips=1, n_neurons=1, **kwargs):
        super().__init__(model, batch_size, **kwargs)
        self.n_bitflips = n_bitflips
        self.n_neurons = n_neurons

    def n_bit_flip_func_multi(self, module, input, output):
        shape = output.shape
        batch_idx = 0 
        neurons_idx = set()
        # Randomly select M distinct neurons
        while len(neurons_idx) < self.n_neurons:
            neuron = tuple(random.randint(0, dim - 1) for dim in shape[1:])
            neurons_idx.add(neuron)
        with torch.no_grad():
            for neuron in neurons_idx:
                full_idx = (batch_idx, *neuron)
                val = output[full_idx].item()
                output[full_idx] = multiple_bit_flip(val, self.n_bitflips)

# Main Execution
batch_size = 1
c = 1 
h = 1
w = 32

pfi_model_multibit = RadiationUpsetInjectorMulti(
    model,
    batch_size,
    n_bitflips=2,       
    n_neurons=2,        
    input_shape=[c, h, w],
    layer_types=[torch.nn.Conv1d, torch.nn.Conv2d],
    use_cuda=use_cuda,
)

inj = pfi_model_multibit.declare_neuron_fi(function=pfi_model_multibit.n_bit_flip_func_multi)
inj_output = inj(x.unsqueeze(0))

# --- Create the output directory if it doesn't exist ---
output_dir = 'plots'
os.makedirs(output_dir, exist_ok=True)

# --- Visualize Original vs. Injected Output ---
fig1, ax1 = plt.subplots(1, 2, figsize=(8, 4))
ax1[0].imshow(out.squeeze().cpu().detach().numpy())
ax1[0].set_title("Original (Fault-Free) Output")
ax1[0].set_xticks([])
ax1[0].set_yticks([])

ax1[1].imshow(inj_output.squeeze().cpu().detach().numpy())
ax1[1].set_title(f"Injected Output ({pfi_model_multibit.n_neurons} neurons, {pfi_model_multibit.n_bitflips} flips each)")
ax1[1].set_xticks([])
ax1[1].set_yticks([])
fig1.tight_layout()

# --- Visualize Error Heatmap ---
out_np = out.squeeze().cpu().detach().numpy()
inj_output_np = inj_output.squeeze().cpu().detach().numpy()
error_map = np.abs(out_np - inj_output_np)

fig2, ax2 = plt.subplots(figsize=(6, 5))
heatmap = ax2.imshow(error_map, cmap='YlOrRd')
plt.colorbar(heatmap, ax=ax2, label='Absolute Error')
ax2.set_title('Error Heatmap')
ax2.set_xticks([])
ax2.set_yticks([])

# --- Save the heatmap figure ---
file_path = os.path.join(output_dir, 'multi_bit_flip.png')
plt.savefig(file_path)
print(f"Error heatmap saved to: {file_path}")

# --- Display figures and wait until closed ---
plt.show()


