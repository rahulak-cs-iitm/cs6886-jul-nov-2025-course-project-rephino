# CollectMaxActivation.py
#   Collect maximum activation value across the FNO network
#   Save to file for use in fault injection experiments

import torch
import numpy as np
import json
from neuralop.models import FNO
from neuralop.data.datasets.darcy import DarcyDataset
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Running on device: {device}')

#==================
# Configuration
#------------------
NUM_SAMPLES = 100  # Number of samples to analyze
SEED = 42
OUTPUT_FILE = 'max_activation_value.json'

#==================
# Load Dataset
#------------------
print("Loading Darcy dataset...")
dataset = DarcyDataset(root_dir="../../../Data/Darcy",
                       n_train=1000,
                       n_tests=[100, 50],
                       batch_size=32,
                       test_batch_sizes=[32, 32],
                       train_resolution=32,
                       test_resolutions=[32, 64],
                       download=True)

data_processor = dataset.data_processor
test_loader = DataLoader(dataset.test_dbs[32],
                         batch_size=1,
                         shuffle=False,
                         num_workers=0,
                         pin_memory=True)

#==================
# Load Model
#------------------
print("Loading FNO model...")
model_path = '../../../Checkpoints/Darcy/darcy_fno_state_dict.pt'

model = FNO(n_modes=(32, 32), in_channels=1, out_channels=1,
            hidden_channels=32, projection_channel_ratio=2)
model = model.to(device)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
model.eval()

print("Model loaded successfully")

#==================
# Activation Tracking Hook
#------------------
class GlobalMaxActivationTracker:
    def __init__(self):
        self.global_max = float('-inf')
        self.layer_max = {}
        
    def hook_fn(self, name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                # Get max absolute value for this layer
                layer_max = output.abs().max().item()
                
                # Check validity
                if not np.isnan(layer_max) and not np.isinf(layer_max):
                    # Update global max
                    self.global_max = max(self.global_max, layer_max)
                    
                    # Track per-layer max
                    if name not in self.layer_max:
                        self.layer_max[name] = layer_max
                    else:
                        self.layer_max[name] = max(self.layer_max[name], layer_max)
        
        return hook

#==================
# Collect Max Activation
#------------------
print("\n" + "="*60)
print("COLLECTING MAXIMUM ACTIVATION VALUES")
print("="*60)

tracker = GlobalMaxActivationTracker()
hooks = []

# Register hooks on all computational layers
for name, module in model.named_modules():
    if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d, 
                           torch.nn.Linear, torch.nn.ConvTranspose2d)) or \
       'SpectralConv' in type(module).__name__:
        hook = module.register_forward_hook(tracker.hook_fn(name))
        hooks.append(hook)

print(f"Registered hooks on {len(hooks)} layers")

# Run inference on samples
print(f"\nProcessing {NUM_SAMPLES} samples...")
sample_count = 0

torch.manual_seed(SEED)
np.random.seed(SEED)

for idx, data in enumerate(test_loader):
    if sample_count >= NUM_SAMPLES:
        break
    
    data = data_processor.preprocess(data, batched=True)
    x = data['x'].to(device)
    
    with torch.no_grad():
        _ = model(x)
    
    sample_count += 1
    
    if (sample_count % 10 == 0):
        print(f"  Processed {sample_count}/{NUM_SAMPLES} samples, "
              f"current max: {tracker.global_max:.6e}")

# Remove hooks
for hook in hooks:
    hook.remove()

print(f"\nProcessed {sample_count} samples")
print(f"\nGlobal maximum activation value: {tracker.global_max:.6e}")

#==================
# Save Results
#------------------
results = {
    'global_max_activation': float(tracker.global_max),
    'num_samples_analyzed': sample_count,
    'model_path': model_path,
    'layer_max_activations': {name: float(val) for name, val in tracker.layer_max.items()}
}

with open(OUTPUT_FILE, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to: {OUTPUT_FILE}")

# Print top 10 layers by max activation
print("\nTop 10 layers by maximum activation:")
sorted_layers = sorted(tracker.layer_max.items(), key=lambda x: x[1], reverse=True)
for i, (name, val) in enumerate(sorted_layers[:10]):
    print(f"  {i+1}. {name}: {val:.6e}")

print("\n" + "="*60)
print("MAX ACTIVATION COLLECTION COMPLETE")
print("="*60)

