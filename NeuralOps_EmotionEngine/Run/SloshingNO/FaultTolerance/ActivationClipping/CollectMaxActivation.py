# CollectMaxActivation_Sloshing.py

import torch
import numpy as np
import json
import glob
import scipy.io as sio
from neuralop.models import FNO
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Running on device: {device}')

NUM_SAMPLES = 100
SEED = 42
OUTPUT_FILE = 'max_activation_value.json'

def load_sloshing_data(data_path, max_samples=None):
    filepaths = sorted(glob.glob(f"{data_path}/FNO_dataset_run_*.mat"))
    if not filepaths:
        raise FileNotFoundError(f"No .mat files found under {data_path}")
    tensors = []
    for p in filepaths:
        mat = sio.loadmat(p)
        arr = torch.tensor(mat['velocity_field_5D'], dtype=torch.float32)
        tensors.append(arr)
    data = torch.cat(tensors, dim=0)
    if max_samples is not None:
        data = data[:max_samples]
    ds = TensorDataset(data)
    return DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

def get_x(batch):
    return batch[0] if isinstance(batch, (list, tuple)) else batch

print("Loading sloshing dataset...")
data_path = '../../../../Data/Sloshing'
test_loader = load_sloshing_data(data_path, max_samples=NUM_SAMPLES)

print("Loading FNO model...")
model_path = '../../../../Checkpoints/Sloshing/sloshing_fno_state_dict.pt'

model = FNO(
    n_modes=(16, 16, 16),
    in_channels=3,
    out_channels=3,
    hidden_channels=32,
    n_layers=4
).to(device)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
model.eval()
print("Model loaded successfully")

class GlobalMaxActivationTracker:
    def __init__(self):
        self.global_max = float('-inf')
        self.layer_max = {}
    def hook_fn(self, name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                layer_max = output.abs().max().item()
                if not np.isnan(layer_max) and not np.isinf(layer_max):
                    self.global_max = max(self.global_max, layer_max)
                    if name not in self.layer_max:
                        self.layer_max[name] = layer_max
                    else:
                        self.layer_max[name] = max(self.layer_max[name], layer_max)
        return hook

print("\n" + "="*60)
print("COLLECTING MAXIMUM ACTIVATION VALUES")
print("="*60)

tracker = GlobalMaxActivationTracker()
hooks = []

for name, module in model.named_modules():
    if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d,
                           torch.nn.Linear, torch.nn.ConvTranspose2d)) or \
       'SpectralConv' in type(module).__name__:
        hook = module.register_forward_hook(tracker.hook_fn(name))
        hooks.append(hook)

print(f"Registered hooks on {len(hooks)} layers")

print(f"\nProcessing {NUM_SAMPLES} samples...")
sample_count = 0

torch.manual_seed(SEED)
np.random.seed(SEED)

for idx, batch in enumerate(test_loader):
    if sample_count >= NUM_SAMPLES:
        break
    x = get_x(batch).to(device)
    with torch.no_grad():
        _ = model(x)
    sample_count += 1
    if (sample_count % 10 == 0):
        print(f"  Processed {sample_count}/{NUM_SAMPLES} samples, current max: {tracker.global_max:.6e}")

for hook in hooks:
    hook.remove()

print(f"\nProcessed {sample_count} samples")
print(f"\nGlobal maximum activation value: {tracker.global_max:.6e}")

results = {
    'global_max_activation': float(tracker.global_max),
    'num_samples_analyzed': sample_count,
    'model_path': model_path,
    'layer_max_activations': {name: float(val) for name, val in tracker.layer_max.items()}
}

with open(OUTPUT_FILE, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to: {OUTPUT_FILE}")

print("\nTop 10 layers by maximum activation:")
sorted_layers = sorted(tracker.layer_max.items(), key=lambda x: x[1], reverse=True)
for i, (name, val) in enumerate(sorted_layers[:10]):
    print(f"  {i+1}. {name}: {val:.6e}")

print("\n" + "="*60)
print("MAX ACTIVATION COLLECTION COMPLETE")
print("="*60)

