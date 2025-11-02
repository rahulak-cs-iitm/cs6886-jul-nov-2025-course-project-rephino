# ActivationClipping_Sloshing.py

import torch
import numpy as np
import random
import struct
import json
import glob
import scipy.io as sio
import matplotlib.pyplot as plt
import os
import time
from neuralop.models import FNO
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Running on device: {device}')

NUM_TRIALS = 100
SEED = 42
TEST_SAMPLES = 50
MAX_ACT_FILE = 'max_activation_value.json'
OUTPUT_DIR = 'plots_sloshing'

BIT_QUADRANTS = {
    'Q1 (bits 0-7)': list(range(0, 8)),
    'Q2 (bits 8-15)': list(range(8, 16)),
    'Q3 (bits 16-23)': list(range(16, 24)),
    'Q4 (bits 24-31)': list(range(24, 32)),
}

print(f"Loading max activations from: {MAX_ACT_FILE}")
with open(MAX_ACT_FILE, 'r') as f:
    max_act_data = json.load(f)

GLOBAL_MAX_ACTIVATION = max_act_data['global_max_activation']
LAYER_MAX_ACTIVATIONS = max_act_data['layer_max_activations']

print(f"Global max activation: {GLOBAL_MAX_ACTIVATION:.6e}")
print(f"Loaded per-layer max for {len(LAYER_MAX_ACTIVATIONS)} layers")

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

test_loader = load_sloshing_data('../../../../Data/Sloshing', max_samples=TEST_SAMPLES*2)

def float_to_int(f):
    return struct.unpack('<I', struct.pack('<f', f))[0]

def int_to_float(i):
    return struct.unpack('<f', struct.pack('<I', i))[0]

def single_bit_flip_quadrant(original_float, quadrant_bits):
    if not isinstance(original_float, float):
        original_float = float(original_float)
    original_int = float_to_int(original_float)
    bit_to_flip = random.choice(quadrant_bits)
    flipper_mask = 1 << bit_to_flip
    flipped_int = original_int ^ flipper_mask
    flipped_float = int_to_float(flipped_int)
    return flipped_float, bit_to_flip

class ActivationFaultInjectorPerLayer:
    def __init__(self, layer_name, layer_module, quadrant_bits, enable_clipping=False, 
                 layer_max_dict=None, global_max=None):
        self.layer_name = layer_name
        self.layer_module = layer_module
        self.quadrant_bits = quadrant_bits
        self.enable_clipping = enable_clipping
        if layer_max_dict and layer_name in layer_max_dict:
            self.max_ceiling = layer_max_dict[layer_name]
        else:
            self.max_ceiling = global_max
        self.hook_handle = None
        self.bit_flipped = None
        self.clipping_applied = False
    
    def single_bit_flip_hook(self, module, input, output):
        if output.numel() == 0:
            return
        shape = output.shape
        rand_idx = tuple([0] + [random.randint(0, dim - 1) for dim in shape[1:]])
        original_val = output[rand_idx].item()
        flipped_val, bit_pos = single_bit_flip_quadrant(original_val, self.quadrant_bits)
        self.bit_flipped = bit_pos
        if self.enable_clipping and self.max_ceiling is not None:
            if flipped_val != flipped_val or flipped_val == float('inf') or flipped_val == float('-inf'):
                self.clipping_applied = True
                flipped_val = self.max_ceiling
            elif abs(flipped_val) > self.max_ceiling:
                self.clipping_applied = True
                flipped_val = self.max_ceiling if flipped_val > 0 else -self.max_ceiling
        with torch.no_grad():
            output[rand_idx] = flipped_val
    
    def register_hook(self):
        self.hook_handle = self.layer_module.register_forward_hook(self.single_bit_flip_hook)
    
    def remove_hook(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

class ActivationClippingModule:
    def __init__(self, model, layer_max_dict, global_max):
        self.model = model
        self.layer_max_dict = layer_max_dict
        self.global_max = global_max
        self.hooks = []
        self._register_clipping_hooks()
    
    def _clipping_hook(self, layer_name, max_val):
        def hook(module, input, output):
            return torch.clamp(output, -max_val, max_val)
        return hook
    
    def _register_clipping_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d,
                                   torch.nn.Linear, torch.nn.ConvTranspose2d)) or \
               'SpectralConv' in type(module).__name__:
                max_val = self.layer_max_dict.get(name, self.global_max)
                hook = module.register_forward_hook(self._clipping_hook(name, max_val))
                self.hooks.append(hook)
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def __call__(self, x):
        return self.model(x)

def get_injectable_layers(model):
    injectable_layers = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d,
                               torch.nn.Linear, torch.nn.ConvTranspose2d)) or \
           'SpectralConv' in type(module).__name__:
            injectable_layers.append((name, module))
    seen = set()
    unique_layers = []
    for name, module in injectable_layers:
        if name not in seen:
            seen.add(name)
            unique_layers.append((name, module))
    return unique_layers

def calculate_mse(output1, output2):
    if torch.isnan(output1).any() or torch.isnan(output2).any():
        return np.nan
    if torch.isinf(output1).any() or torch.isinf(output2).any():
        return np.nan
    mse = torch.mean((output1 - output2) ** 2).item()
    if np.isnan(mse) or np.isinf(mse):
        return np.nan
    return mse

def benchmark_clipping_overhead(model_path, test_loader, num_samples=100, warmup=10):
    print("\n" + "="*80)
    print("ACTIVATION CLIPPING INFERENCE TIME BENCHMARK")
    print("="*80)
    
    model = FNO(n_modes=(16, 16, 16), in_channels=3, out_channels=3,
                hidden_channels=32, n_layers=4)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()
    
    original_model = model
    clipping_model = ActivationClippingModule(model, LAYER_MAX_ACTIVATIONS, GLOBAL_MAX_ACTIVATION)
    
    test_inputs = []
    for idx, batch in enumerate(test_loader):
        if idx >= num_samples:
            break
        x = get_x(batch).to(device)
        test_inputs.append(x)
    
    print(f"\nBenchmarking on {len(test_inputs)} samples (with {warmup} warmup iterations)")
    
    print("\nWarming up original model...")
    for i in range(min(warmup, len(test_inputs))):
        with torch.no_grad():
            _ = original_model(test_inputs[i % len(test_inputs)])
    
    print("Benchmarking original model (no clipping)...")
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start_time = time.time()
    for x in test_inputs:
        with torch.no_grad():
            _ = original_model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    original_time = time.time() - start_time
    original_time_per_sample = original_time / len(test_inputs)
    
    print("Warming up model with clipping...")
    for i in range(min(warmup, len(test_inputs))):
        with torch.no_grad():
            _ = clipping_model(test_inputs[i % len(test_inputs)])
    
    print("Benchmarking model with activation clipping...")
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start_time = time.time()
    for x in test_inputs:
        with torch.no_grad():
            _ = clipping_model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    clipping_time = time.time() - start_time
    clipping_time_per_sample = clipping_time / len(test_inputs)
    
    time_overhead = clipping_time - original_time
    time_overhead_pct = (clipping_time / original_time - 1) * 100
    
    clipping_model.remove_hooks()
    
    print("\n" + "-"*80)
    print("BENCHMARK RESULTS")
    print("-"*80)
    print(f"Original Model (No Clipping):")
    print(f"  Total time:       {original_time:.4f} seconds")
    print(f"  Time per sample:  {original_time_per_sample*1000:.2f} ms")
    print(f"  Throughput:       {len(test_inputs)/original_time:.2f} samples/sec")
    print(f"\nModel with Activation Clipping:")
    print(f"  Total time:       {clipping_time:.4f} seconds")
    print(f"  Time per sample:  {clipping_time_per_sample*1000:.2f} ms")
    print(f"  Throughput:       {len(test_inputs)/clipping_time:.2f} samples/sec")
    print(f"\nOverhead:")
    print(f"  Additional time:  {time_overhead:.4f} seconds")
    print(f"  Time overhead:    {time_overhead_pct:.2f}%")
    print(f"  Slowdown factor:  {clipping_time/original_time:.2f}x")
    print("-"*80)
    
    return {
        'original_time': original_time,
        'clipping_time': clipping_time,
        'overhead_pct': time_overhead_pct,
        'original_per_sample': original_time_per_sample,
        'clipping_per_sample': clipping_time_per_sample
    }

def analyze_activation_with_mse_tracking(model_path, test_loader):
    print("\n" + "="*60)
    print("ACTIVATION FAULT INJECTION - MSE ANALYSIS")
    print("="*60)
    
    model = FNO(n_modes=(16, 16, 16), in_channels=3, out_channels=3,
                hidden_channels=32, n_layers=4)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()
    
    injectable_layers = get_injectable_layers(model)
    print(f"\nFound {len(injectable_layers)} injectable layers")
    
    print("\nCollecting baseline outputs...")
    baseline_outputs = []
    test_inputs = []
    
    for idx, batch in enumerate(test_loader):
        if idx >= TEST_SAMPLES:
            break
        x = get_x(batch).to(device)
        test_inputs.append(x)
        with torch.no_grad():
            out = model(x)
            baseline_outputs.append(out.cpu())
    
    print(f"Collected {len(baseline_outputs)} baseline outputs")
    
    comparison_results = {}
    
    for quad_name, quad_bits in BIT_QUADRANTS.items():
        print(f"\n{'='*60}")
        print(f"Quadrant: {quad_name}")
        print(f"{'='*60}")
        
        layer_results = {}
        
        for layer_idx, (layer_name, layer_module) in enumerate(injectable_layers):
            layer_max = LAYER_MAX_ACTIVATIONS.get(layer_name, GLOBAL_MAX_ACTIVATION)
            print(f"\nLayer {layer_idx+1}/{len(injectable_layers)}: {layer_name}")
            
            results_without = {'nan_count': 0, 'mse_values': [], 'valid_mse_count': 0}
            results_with = {'nan_count': 0, 'mse_values': [], 'valid_mse_count': 0, 'clipping_events': 0}
            
            for trial in tqdm(range(NUM_TRIALS), desc="  No-clip", leave=False):
                seed_val = (SEED + trial + layer_idx * 1000 + abs(hash(quad_name))) % (2**32)
                random.seed(seed_val); np.random.seed(seed_val); torch.manual_seed(seed_val)
                
                injector = ActivationFaultInjectorPerLayer(layer_name, layer_module, quad_bits, 
                                                           enable_clipping=False, layer_max_dict=LAYER_MAX_ACTIVATIONS, 
                                                           global_max=GLOBAL_MAX_ACTIVATION)
                injector.register_hook()
                
                trial_mse = []
                trial_has_nan = False
                for test_idx in range(len(test_inputs)):
                    with torch.no_grad():
                        faulty_output = model(test_inputs[test_idx])
                    mse = calculate_mse(faulty_output.cpu(), baseline_outputs[test_idx])
                    if np.isnan(mse):
                        trial_has_nan = True; break
                    trial_mse.append(mse)
                
                injector.remove_hook()
                if trial_has_nan:
                    results_without['nan_count'] += 1
                else:
                    avg_mse = np.mean(trial_mse)
                    if not np.isnan(avg_mse) and not np.isinf(avg_mse):
                        results_without['mse_values'].append(avg_mse)
                        results_without['valid_mse_count'] += 1
            
            for trial in tqdm(range(NUM_TRIALS), desc="  Clipped", leave=False):
                seed_val = (SEED + trial + layer_idx * 1000 + abs(hash(quad_name))) % (2**32)
                random.seed(seed_val); np.random.seed(seed_val); torch.manual_seed(seed_val)
                
                injector = ActivationFaultInjectorPerLayer(layer_name, layer_module, quad_bits,
                                                           enable_clipping=True, layer_max_dict=LAYER_MAX_ACTIVATIONS,
                                                           global_max=GLOBAL_MAX_ACTIVATION)
                injector.register_hook()
                
                trial_mse = []
                trial_has_nan = False
                for test_idx in range(len(test_inputs)):
                    with torch.no_grad():
                        faulty_output = model(test_inputs[test_idx])
                    mse = calculate_mse(faulty_output.cpu(), baseline_outputs[test_idx])
                    if np.isnan(mse):
                        trial_has_nan = True; break
                    trial_mse.append(mse)
                
                if injector.clipping_applied:
                    results_with['clipping_events'] += 1
                injector.remove_hook()
                
                if trial_has_nan:
                    results_with['nan_count'] += 1
                else:
                    avg_mse = np.mean(trial_mse)
                    if not np.isnan(avg_mse) and not np.isinf(avg_mse):
                        results_with['mse_values'].append(avg_mse)
                        results_with['valid_mse_count'] += 1
            
            layer_results[layer_name] = {'without_clipping': results_without, 'with_clipping': results_with}
            
            nan_reduction = results_without['nan_count'] - results_with['nan_count']
            if len(results_without['mse_values']) > 0:
                mse_without_mean, mse_without_std = np.mean(results_without['mse_values']), np.std(results_without['mse_values'])
            else:
                mse_without_mean, mse_without_std = np.nan, np.nan
            if len(results_with['mse_values']) > 0:
                mse_with_mean, mse_with_std = np.mean(results_with['mse_values']), np.std(results_with['mse_values'])
            else:
                mse_with_mean, mse_with_std = np.nan, np.nan
            
            print(f"  NaNs: Without={results_without['nan_count']}, With={results_with['nan_count']}, Reduced={nan_reduction}")
            print(f"  MSE (without): {mse_without_mean:.6e} ± {mse_without_std:.6e} (n={results_without['valid_mse_count']})")
            print(f"  MSE (with):    {mse_with_mean:.6e} ± {mse_with_std:.6e} (n={results_with['valid_mse_count']})")
        
        comparison_results[quad_name] = layer_results
    
    return comparison_results, injectable_layers

def plot_nan_and_mse_comparison(comparison_results):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Activation Clipping: NaN Reduction vs MSE Impact', fontsize=16, fontweight='bold')
    
    quadrants = list(BIT_QUADRANTS.keys())
    
    for idx, quad_name in enumerate(quadrants):
        ax = axes[idx // 2, idx % 2]
        
        layer_results = comparison_results[quad_name]
        total_nans_without = sum(r['without_clipping']['nan_count'] for r in layer_results.values())
        total_nans_with = sum(r['with_clipping']['nan_count'] for r in layer_results.values())
        mse_without_list = [m for r in layer_results.values() for m in r['without_clipping']['mse_values']]
        mse_with_list = [m for r in layer_results.values() for m in r['with_clipping']['mse_values']]
        
        total_trials = NUM_TRIALS * len(layer_results)
        nan_reduction = total_nans_without - total_nans_with
        nan_reduction_pct = (nan_reduction / total_trials) * 100
        
        
        if len(mse_without_list) > 0:
            mse_without_mean, mse_without_std = np.mean(mse_without_list), np.std(mse_without_list)
        else:
            mse_without_mean, mse_without_std = 0, 0
        
        if len(mse_with_list) > 0:
            mse_with_mean, mse_with_std = np.mean(mse_with_list), np.std(mse_with_list)
        else:
            mse_with_mean, mse_with_std = 0, 0
        
        ax2 = ax.twinx()
        categories = ['Without\nClipping', 'With\nClipping', 'NaNs\nSaved']
        nan_values = [(total_nans_without / total_trials) * 100, (total_nans_with / total_trials) * 100, nan_reduction_pct]
        colors = ['#d62728', '#ff7f0e', '#2ca02c']
        
        bars = ax.bar(categories[:2], nan_values[:2], color=colors[:2], alpha=0.7, label='NaN Rate (%)')
        ax.bar(categories[2], nan_values[2], color=colors[2], alpha=0.7, label='NaN Reduction (%)')
        
        if len(mse_without_list) > 0 or len(mse_with_list) > 0:
            mse_categories = ['Without\nClipping', 'With\nClipping']
            mse_means = [mse_without_mean, mse_with_mean]
            mse_stds = [mse_without_std, mse_with_std]
            x_pos = np.arange(len(mse_categories))
            ax2.errorbar(x_pos, mse_means, yerr=mse_stds, fmt='o-', color='blue', linewidth=2, markersize=8, capsize=5, label='MSE')
        
        ax.set_xlabel('Metric', fontsize=11)
        ax.set_ylabel('NaN Rate / Reduction (%)', fontsize=11, color='black')
        ax2.set_ylabel('Mean Squared Error', fontsize=11, color='blue')
        ax.set_title(f'{quad_name}', fontsize=12, fontweight='bold')
        ax.tick_params(axis='y', labelcolor='black')
        ax2.tick_params(axis='y', labelcolor='blue')
        ax.grid(axis='y', alpha=0.3)
        
        for i, (bar, val) in enumerate(zip(bars, nan_values[:2])):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
        
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    filename = os.path.join(OUTPUT_DIR, 'activation_clipping_analysis.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {filename}")
    return fig

def print_summary_table(comparison_results):
    print("\n" + "="*80)
    print("SUMMARY: NaN REDUCTION AND MSE IMPACT")
    print("="*80)
    
    print(f"\n{'Quadrant':<20} {'NaN%':<10} {'→':<5} {'NaN%':<10} {'Saved%':<10} {'MSE (no clip)':<15} {'MSE (clip)':<15}")
    print("-" * 80)
    
    for quad_name in BIT_QUADRANTS.keys():
        layer_results = comparison_results[quad_name]
        total_nans_without = sum(r['without_clipping']['nan_count'] for r in layer_results.values())
        total_nans_with = sum(r['with_clipping']['nan_count'] for r in layer_results.values())
        mse_without = [m for r in layer_results.values() for m in r['without_clipping']['mse_values']]
        mse_with = [m for r in layer_results.values() for m in r['with_clipping']['mse_values']]
        
        total_trials = NUM_TRIALS * len(layer_results)
        nan_without_pct, nan_with_pct = (total_nans_without / total_trials) * 100, (total_nans_with / total_trials) * 100
        saved_pct = ((total_nans_without - total_nans_with) / total_trials) * 100
        
        mse_without_mean = np.mean(mse_without) if len(mse_without) > 0 else np.nan
        mse_with_mean = np.mean(mse_with) if len(mse_with) > 0 else np.nan
        
        print(f"{quad_name:<20} {nan_without_pct:>7.1f}%  {'→':<5} {nan_with_pct:>7.1f}%  {saved_pct:>7.1f}%  {mse_without_mean:>13.6e}  {mse_with_mean:>13.6e}")
    
    print("="*80)

if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    model_path = '../../../../Checkpoints/Sloshing/sloshing_fno_state_dict.pt'
    
    timing_results = benchmark_clipping_overhead(model_path, test_loader, num_samples=100, warmup=10)
    comparison_results, injectable_layers = analyze_activation_with_mse_tracking(model_path, test_loader)
    plot_nan_and_mse_comparison(comparison_results)
    print_summary_table(comparison_results)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

