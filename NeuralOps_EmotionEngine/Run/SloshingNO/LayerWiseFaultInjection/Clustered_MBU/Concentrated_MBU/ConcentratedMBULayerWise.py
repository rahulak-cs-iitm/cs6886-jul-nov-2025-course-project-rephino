# ClusteredMBULayerWise_Sloshing.py
#   1. Simulate clustered multi-bit upsets (2-5 bits) on different layers 
#      of the Neural Operator for weights (real and complex) and activations
#   2. MSE Analysis based on quadrants of bits (B0-B7, B8-15, B16-23, B24-31)
#   3. Separate analysis for real vs complex components of SpectralConv weights
#   4. Bit flip probability: 60% (2-bit), 20% (3-bit), 10% (4-bit), 10% (5-bit)

import torch
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import random
import struct
import glob  
import scipy.io as sio  
from neuralop.models import FNO
from torch.utils.data import DataLoader, TensorDataset  
from tqdm import tqdm
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Running on device: {device}')

#==================
# Configuration
#------------------
NUM_TRIALS = 100
SEED = 42
TEST_SAMPLES = 50

# Define bit quadrants
BIT_QUADRANTS = {
    'Q1 (bits 0-7)': list(range(0, 8)),
    'Q2 (bits 8-15)': list(range(8, 16)),
    'Q3 (bits 16-23)': list(range(16, 24)),
    'Q4 (bits 24-31)': list(range(24, 32)),
}

# Multi-bit upset probability distribution
MBU_PROBABILITIES = {
    2: 0.60,
    3: 0.20,
    4: 0.10,
    5: 0.10,
}

#==================
# Get-Set Data (UPDATED for slosh FNO)
#------------------
def load_sloshing_data(data_path, num_samples):
    """
    Load sloshing data from .mat files:
      - Expects FNO_dataset_run_*.mat files
      - Uses variable 'velocity_field_5D' stacked across files
    Returns a DataLoader of tensors with batch_size=1
    """
    print(f"Loading data from: {data_path}  ")
    filepaths = sorted(glob.glob(os.path.join(data_path, "FNO_dataset_run_*.mat")))
    if not filepaths:
        raise FileNotFoundError(f"No .mat files found in {data_path}")

    all_tensors = []
    for p in filepaths:
        try:
            mat_data = sio.loadmat(p)
            tensor_data = torch.tensor(mat_data['velocity_field_5D'], dtype=torch.float)
            all_tensors.append(tensor_data)
        except Exception as e:
            print(f"Warning: Error loading {p}: {e}")

    full_data_sequence = torch.cat(all_tensors, dim=0)
    print(f"Full data sequence shape: {full_data_sequence.shape}  ")

    dataset = TensorDataset(full_data_sequence)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    return loader

# UPDATED for slosh FNO
data_path = '../../../../../Data/Sloshing'
test_loader = load_sloshing_data(data_path, TEST_SAMPLES)

#==================
# Bit-Flip Functions for Real Values
#------------------
def float_to_int(f):
    return struct.unpack('<I', struct.pack('<f', f))[0]

def int_to_float(i):
    return struct.unpack('<f', struct.pack('<I', i))[0]

def is_valid_float(f):
    return not (np.isnan(f) or np.isinf(f))

def sample_num_bit_flips():
    rand_val = random.random()
    cumulative_prob = 0.0
    for num_bits, prob in sorted(MBU_PROBABILITIES.items()):
        cumulative_prob += prob
        if rand_val <= cumulative_prob:
            return num_bits
    return 5

def multi_bit_flip_quadrant(original_float, quadrant_bits, n_bits):
    if not isinstance(original_float, float):
        original_float = float(original_float)
    original_int = float_to_int(original_float)
    n_bits = min(n_bits, len(quadrant_bits))
    bits_to_flip = random.sample(quadrant_bits, n_bits)
    flipped_int = original_int
    for bit_pos in bits_to_flip:
        flipper_mask = 1 << bit_pos
        flipped_int ^= flipper_mask
    flipped_float = int_to_float(flipped_int)
    return flipped_float, bits_to_flip

#==================
# Bit-Flip Functions for Complex Values
#------------------
def multi_bit_flip_complex(original_complex, quadrant_bits, n_bits, component='random'):
    if component == 'random':
        component = random.choice(['real', 'imag'])
    device_val = original_complex.device
    if component == 'real':
        real_part = original_complex.real.item()
        flipped_real, bit_positions = multi_bit_flip_quadrant(float(real_part), quadrant_bits, n_bits)
        result = torch.complex(torch.tensor(flipped_real, device=device_val), original_complex.imag)
    else:
        imag_part = original_complex.imag.item()
        flipped_imag, bit_positions = multi_bit_flip_quadrant(float(imag_part), quadrant_bits, n_bits)
        result = torch.complex(original_complex.real, torch.tensor(flipped_imag, device=device_val))
    return result, component, bit_positions

#==================
# Layer Extraction (extended to Conv3d)  
#------------------
def get_injectable_layers(model):
    injectable_layers = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d,
                               torch.nn.Linear, torch.nn.ConvTranspose2d)):
            if hasattr(module, 'weight') and module.weight is not None:
                injectable_layers.append((name, module, 'real'))
        elif 'SpectralConv' in type(module).__name__:
            if hasattr(module, 'weight') and module.weight is not None:
                injectable_layers.append((name + ' [Real]', module, 'complex_real'))
                injectable_layers.append((name + ' [Imag]', module, 'complex_imag'))
    return injectable_layers

#==================
# Weight Fault Injection
#------------------
def inject_weight_fault_multibit_quadrant(model, layer_name, layer_module, weight_type, quadrant_bits):
    if not hasattr(layer_module, 'weight') or layer_module.weight is None:
        return None, None, None, 0
    weight_obj = layer_module.weight
    if hasattr(weight_obj, 'tensor'):
        weight = weight_obj.tensor
    elif hasattr(weight_obj, 'data'):
        weight = weight_obj.data
    else:
        weight = weight_obj
    shape = weight.shape
    rand_idx = tuple(random.randint(0, dim - 1) for dim in shape)
    original_val = weight[rand_idx].clone()
    n_bits = sample_num_bit_flips()
    if weight_type == 'real':
        original_float = original_val.item()
        flipped_val, bit_positions = multi_bit_flip_quadrant(original_float, quadrant_bits, n_bits)
        with torch.no_grad():
            weight[rand_idx] = flipped_val
        return rand_idx, original_val, bit_positions, n_bits
    elif weight_type == 'complex_real':
        flipped_val, component, bit_positions = multi_bit_flip_complex(
            weight[rand_idx], quadrant_bits, n_bits, component='real')
        with torch.no_grad():
            weight[rand_idx] = flipped_val
        return rand_idx, original_val, bit_positions, n_bits
    elif weight_type == 'complex_imag':
        flipped_val, component, bit_positions = multi_bit_flip_complex(
            weight[rand_idx], quadrant_bits, n_bits, component='imag')
        with torch.no_grad():
            weight[rand_idx] = flipped_val
        return rand_idx, original_val, bit_positions, n_bits

#==================
# Activation Fault Injection (Always Real-Valued)
#------------------
class ActivationFaultInjectorMultiBitQuadrant:
    def __init__(self, layer_module, quadrant_bits):
        self.layer_module = layer_module
        self.quadrant_bits = quadrant_bits
        self.hook_handle = None
        self.bits_flipped = None
        self.n_bits_flipped = 0
    def multi_bit_flip_hook(self, module, input, output):
        if output.numel() == 0:
            return
        shape = output.shape
        rand_idx = tuple([0] + [random.randint(0, dim - 1) for dim in shape[1:]])
        original_val = output[rand_idx].item()
        n_bits = sample_num_bit_flips()
        self.n_bits_flipped = n_bits
        flipped_val, bit_positions = multi_bit_flip_quadrant(original_val, self.quadrant_bits, n_bits)
        self.bits_flipped = bit_positions
        with torch.no_grad():
            output[rand_idx] = flipped_val
    def register_hook(self):
        self.hook_handle = self.layer_module.register_forward_hook(self.multi_bit_flip_hook)
    def remove_hook(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

#==================
# MSE Calculation
#------------------
def calculate_mse(output1, output2):
    if torch.isnan(output1).any() or torch.isnan(output2).any():
        return np.nan
    if torch.isinf(output1).any() or torch.isinf(output2).any():
        return np.nan
    mse = torch.mean((output1 - output2) ** 2).item()
    if not is_valid_float(mse):
        return np.nan
    return mse

#==================
# Analysis Functions
#------------------
def analyze_weight_sensitivity_multibit_by_quadrant(model_path, test_loader):
    print(f"\n{'='*60}")
    print(f"MULTI-BIT WEIGHT FAULT INJECTION - BY BIT QUADRANT")
    print(f"{'='*60}")
    print(f"MBU Distribution: 2-bit (60%), 3-bit (20%), 4-bit (10%), 5-bit (10%)")
    
    # Load sloshing 3D FNO  
    model = FNO(n_modes=(16, 16, 16), hidden_channels=32, in_channels=3, out_channels=3, n_layers=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()
    
    injectable_layers = get_injectable_layers(model)
    print(f"\nFound {len(injectable_layers)} injectable layer components")
    print("(SpectralConv layers counted twice: Real and Imaginary components)")
    
    print("\nCollecting baseline (fault-free) outputs...")
    baseline_outputs = []
    test_inputs = []
    for idx, data_batch in enumerate(test_loader):
        if idx >= TEST_SAMPLES:
            break
        x = data_batch[0].to(device)
        test_inputs.append(x)
        with torch.no_grad():
            out = model(x)
            baseline_outputs.append(out.cpu())
    print(f"Collected {len(baseline_outputs)} baseline outputs")
    
    quadrant_results = {}
    for quad_name, quad_bits in BIT_QUADRANTS.items():
        print(f"\n{'='*60}")
        print(f"Processing Quadrant: {quad_name}")
        print(f"{'='*60}")
        layer_results = {}
        for layer_idx, (layer_name, layer_module, weight_type) in enumerate(injectable_layers):
            print(f"\nLayer {layer_idx+1}/{len(injectable_layers)}: {layer_name}")
            print(f"  Type: {weight_type}")
            mse_values, nan_count = [], 0
            bit_distribution = {bit: 0 for bit in quad_bits}
            bit_count_distribution = {2: 0, 3: 0, 4: 0, 5: 0}
            for trial in tqdm(range(NUM_TRIALS), desc=f"  Trials"):
                seed_val = (SEED + trial + layer_idx * 1000 + abs(hash(quad_name))) % (2**32)
                random.seed(seed_val); np.random.seed(seed_val); torch.manual_seed(seed_val)
                model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
                model.eval()
                base_layer_name = layer_name.replace(' [Real]', '').replace(' [Imag]', '')
                current_layer = dict(model.named_modules())[base_layer_name]
                idx_tuple, orig_val, bit_positions, n_bits = inject_weight_fault_multibit_quadrant(
                    model, base_layer_name, current_layer, weight_type, quad_bits)
                if idx_tuple is None:
                    continue
                for bit_pos in bit_positions:
                    bit_distribution[bit_pos] += 1
                bit_count_distribution[n_bits] += 1
                trial_mse, trial_has_nan = [], False
                for test_idx in range(len(test_inputs)):
                    with torch.no_grad():
                        faulty_output = model(test_inputs[test_idx])
                    mse = calculate_mse(faulty_output.cpu(), baseline_outputs[test_idx])
                    if np.isnan(mse):
                        trial_has_nan = True
                        break
                    trial_mse.append(mse)
                if trial_has_nan:
                    nan_count += 1
                    continue
                avg_mse = np.mean(trial_mse)
                if is_valid_float(avg_mse):
                    mse_values.append(avg_mse)
                else:
                    nan_count += 1
            layer_results[layer_name] = {
                'mse_values': mse_values,
                'nan_count': nan_count,
                'bit_distribution': bit_distribution,
                'bit_count_distribution': bit_count_distribution,
                'weight_type': weight_type
            }
            valid_count = len(mse_values)
            nan_rate = (nan_count / NUM_TRIALS) * 100
            print(f"  Valid: {valid_count}/{NUM_TRIALS}, NaN Rate: {nan_rate:.1f}%")
            print(f"  Bit count dist: 2={bit_count_distribution[2]}, 3={bit_count_distribution[3]}, 4={bit_count_distribution[4]}, 5={bit_count_distribution[5]}")
            if valid_count > 0:
                print(f"  Mean MSE: {np.mean(mse_values):.6e} ± {np.std(mse_values):.6e}")
        quadrant_results[quad_name] = layer_results
    return quadrant_results, injectable_layers

def analyze_activation_sensitivity_multibit_by_quadrant(model_path, test_loader):
    print(f"\n{'='*60}")
    print(f"MULTI-BIT ACTIVATION FAULT INJECTION - BY BIT QUADRANT")
    print(f"{'='*60}")
    print(f"MBU Distribution: 2-bit (60%), 3-bit (20%), 4-bit (10%), 5-bit (10%)")
    print("Note: All activations are real-valued (even after SpectralConv)")
    
    # Load sloshing 3D FNO  
    model = FNO(n_modes=(16, 16, 16), hidden_channels=32, in_channels=3, out_channels=3, n_layers=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()
    
    unique_modules, seen_modules = [], set()
    for name, module, weight_type in get_injectable_layers(model):
        base_name = name.replace(' [Real]', '').replace(' [Imag]', '')
        if base_name not in seen_modules:
            unique_modules.append((base_name, module))
            seen_modules.add(base_name)
    print(f"\nFound {len(unique_modules)} injectable layers")
    
    print("\nCollecting baseline (fault-free) outputs...")
    baseline_outputs, test_inputs = [], []
    for idx, data_batch in enumerate(test_loader):
        if idx >= TEST_SAMPLES:
            break
        x = data_batch[0].to(device)  
        test_inputs.append(x)
        with torch.no_grad():
            out = model(x)
            baseline_outputs.append(out.cpu())
    print(f"Collected {len(baseline_outputs)} baseline outputs")
    
    quadrant_results = {}
    for quad_name, quad_bits in BIT_QUADRANTS.items():
        print(f"\n{'='*60}")
        print(f"Processing Quadrant: {quad_name}")
        print(f"{'='*60}")
        layer_results = {}
        for layer_idx, (layer_name, layer_module) in enumerate(unique_modules):
            print(f"\nLayer {layer_idx+1}/{len(unique_modules)}: {layer_name}")
            mse_values, nan_count = [], 0
            bit_distribution = {bit: 0 for bit in quad_bits}
            bit_count_distribution = {2: 0, 3: 0, 4: 0, 5: 0}
            for trial in tqdm(range(NUM_TRIALS), desc=f"  Trials"):
                seed_val = (SEED + trial + layer_idx * 1000 + abs(hash(quad_name))) % (2**32)
                random.seed(seed_val); np.random.seed(seed_val); torch.manual_seed(seed_val)
                injector = ActivationFaultInjectorMultiBitQuadrant(layer_module, quad_bits)
                injector.register_hook()
                trial_mse, trial_has_nan = [], False
                for test_idx in range(len(test_inputs)):
                    with torch.no_grad():
                        faulty_output = model(test_inputs[test_idx])
                    mse = calculate_mse(faulty_output.cpu(), baseline_outputs[test_idx])
                    if np.isnan(mse):
                        trial_has_nan = True
                        break
                    trial_mse.append(mse)
                injector.remove_hook()
                if injector.bits_flipped is not None:
                    for bit_pos in injector.bits_flipped:
                        bit_distribution[bit_pos] += 1
                    bit_count_distribution[injector.n_bits_flipped] += 1
                if trial_has_nan:
                    nan_count += 1
                    continue
                avg_mse = np.mean(trial_mse)
                if is_valid_float(avg_mse):
                    mse_values.append(avg_mse)
                else:
                    nan_count += 1
            layer_results[layer_name] = {
                'mse_values': mse_values,
                'nan_count': nan_count,
                'bit_distribution': bit_distribution,
                'bit_count_distribution': bit_count_distribution
            }
            valid_count = len(mse_values)
            nan_rate = (nan_count / NUM_TRIALS) * 100
            print(f"  Valid: {valid_count}/{NUM_TRIALS}, NaN Rate: {nan_rate:.1f}%")
            print(f"  Bit count dist: 2={bit_count_distribution[2]}, 3={bit_count_distribution[3]}, 4={bit_count_distribution[4]}, 5={bit_count_distribution[5]}")
            if valid_count > 0:
                print(f"  Mean MSE: {np.mean(mse_values):.6e} ± {np.std(mse_values):.6e}")
        quadrant_results[quad_name] = layer_results
    return quadrant_results, unique_modules

#==================
# Plotting Functions
#------------------
def plot_quadrant_comparison(quadrant_results, injectable_layers, fault_target, output_dir='plots'):
    os.makedirs(output_dir, exist_ok=True)
    if fault_target == 'weight':
        layer_names = [name for name, _, _ in injectable_layers]
    else:
        layer_names = [name for name, _ in injectable_layers]
    n_layers = len(layer_names)
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()
    for idx, (quad_name, layer_results) in enumerate(quadrant_results.items()):
        ax = axes[idx]
        mean_mse, std_mse, nan_rates, colors = [], [], [], []
        for layer_name in layer_names:
            mse_vals = layer_results[layer_name]['mse_values']
            nan_count = layer_results[layer_name]['nan_count']
            if len(mse_vals) > 0:
                mean_mse.append(np.mean(mse_vals)); std_mse.append(np.std(mse_vals))
            else:
                mean_mse.append(0); std_mse.append(0)
            nan_rate = (nan_count / NUM_TRIALS) * 100
            nan_rates.append(nan_rate)
            if nan_rate > 80: colors.append('darkred')
            elif nan_rate > 50: colors.append('orangered')
            elif nan_rate > 20: colors.append('orange')
            else:
                if fault_target == 'weight' and 'weight_type' in layer_results[layer_name]:
                    wtype = layer_results[layer_name]['weight_type']
                    if wtype == 'complex_real': colors.append('steelblue')
                    elif wtype == 'complex_imag': colors.append('royalblue')
                    else: colors.append('lightsteelblue')
                else:
                    colors.append('steelblue')
        x_pos = np.arange(n_layers)
        ax.bar(x_pos, mean_mse, yerr=std_mse, capsize=3, alpha=0.7, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Layer Index', fontsize=10)
        ax.set_ylabel('Mean Squared Error', fontsize=10)
        ax.set_title(f'{quad_name}', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"L{i}" for i in range(n_layers)], rotation=90, fontsize=7)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        if any(m > 0 for m in mean_mse):
            ax.set_yscale('log')
        from matplotlib.patches import Patch
        if fault_target == 'weight':
            legend_elements = [
                Patch(facecolor='lightsteelblue', label='Real Weights'),
                Patch(facecolor='steelblue', label='Complex (Real part)'),
                Patch(facecolor='royalblue', label='Complex (Imag part)'),
                Patch(facecolor='orange', label='NaN 20-50%'),
                Patch(facecolor='orangered', label='NaN 50-80%'),
                Patch(facecolor='darkred', label='NaN > 80%')
            ]
        else:
            legend_elements = [
                Patch(facecolor='steelblue', label='NaN < 20%'),
                Patch(facecolor='orange', label='NaN 20-50%'),
                Patch(facecolor='orangered', label='NaN 50-80%'),
                Patch(facecolor='darkred', label='NaN > 80%')
            ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=7)
    plt.suptitle(f'Layer-wise Multi-Bit {fault_target.capitalize()} Fault Sensitivity by Bit Quadrant', fontsize=16, fontweight='bold')
    plt.tight_layout()
    filename = os.path.join(output_dir, f'multibit_quadrant_comparison_{fault_target}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nQuadrant comparison plot saved to: {filename}")
    return fig

def plot_real_vs_complex_comparison(quadrant_results, injectable_layers, output_dir='plots'):
    os.makedirs(output_dir, exist_ok=True)
    spectral_layers = [(name, wtype) for name, _, wtype in injectable_layers if 'convs' in name]
    if len(spectral_layers) == 0:
        print("No SpectralConv layers found for comparison")
        return None
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    for idx, (quad_name, layer_results) in enumerate(quadrant_results.items()):
        ax = axes[idx]
        real_mse, imag_mse, layer_labels = [], [], []
        processed = set()
        for layer_name, wtype in spectral_layers:
            base_name = layer_name.replace(' [Real]', '').replace(' [Imag]', '')
            if base_name in processed:
                continue
            processed.add(base_name)
            real_name = base_name + ' [Real]'
            imag_name = base_name + ' [Imag]'
            real_vals = layer_results[real_name]['mse_values']
            imag_vals = layer_results[imag_name]['mse_values']
            real_mse.append(np.mean(real_vals) if len(real_vals) > 0 else 0)
            imag_mse.append(np.mean(imag_vals) if len(imag_vals) > 0 else 0)
            layer_labels.append(base_name.split('.')[-1])
        x = np.arange(len(layer_labels)); width = 0.35
        ax.bar(x - width/2, real_mse, width, label='Real Component', color='steelblue', alpha=0.8)
        ax.bar(x + width/2, imag_mse, width, label='Imag Component', color='royalblue', alpha=0.8)
        ax.set_xlabel('SpectralConv Layer', fontsize=11)
        ax.set_ylabel('Mean Squared Error', fontsize=11)
        ax.set_title(f'{quad_name}', fontsize=12, fontweight='bold')
        ax.set_xticks(x); ax.set_xticklabels(layer_labels)
        ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3, linestyle='--')
        if any(m > 0 for m in real_mse + imag_mse):
            ax.set_yscale('log')
    plt.suptitle('SpectralConv: Real vs Imaginary Component Sensitivity (Multi-Bit Faults)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    filename = os.path.join(output_dir, 'multibit_real_vs_complex_comparison.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Real vs Complex comparison plot saved to: {filename}")
    return fig

def create_summary_dataframe(quadrant_results, injectable_layers, is_weight=True):
    if is_weight:
        layer_names = [name for name, _, _ in injectable_layers]
    else:
        layer_names = [name for name, _ in injectable_layers]
    data = []
    for layer_name in layer_names:
        row = {'Layer': layer_name}
        for quad_name in BIT_QUADRANTS.keys():
            mse_vals = quadrant_results[quad_name][layer_name]['mse_values']
            nan_count = quadrant_results[quad_name][layer_name]['nan_count']
            row[f'{quad_name}_MSE'] = np.mean(mse_vals) if len(mse_vals) > 0 else np.nan
            row[f'{quad_name}_Std'] = np.std(mse_vals) if len(mse_vals) > 0 else np.nan
            row[f'{quad_name}_NaN%'] = (nan_count / NUM_TRIALS) * 100
        data.append(row)
    return pd.DataFrame(data)

#==================
# Main Execution
#------------------
if __name__ == "__main__":
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    
    # UPDATED for slosh FNO
    model_path = '../../../../../Checkpoints/Sloshing/sloshing_fno_state_dict.pt'
    output_dir = 'plots'  
    
    print(f"\nConfiguration:")
    print(f"  Trials per layer per quadrant: {NUM_TRIALS}")
    print(f"  Test samples: {TEST_SAMPLES}")
    print(f"  Multi-Bit Upset (MBU) Distribution:")
    for num_bits, prob in MBU_PROBABILITIES.items():
        print(f"    {num_bits}-bit: {prob*100:.0f}%")
    print(f"  Bit Quadrants:")
    for quad_name, bits in BIT_QUADRANTS.items():
        print(f"    {quad_name}: {bits}")
    
    # ==================
    # WEIGHT FAULT INJECTION
    # ==================
    print("\n" + "="*60)
    print("STARTING MULTI-BIT WEIGHT FAULT INJECTION BY QUADRANT")
    print("="*60)
    weight_results, injectable_layers = analyze_weight_sensitivity_multibit_by_quadrant(model_path, test_loader)
    df_weight_summary = create_summary_dataframe(weight_results, injectable_layers, is_weight=True)
    os.makedirs(output_dir, exist_ok=True)
    df_weight_summary.to_csv(os.path.join(output_dir, 'multibit_weight_quadrant_summary.csv'), index=False)
    print("\nMulti-bit weight results saved to multibit_weight_quadrant_summary.csv")
    plot_quadrant_comparison(weight_results, injectable_layers, 'weight', output_dir)
    plot_real_vs_complex_comparison(weight_results, injectable_layers, output_dir)
    
    # ==================
    # ACTIVATION FAULT INJECTION
    # ==================
    print("\n" + "="*60)
    print("STARTING MULTI-BIT ACTIVATION FAULT INJECTION BY QUADRANT")
    print("="*60)
    activation_results, unique_modules = analyze_activation_sensitivity_multibit_by_quadrant(model_path, test_loader)
    df_activation_summary = create_summary_dataframe(activation_results, unique_modules, is_weight=False)
    df_activation_summary.to_csv(os.path.join(output_dir, 'multibit_activation_quadrant_summary.csv'), index=False)
    print("\nMulti-bit activation results saved to multibit_activation_quadrant_summary.csv")
    plot_quadrant_comparison(activation_results, unique_modules, 'activation', output_dir)
    
    # ==================
    # SUMMARY
    # ==================
    print("\n" + "="*60)
    print("MULTI-BIT QUADRANT ANALYSIS SUMMARY")
    print("="*60)
    print("\n--- Multi-Bit Weight Faults ---")
    print(df_weight_summary.to_string(index=False))
    print("\n\n--- Multi-Bit Activation Faults ---")
    print(df_activation_summary.to_string(index=False))
    
    spectral_layers = [(name, wtype) for name, _, wtype in injectable_layers if 'convs' in name]
    if spectral_layers:
        print("\n" + "="*60)
        print("SPECTRAL CONVOLUTION: REAL vs IMAGINARY COMPONENT SENSITIVITY (MULTI-BIT)")
        print("="*60)
        for quad_name in BIT_QUADRANTS.keys():
            print(f"\n{quad_name}:")
            for layer_name, wtype in spectral_layers:
                results = weight_results[quad_name][layer_name]
                mse_vals = results['mse_values']
                if len(mse_vals) > 0:
                    print(f"  {layer_name}: {np.mean(mse_vals):.6e}")
    
    # Identify most critical quadrant
    print("\n" + "="*60)
    print("MOST CRITICAL BIT QUADRANTS (MULTI-BIT FAULTS)")
    print("="*60)
    for quad_name in BIT_QUADRANTS.keys():
        weight_nan_avg = df_weight_summary[f'{quad_name}_NaN%'].mean()
        activation_nan_avg = df_activation_summary[f'{quad_name}_NaN%'].mean()
        print(f"\n{quad_name}:")
        print(f"  Weight NaN Rate: {weight_nan_avg:.1f}%")
        print(f"  Activation NaN Rate: {activation_nan_avg:.1f}%")
    
    plt.show()

