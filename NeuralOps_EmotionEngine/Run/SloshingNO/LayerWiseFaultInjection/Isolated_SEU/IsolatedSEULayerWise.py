# IsolatedSEULayerWise.py
#   Updated for 3D Sloshing FNO Model
#   1. Simulate isolated bit-flips on different layers of the Neural Operator
#      for weights (real and complex) and activations
#   2. MSE Analysis based on quadrants of bits (B0-B7, B8-B15, B16-B23, B24-B31)
#   3. Separate analysis for real vs complex components of SpectralConv weights
#   4. Identification of vulnerability to Functional Interrupts (FI) or
#      Silent Data Corruption (SDC) by tracking NaN for both weights and activations

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
from neuralop.utils import count_model_params
from torch.utils.data import DataLoader, TensorDataset
from matplotlib.patches import Patch
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
OUTPUT_DIR = 'plots_sloshing'

# Define bit quadrants
BIT_QUADRANTS = {
    'Q1 (bits 0-7)': list(range(0, 8)),
    'Q2 (bits 8-15)': list(range(8, 16)),
    'Q3 (bits 16-23)': list(range(16, 24)),
    'Q4 (bits 24-31)': list(range(24, 32)),
}

#==================
# Data for Sloshing Dataset
#------------------
def load_sloshing_data(data_path, num_samples):
    """Loads and preprocesses data from the .mat files."""
    print(f"Loading data from: {data_path}")
    filepaths = sorted(glob.glob(os.path.join(data_path, "FNO_dataset_run_*.mat")))
    if not filepaths:
        raise FileNotFoundError(f"No .mat files found in {data_path}")

    print(f"Found {len(filepaths)} files. Loading...")
    all_tensors = []
    for p in filepaths:
        try:
            mat_data = sio.loadmat(p)
            # The key for the data is 'velocity_field_5D'
            tensor_data = torch.tensor(mat_data['velocity_field_5D'], dtype=torch.float)
            all_tensors.append(tensor_data)
        except Exception as e:
            print(f"Warning: Error loading {p}: {e}")

    # Concatenate all runs along the time dimension (dim=0)
    full_data_sequence = torch.cat(all_tensors, dim=0)
    print(f"Full data sequence shape: {full_data_sequence.shape}")

    dataset = TensorDataset(full_data_sequence)

    # Create a DataLoader for the test samples
    loader = DataLoader(dataset,
                        batch_size=1,
                        shuffle=False, 
                        num_workers=0,
                        pin_memory=True)

    return loader

data_path = '../../../../Data/Sloshing'
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

def single_bit_flip_quadrant(original_float, quadrant_bits):
    if not isinstance(original_float, float):
        original_float = float(original_float)
    original_int = float_to_int(original_float)
    bit_to_flip = random.choice(quadrant_bits)
    flipper_mask = 1 << bit_to_flip
    flipped_int = original_int ^ flipper_mask
    flipped_float = int_to_float(flipped_int)
    return flipped_float, bit_to_flip

#==================
# Bit-Flip Functions for Complex Values
#------------------
def single_bit_flip_complex(original_complex, quadrant_bits, component='random'):
    if component == 'random':
        component = random.choice(['real', 'imag'])
    if component == 'real':
        real_part = original_complex.real.item()
        flipped_real, bit_pos = single_bit_flip_quadrant(float(real_part), quadrant_bits)
        result = torch.complex(torch.tensor(flipped_real, device=device), original_complex.imag)
    else:
        imag_part = original_complex.imag.item()
        flipped_imag, bit_pos = single_bit_flip_quadrant(float(imag_part), quadrant_bits)
        result = torch.complex(original_complex.real, torch.tensor(flipped_imag, device=device))
    return result, component, bit_pos

#==================
# Layer Extraction for 3D Models
#------------------
def get_injectable_layers(model):
    """Extract all layers that can have faults injected, marking complex layers."""
    injectable_layers = []
    for name, module in model.named_modules():
        # Added Conv3d for the new model
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
def inject_weight_fault_quadrant(model, layer_name, layer_module, weight_type, quadrant_bits):
    if not hasattr(layer_module, 'weight') or layer_module.weight is None:
        return None, None, None
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
    if weight_type == 'real':
        original_float = original_val.item()
        flipped_val, bit_pos = single_bit_flip_quadrant(original_float, quadrant_bits)
        with torch.no_grad():
            weight[rand_idx] = flipped_val
        return rand_idx, original_val, ('real', bit_pos)
    elif weight_type == 'complex_real':
        flipped_val, component, bit_pos = single_bit_flip_complex(
            weight[rand_idx], quadrant_bits, component='real')
        with torch.no_grad():
            weight[rand_idx] = flipped_val
        return rand_idx, original_val, ('complex_real', bit_pos)
    elif weight_type == 'complex_imag':
        flipped_val, component, bit_pos = single_bit_flip_complex(
            weight[rand_idx], quadrant_bits, component='imag')
        with torch.no_grad():
            weight[rand_idx] = flipped_val
        return rand_idx, original_val, ('complex_imag', bit_pos)

#==================
# Activation Fault Injection
#------------------
class ActivationFaultInjectorQuadrant:
    def __init__(self, layer_module, quadrant_bits):
        self.layer_module = layer_module
        self.quadrant_bits = quadrant_bits
        self.hook_handle = None
        self.bit_flipped = None
    def single_bit_flip_hook(self, module, input, output):
        if output.numel() == 0:
            return
        shape = output.shape
        rand_idx = tuple([0] + [random.randint(0, dim - 1) for dim in shape[1:]])
        original_val = output[rand_idx].item()
        flipped_val, bit_pos = single_bit_flip_quadrant(original_val, self.quadrant_bits)
        self.bit_flipped = bit_pos
        with torch.no_grad():
            output[rand_idx] = flipped_val
    def register_hook(self):
        self.hook_handle = self.layer_module.register_forward_hook(self.single_bit_flip_hook)
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
# Analysis Functions for updated model
#------------------
def analyze_weight_sensitivity_by_quadrant(model_path, test_loader):
    print(f"\n{'='*60}")
    print(f"WEIGHT FAULT INJECTION - BY BIT QUADRANT")
    print(f"{'='*60}")

    # Load updated 3D FNO model
    model = FNO(n_modes=(16, 16, 16),
                hidden_channels=32,
                in_channels=3,
                out_channels=3,
                n_layers=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    n_params = count_model_params(model)
    print(f'\nThe model has {n_params} parameters.')
    sys.stdout.flush()
    print('\n### MODEL ###\n', model)
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
        print(f"\n{'='*60}\nProcessing Quadrant: {quad_name}\n{'='*60}")
        layer_results = {}
        for layer_idx, (layer_name, layer_module, weight_type) in enumerate(injectable_layers):
            print(f"\nLayer {layer_idx+1}/{len(injectable_layers)}: {layer_name}")
            print(f"  Type: {weight_type}")
            mse_values = []
            nan_count = 0
            bit_distribution = {bit: 0 for bit in quad_bits}
            for trial in tqdm(range(NUM_TRIALS), desc=f"  Trials"):
                seed_val = (SEED + trial + layer_idx * 1000 + abs(hash(quad_name))) % (2**32)
                random.seed(seed_val)
                np.random.seed(seed_val)
                torch.manual_seed(seed_val)

                model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
                model.eval()

                base_layer_name = layer_name.replace(' [Real]', '').replace(' [Imag]', '')
                current_layer = dict(model.named_modules())[base_layer_name]

                idx, orig_val, bit_info = inject_weight_fault_quadrant(
                    model, base_layer_name, current_layer, weight_type, quad_bits)

                if idx is None:
                    continue

                component, bit_pos = bit_info
                bit_distribution[bit_pos] += 1

                trial_mse = []
                trial_has_nan = False
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
                'weight_type': weight_type
            }
            valid_count = len(mse_values)
            nan_rate = (nan_count / NUM_TRIALS) * 100
            print(f"  Valid: {valid_count}/{NUM_TRIALS}, NaN Rate: {nan_rate:.1f}%")
            if valid_count > 0:
                print(f"  Mean MSE: {np.mean(mse_values):.6e} ± {np.std(mse_values):.6e}")

        quadrant_results[quad_name] = layer_results
    return quadrant_results, injectable_layers

def analyze_activation_sensitivity_by_quadrant(model_path, test_loader):
    print(f"\n{'='*60}\nACTIVATION FAULT INJECTION - BY BIT QUADRANT\n{'='*60}")

    # MODIFIED: Load new 3D FNO model
    model = FNO(n_modes=(16, 16, 16),
                hidden_channels=32,
                in_channels=3,
                out_channels=3,
                n_layers=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()

    unique_modules = []
    seen_modules = set()
    for name, module, weight_type in get_injectable_layers(model):
        base_name = name.replace(' [Real]', '').replace(' [Imag]', '')
        if base_name not in seen_modules:
            unique_modules.append((base_name, module))
            seen_modules.add(base_name)
    print(f"\nFound {len(unique_modules)} injectable layers")

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
        print(f"\n{'='*60}\nProcessing Quadrant: {quad_name}\n{'='*60}")
        layer_results = {}
        for layer_idx, (layer_name, layer_module) in enumerate(unique_modules):
            print(f"\nLayer {layer_idx+1}/{len(unique_modules)}: {layer_name}")
            mse_values = []
            nan_count = 0
            bit_distribution = {bit: 0 for bit in quad_bits}
            for trial in tqdm(range(NUM_TRIALS), desc=f"  Trials"):
                seed_val = (SEED + trial + layer_idx * 1000 + abs(hash(quad_name))) % (2**32)
                random.seed(seed_val)
                np.random.seed(seed_val)
                torch.manual_seed(seed_val)

                injector = ActivationFaultInjectorQuadrant(layer_module, quad_bits)
                injector.register_hook()

                trial_mse = []
                trial_has_nan = False
                for test_idx in range(len(test_inputs)):
                    with torch.no_grad():
                        faulty_output = model(test_inputs[test_idx])
                    mse = calculate_mse(faulty_output.cpu(), baseline_outputs[test_idx])
                    if np.isnan(mse):
                        trial_has_nan = True
                        break
                    trial_mse.append(mse)

                injector.remove_hook()
                if injector.bit_flipped is not None:
                    bit_distribution[injector.bit_flipped] += 1
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
                'bit_distribution': bit_distribution
            }
            valid_count = len(mse_values)
            nan_rate = (nan_count / NUM_TRIALS) * 100
            print(f"  Valid: {valid_count}/{NUM_TRIALS}, NaN Rate: {nan_rate:.1f}%")
            if valid_count > 0:
                print(f"  Mean MSE: {np.mean(mse_values):.6e} ± {np.std(mse_values):.6e}")
        quadrant_results[quad_name] = layer_results
    return quadrant_results, unique_modules

#==================
# Plotting and Summary Functions (Unchanged, but with a note on labels)
#------------------
# Note: The 'simplified_labels' logic in plot_overall_mse_comparison might need
# adjustment if the new 3D FNO model has a significantly different layer naming scheme.
# For now, it will use the default fallbacks which should still be informative.
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
                mean_mse.append(np.mean(mse_vals))
                std_mse.append(np.std(mse_vals))
            else:
                mean_mse.append(0)
                std_mse.append(0)
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
                else: colors.append('steelblue')
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
        if fault_target == 'weight':
            legend_elements = [Patch(facecolor='lightsteelblue', label='Real'), Patch(facecolor='steelblue', label='Complex (Real)'), Patch(facecolor='royalblue', label='Complex (Imag)'), Patch(facecolor='orange', label='NaN 20-50%'), Patch(facecolor='orangered', label='NaN 50-80%'), Patch(facecolor='darkred', label='NaN > 80%')]
        else:
            legend_elements = [Patch(facecolor='steelblue', label='NaN < 20%'), Patch(facecolor='orange', label='NaN 20-50%'), Patch(facecolor='orangered', label='NaN 50-80%'), Patch(facecolor='darkred', label='NaN > 80%')]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=7)
    plt.suptitle(f'Layer-wise {fault_target.capitalize()} Fault Sensitivity by Bit Quadrant', fontsize=16, fontweight='bold')
    plt.tight_layout()
    filename = os.path.join(output_dir, f'quadrant_comparison_{fault_target}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nQuadrant comparison plot saved to: {filename}")
    return fig

def plot_overall_mse_comparison(quadrant_results, injectable_layers, fault_target, output_dir='plots'):
    os.makedirs(output_dir, exist_ok=True)
    if fault_target == 'weight':
        layer_names = [name for name, _, _ in injectable_layers]
    else:
        layer_names = [name for name, _ in injectable_layers]
    overall_mean, overall_std, overall_nan_rate = [], [], []
    for layer_name in layer_names:
        all_mse, total_nan, total_trials = [], 0, 0
        for quad_name, layer_results in quadrant_results.items():
            all_mse.extend(layer_results[layer_name]['mse_values'])
            total_nan += layer_results[layer_name]['nan_count']
            total_trials += NUM_TRIALS
        if len(all_mse) > 0:
            overall_mean.append(np.mean(all_mse))
            overall_std.append(np.std(all_mse))
        else:
            overall_mean.append(0)
            overall_std.append(0)
        overall_nan_rate.append((total_nan / total_trials) * 100)
    fig, ax = plt.subplots(figsize=(16, 8))
    x_pos = np.arange(len(layer_names))
    colors = []
    for i, layer_name in enumerate(layer_names):
        nan_rate = overall_nan_rate[i]
        if nan_rate > 50: colors.append('darkred')
        elif nan_rate > 20: colors.append('orange')
        else:
            if '[Real]' in layer_name: colors.append('steelblue')
            elif '[Imag]' in layer_name: colors.append('royalblue')
            else: colors.append('lightsteelblue')
    ax.bar(x_pos, overall_mean, yerr=overall_std, capsize=3, alpha=0.7, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Mean Squared Error (All Bits)', fontsize=12)
    ax.set_title(f'Overall {fault_target.capitalize()} Fault Sensitivity', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    simplified_labels = []
    for name in layer_names:
        if 'fno_blocks.convs' in name: simplified_labels.append(f"Spectral{name.split('.')[2]}{'R' if '[Real]' in name else 'I' if '[Imag]' in name else ''}")
        elif 'fno_blocks.fno_skips' in name: simplified_labels.append(f"Skip{name.split('.')[2]}")
        elif 'channel_mlp' in name: simplified_labels.append(f"MLP{name.split('.')[2]}.{name.split('.')[4]}")
        elif 'lifting' in name: simplified_labels.append(f"Lift.{name.split('.')[-1]}")
        elif 'projection' in name: simplified_labels.append(f"Proj.{name.split('.')[-1]}")
        else: simplified_labels.append(name[-15:])
    ax.set_xticklabels(simplified_labels, rotation=90, fontsize=8)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    if any(m > 0 for m in overall_mean): ax.set_yscale('log')
    from matplotlib.patches import Patch
    if fault_target == 'weight':
        legend_elements = [Patch(facecolor='lightsteelblue', label='Real (NaN<20%)'), Patch(facecolor='steelblue', label='Complex Real (NaN<20%)'), Patch(facecolor='royalblue', label='Complex Imag (NaN<20%)'), Patch(facecolor='orange', label='NaN 20-50%'), Patch(facecolor='darkred', label='NaN > 50%')]
    else:
        legend_elements = [Patch(facecolor='lightsteelblue', label='NaN < 20%'), Patch(facecolor='orange', label='NaN 20-50%'), Patch(facecolor='darkred', label='NaN > 50%')]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    plt.tight_layout()
    filename = os.path.join(output_dir, f'overall_mse_{fault_target}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Overall MSE plot saved to: {filename}")
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
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Model path for the new sloshing model
    model_path = '../../../../Checkpoints/Sloshing/sloshing_fno_state_dict.pt'

    print(f"\nConfiguration:")
    print(f"  Model Path: {model_path}")
    print(f"  Trials per layer per quadrant: {NUM_TRIALS}")
    print(f"  Test samples: {TEST_SAMPLES}")

    # WEIGHT FAULT INJECTION
    print("\n" + "="*60 + "\nSTARTING WEIGHT FAULT INJECTION BY QUADRANT\n" + "="*60)
    weight_results, injectable_layers = analyze_weight_sensitivity_by_quadrant(model_path, test_loader)
    df_weight_summary = create_summary_dataframe(weight_results, injectable_layers, is_weight=True)
    df_weight_summary.to_csv(os.path.join(OUTPUT_DIR, 'weight_quadrant_summary.csv'), index=False)
    print("\nWeight results saved to weight_quadrant_summary.csv")
    plot_quadrant_comparison(weight_results, injectable_layers, 'weight', OUTPUT_DIR)
    plot_overall_mse_comparison(weight_results, injectable_layers, 'weight', OUTPUT_DIR)

    # ACTIVATION FAULT INJECTION
    print("\n" + "="*60 + "\nSTARTING ACTIVATION FAULT INJECTION BY QUADRANT\n" + "="*60)
    activation_results, unique_modules = analyze_activation_sensitivity_by_quadrant(model_path, test_loader)
    df_activation_summary = create_summary_dataframe(activation_results, unique_modules, is_weight=False)
    df_activation_summary.to_csv(os.path.join(OUTPUT_DIR, 'activation_quadrant_summary.csv'), index=False)
    print("\nActivation results saved to activation_quadrant_summary.csv")
    plot_quadrant_comparison(activation_results, unique_modules, 'activation', OUTPUT_DIR)
    plot_overall_mse_comparison(activation_results, unique_modules, 'activation', OUTPUT_DIR)

    # SUMMARY ANALYSIS
    print("\n" + "="*60 + "\nQUADRANT ANALYSIS SUMMARY\n" + "="*60)
    print("\n--- Weight Faults ---")
    print(df_weight_summary.to_string(index=False))
    print("\n\n--- Activation Faults ---")
    print(df_activation_summary.to_string(index=False))

    spectral_layers = [(name, wtype) for name, _, wtype in injectable_layers if 'convs' in name]
    if spectral_layers:
        print("\n" + "="*60 + "\nSPECTRAL CONVOLUTION: REAL vs IMAGINARY SENSITIVITY\n" + "="*60)
        for quad_name in BIT_QUADRANTS.keys():
            print(f"\n{quad_name}:")
            for layer_name, wtype in spectral_layers:
                results = weight_results[quad_name][layer_name]
                if len(results['mse_values']) > 0:
                    print(f"  {layer_name}: {np.mean(results['mse_values']):.6e}")

    plt.show()


