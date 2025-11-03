# TMR_Analysis_Sloshing.py
#   Analyze Triple Modular Redundancy (TMR) protection for all non-spectral layers
#   1. Load MSE values from fault injection results
#   2. Calculate expected MSE reduction with TMR on all layers except SpectralConv
#   3. Compute area/memory overhead for TMR implementation

import pandas as pd
import numpy as np
import torch
from neuralop.models import FNO
import matplotlib.pyplot as plt
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#==================
# Configuration
#------------------

RESULTS_DIR = '../../LayerWiseFaultInjection/Isolated_SEU/plots_sloshing'  
MODEL_PATH = '../../../../Checkpoints/Sloshing/sloshing_fno_state_dict.pt'    
OUTPUT_DIR = 'results_sloshing'                                            

# TMR can correct single-bit errors with 100% success
TMR_SINGLE_BIT_CORRECTION = 1.0  # 100% correction for single-bit errors

#==================
# Load Results and Model
#------------------
def load_results():
    """Load fault injection results (sloshing)."""
    weight_results = pd.read_csv(os.path.join(RESULTS_DIR, 'weight_quadrant_summary.csv'))
    activation_results = pd.read_csv(os.path.join(RESULTS_DIR, 'activation_quadrant_summary.csv'))
    return weight_results, activation_results

def load_model_and_count_params():
    """Load sloshing FNO and analyze parameter distribution."""
    
    model = FNO(
        n_modes=(16, 16, 16),
        in_channels=3,
        out_channels=3,
        hidden_channels=32,
        n_layers=4
    ).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
    
    param_breakdown = {
        'lifting': 0,
        'projection': 0,
        'spectral_conv': 0,
        'skip': 0,
        'channel_mlp': 0,
        'other': 0
    }
    
    for name, param in model.named_parameters():
        # Complex params: count real + imag
        num_params = param.numel() * 2 if param.is_complex() else param.numel()
        
        if 'lifting' in name:
            param_breakdown['lifting'] += num_params
        elif 'projection' in name:
            param_breakdown['projection'] += num_params
        elif 'fno_blocks.convs' in name:
            param_breakdown['spectral_conv'] += num_params
        elif 'fno_blocks.fno_skips' in name or 'skip' in name:
            param_breakdown['skip'] += num_params
        elif 'channel_mlp' in name:
            param_breakdown['channel_mlp'] += num_params
        else:
            param_breakdown['other'] += num_params
    
    total_params = sum(param_breakdown.values())
    return model, param_breakdown, total_params

#==================
# MSE Analysis Functions
#------------------
def extract_mse_by_layer(weight_results):
    """Extract average MSE across all quadrants for each layer."""
    mse_by_layer = {}
    # Column headers in your CSV are of the form: 'Q1 (bits 0-7)_MSE'
    for _, row in weight_results.iterrows():
        layer_name = row['Layer']
        mse_values = []
        for q in range(4):
            lo = q*8
            hi = q*8 + 7
            col = f'Q{q+1} (bits {lo}-{hi})_MSE'
            if col in row and not pd.isna(row[col]):
                mse_values.append(row[col])
        if mse_values:
            mse_by_layer[layer_name] = {
                'avg_mse': float(np.mean(mse_values)),
                'max_mse': float(np.max(mse_values)),
                'std_mse': float(np.std(mse_values))
            }
    return mse_by_layer

def categorize_layers(mse_by_layer):
    """Categorize layers by type."""
    categorized = {
        'lifting': [],
        'projection': [],
        'spectral_conv': [],
        'skip': [],
        'channel_mlp': []
    }
    for layer_name, mse_data in mse_by_layer.items():
        if 'lifting' in layer_name:
            categorized['lifting'].append((layer_name, mse_data))
        elif 'projection' in layer_name:
            categorized['projection'].append((layer_name, mse_data))
        elif 'convs' in layer_name:
            categorized['spectral_conv'].append((layer_name, mse_data))
        elif 'fno_skips' in layer_name or 'skip' in layer_name:
            categorized['skip'].append((layer_name, mse_data))
        elif 'channel_mlp' in layer_name:
            categorized['channel_mlp'].append((layer_name, mse_data))
    return categorized

#==================
# TMR Protection Analysis
#------------------
def calculate_tmr_mse_reduction(categorized_layers):
    """
    Calculate expected MSE reduction with TMR protection.
    Protect: lifting, projection, skip, channel_mlp, and 'other'
    Leave unprotected: spectral_conv
    """
    results = {}
    protected_types = ['lifting', 'projection', 'skip', 'channel_mlp']
    for layer_type in protected_types:
        if layer_type not in categorized_layers or not categorized_layers[layer_type]:
            continue
        original_mse = [m['avg_mse'] for _, m in categorized_layers[layer_type]]
        results[layer_type] = {
            'original_avg_mse': float(np.mean(original_mse)),
            'original_max_mse': float(np.max(original_mse)),
            'tmr_protected_mse': 0.0,  # single-bit TMR
            'mse_reduction_factor': float(np.inf) if np.mean(original_mse) > 0 else 1.0,
            'layers_protected': len(original_mse)
        }
    if 'spectral_conv' in categorized_layers and categorized_layers['spectral_conv']:
        original_mse = [m['avg_mse'] for _, m in categorized_layers['spectral_conv']]
        results['spectral_conv'] = {
            'original_avg_mse': float(np.mean(original_mse)),
            'original_max_mse': float(np.max(original_mse)),
            'tmr_protected_mse': float(np.mean(original_mse)),  # left as-is
            'mse_reduction_factor': 1.0,
            'layers_protected': 0
        }
    return results

def calculate_tmr_overhead(param_breakdown):
    """
    Calculate memory overhead for TMR implementation on non-spectral layers.
    """
    lifting_params = param_breakdown['lifting']
    projection_params = param_breakdown['projection']
    skip_params = param_breakdown['skip']
    channel_mlp_params = param_breakdown['channel_mlp']
    other_params = param_breakdown['other']
    spectral_params = param_breakdown['spectral_conv']
    protected_params = lifting_params + projection_params + skip_params + channel_mlp_params + other_params
    total_params = sum(param_breakdown.values())
    tmr_memory_overhead = protected_params * 3  # triple copies
    voter_logic_overhead = protected_params * 0.15  # ignored in final tally
    total_tmr_overhead = tmr_memory_overhead  # ignore voter area in memory calc
    return {
        'protected_params': protected_params,
        'unprotected_params': spectral_params,
        'total_params': total_params,
        'tmr_copies': tmr_memory_overhead,
        'voter_logic_equiv_params': voter_logic_overhead,
        'total_overhead_params': total_tmr_overhead,
        'memory_multiplier': 3,
        'percentage_of_model': (total_tmr_overhead / total_params) * 100 if total_params else 0.0,
        'absolute_overhead': total_tmr_overhead,
        'protected_percentage': (protected_params / total_params) * 100 if total_params else 0.0,
        'lifting': lifting_params,
        'projection': projection_params,
        'skip': skip_params,
        'channel_mlp': channel_mlp_params,
        'spectral_conv': spectral_params,
        'full_model_tmr': total_params * 3,
        'selective_vs_full': (total_tmr_overhead / (total_params * 3)) * 100 if total_params else 0.0
    }

#==================
# Visualization
#------------------
def plot_mse_comparison(categorized_layers, tmr_results, output_dir):
    """Plot MSE comparison: original vs TMR-protected."""
    os.makedirs(output_dir, exist_ok=True)
    layer_types = ['lifting', 'projection', 'skip', 'channel_mlp', 'spectral_conv']
    original_mse_vals, tmr_mse_vals, labels, colors = [], [], [], []
    for layer_type in layer_types:
        if layer_type not in categorized_layers or not categorized_layers[layer_type]:
            continue
        for layer_name, mse_data in categorized_layers[layer_type]:
            original_mse_vals.append(mse_data['avg_mse'])
            if layer_type != 'spectral_conv':
                tmr_mse_vals.append(0.0); colors.append('green')
            else:
                tmr_mse_vals.append(mse_data['avg_mse']); colors.append('steelblue')
            simplified = layer_name.split('.')[-1][:20]
            if '[Real]' in layer_name: simplified += '-R'
            elif '[Imag]' in layer_name: simplified += '-I'
            labels.append(simplified)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    x = np.arange(len(labels)); width = 0.35
    ax1.bar(x, original_mse_vals, width, label='Original (No Protection)', color='steelblue', alpha=0.8, edgecolor='black')
    ax1.set_xlabel('Layer', fontsize=11); ax1.set_ylabel('Mean Squared Error', fontsize=11)
    ax1.set_title('Original Fault Sensitivity (All Layers)', fontsize=13, fontweight='bold')
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=90, fontsize=7)
    if any(v > 0 for v in original_mse_vals): ax1.set_yscale('log')
    ax1.grid(axis='y', alpha=0.3, linestyle='--'); ax1.legend()
    ax2.bar(x, tmr_mse_vals, width, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_xlabel('Layer', fontsize=11); ax2.set_ylabel('Mean Squared Error', fontsize=11)
    ax2.set_title('With TMR Protection (All Except SpectralConv)', fontsize=13, fontweight='bold')
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=90, fontsize=7)
    if any(v > 0 for v in tmr_mse_vals): ax2.set_yscale('log')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', label='TMR Protected (MSE ≈ 0)'), Patch(facecolor='steelblue', label='Unprotected')]
    ax2.legend(handles=legend_elements, fontsize=9)
    plt.tight_layout()
    outpath = os.path.join(output_dir, 'tmr_mse_comparison.png')
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"\nMSE comparison plot saved to {outpath}")
    return fig

#==================
# Reporting
#------------------
def print_tmr_analysis_report(mse_by_layer, categorized_layers, tmr_results, overhead_results):
    print("\n" + "="*80)
    print("TRIPLE MODULAR REDUNDANCY (TMR) PROTECTION ANALYSIS (Sloshing FNO)")
    print("Strategy: Protect ALL layers EXCEPT SpectralConv")
    print("="*80)
    print("\n### 1. FAULT SENSITIVITY (MSE) FOR ALL LAYERS ###\n")
    print(f"{'Layer Type':<20} {'Layer Name':<60} {'Avg MSE':<15} {'Max MSE':<15}")
    print("-" * 115)
    for layer_type in ['lifting', 'projection', 'skip', 'channel_mlp', 'spectral_conv']:
        if layer_type not in categorized_layers: continue
        for layer_name, mse_data in categorized_layers[layer_type]:
            print(f"{layer_type:<20} {layer_name:<60} {mse_data['avg_mse']:<15.6e} {mse_data['max_mse']:<15.6e}")
    print("\n" + "="*80)
    print("### 2. TMR MEMORY OVERHEAD ANALYSIS ###")
    print("="*80)
    print(f"\n--- Parameter Breakdown ---")
    print(f"Total Model Parameters:               {overhead_results['total_params']:>15,}")
    print(f"  Lifting:                            {overhead_results['lifting']:>15,}")
    print(f"  Projection:                         {overhead_results['projection']:>15,}")
    print(f"  Skip Connections:                   {overhead_results['skip']:>15,}")
    print(f"  Channel MLP:                        {overhead_results['channel_mlp']:>15,}")
    print(f"  SpectralConv:                       {overhead_results['spectral_conv']:>15,}")
    print(f"\n--- TMR Protection ---")
    print(f"Parameters Protected (non-spectral):  {overhead_results['protected_params']:>15,}")
    print(f"Parameters Left Unprotected:          {overhead_results['unprotected_params']:>15,} (SpectralConv)")
    print(f"\n--- TMR Implementation Overhead ---")
    print(f"TMR Copies (3x replication):          {overhead_results['tmr_copies']:>15,}")
    print(f"\n--- Memory Savings: Full vs Selective ---")
    print(f"Full Model TMR (3x all params):       {overhead_results['full_model_tmr']:>15,.0f}")
    print(f"Selective TMR (non-spectral only):    {overhead_results['total_overhead_params']:>15,.0f}")
    print(f"Selective vs Full:                    {overhead_results['selective_vs_full']:>15.2f}%")
    print(f"\nMemory Savings:                       {100 - overhead_results['selective_vs_full']:>15.2f}%")

#==================
# Main Execution
#------------------
if __name__ == "__main__":
    weight_results, activation_results = load_results()
    model, param_breakdown, total_params = load_model_and_count_params()
    mse_by_layer = extract_mse_by_layer(weight_results)
    categorized_layers = categorize_layers(mse_by_layer)
    tmr_results = calculate_tmr_mse_reduction(categorized_layers)
    overhead_results = calculate_tmr_overhead(param_breakdown)
    plot_mse_comparison(categorized_layers, tmr_results, OUTPUT_DIR)
    print_tmr_analysis_report(mse_by_layer, categorized_layers, tmr_results, overhead_results)
    print("\n" + "="*80)
    print("="*80)

