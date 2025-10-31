# TMR_Implementation.py
#   Implement Triple Modular Redundancy (TMR) for FNO model
#   - Triplicate weights for non-SpectralConv layers
#   - Implement bit-level majority voting during inference
#   - Save TMR-protected model as .pth file
#   - Benchmark inference time overhead

import torch
import torch.nn as nn
import struct
import numpy as np
import random
import time
from neuralop.models import FNO
import warnings
warnings.filterwarnings('ignore', message='.*non-tuple sequence for multidimensional indexing.*')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set seeds for reproducibility
SEED = 99 
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#==================
# Bit-Level Majority Voting
#------------------
def float_to_bits(f):
    """Convert float32 to 32-bit binary representation."""
    return format(struct.unpack('>I', struct.pack('>f', f))[0], '032b')

def bits_to_float(bits):
    """Convert 32-bit binary string to float32."""
    return struct.unpack('>f', struct.pack('>I', int(bits, 2)))[0]

def majority_vote_bits(f1, f2, f3):
    """
    Perform bit-level majority voting on three float32 values.
    For each bit position, take the majority value.
    """
    bits1 = float_to_bits(f1)
    bits2 = float_to_bits(f2)
    bits3 = float_to_bits(f3)
    
    result_bits = ''
    for i in range(32):
        # Count votes for bit position i
        vote = int(bits1[i]) + int(bits2[i]) + int(bits3[i])
        # Majority is 1 if vote >= 2, else 0
        result_bits += '1' if vote >= 2 else '0'
    
    return bits_to_float(result_bits)

def majority_vote_tensor(t1, t2, t3):
    """
    Apply bit-level majority voting to three tensors element-wise.
    """
    i1 = t1.view(torch.int32)
    i2 = t2.view(torch.int32)
    i3 = t3.view(torch.int32)
    
    # Bitwise majority vote: (A&B) | (B&C) | (A&C)
    # This computes majority for each bit position simultaneously
    result_int = (i1 & i2) | (i2 & i3) | (i1 & i3)
    
    # Convert back to float32
    return result_int.view(torch.float32)

    # CPU - Slow! # Flatten tensors
    # CPU - Slow! flat1 = t1.flatten().cpu().numpy()
    # CPU - Slow! flat2 = t2.flatten().cpu().numpy()
    # CPU - Slow! flat3 = t3.flatten().cpu().numpy()
    # CPU - Slow! 
    # CPU - Slow! result = np.zeros_like(flat1)
    # CPU - Slow! for i in range(len(flat1)):
    # CPU - Slow!     result[i] = majority_vote_bits(flat1[i], flat2[i], flat3[i])
    # CPU - Slow! 
    # CPU - Slow! return torch.tensor(result, dtype=t1.dtype, device=t1.device).reshape(t1.shape)

#==================
# TMR-Protected FNO Model
#------------------
class TMR_FNO(nn.Module):
    """
    FNO with TMR protection for non-SpectralConv layers.
    Stores 3 copies of vulnerable weights and uses majority voting.
    """
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
        # Storage for TMR-protected weights
        self.tmr_weights = {}
        
        # Identify and triplicate non-SpectralConv weights
        for name, param in base_model.named_parameters():
            if 'convs' not in name and 'weight' in name:
                # This is a non-SpectralConv weight that needs TMR protection
                # Store 3 copies as buffers (non-trainable)
                name_upd = name.replace('.','_')
                self.register_buffer(f'tmr_{name_upd}_copy1', param.data.clone())
                self.register_buffer(f'tmr_{name_upd}_copy2', param.data.clone())
                self.register_buffer(f'tmr_{name_upd}_copy3', param.data.clone())
                
                self.tmr_weights[name] = {
                    'copy1': f'tmr_{name_upd}_copy1',
                    'copy2': f'tmr_{name_upd}_copy2',
                    'copy3': f'tmr_{name_upd}_copy3'
                }
    
    def apply_majority_voting(self):
        """
        Apply majority voting to all TMR-protected weights and update base model.
        Call this before each forward pass to correct any bit flips.
        """
        param_dict = dict(self.base_model.named_parameters())

        for name, copies in self.tmr_weights.items():
            copy1 = getattr(self, copies['copy1'])
            copy2 = getattr(self, copies['copy2'])
            copy3 = getattr(self, copies['copy3'])
            
            # Perform majority voting
            corrected_weight = majority_vote_tensor(copy1, copy2, copy3)
            
            # Update the base model's weight
            if name in param_dict:
                param_dict[name].data.copy_(corrected_weight)
            else:
                print(f"ERROR :: Parameter {name} not found in base_model")
    
    def forward(self, x):
        """
        Forward pass with TMR protection.
        """
        # Apply majority voting to correct any bit flips
        self.apply_majority_voting()
        
        # Run normal forward pass
        return self.base_model(x)
    
    def inject_fault(self, layer_name, copy_num, bit_pos):
        """
        Inject a single-bit fault for testing.
        
        Args:
            layer_name: Name of the layer (e.g., 'lifting.fcs.0.weight')
            copy_num: Which copy to corrupt (1, 2, or 3)
            bit_pos: Bit position to flip (0-31)
        """
        if layer_name not in self.tmr_weights:
            print(f"Warning: {layer_name} not TMR-protected")
            return
        
        copy_name = self.tmr_weights[layer_name][f'copy{copy_num}']
        weight_copy = getattr(self, copy_name)
        
        # Flip a random weight bit
        flat = weight_copy.flatten()
        idx = torch.randint(0, flat.numel(), (1,)).item()
        
        original = flat[idx].item()
        bits = float_to_bits(original)
        
        # Flip the bit
        flipped_bit = '0' if bits[bit_pos] == '1' else '1'
        new_bits = bits[:bit_pos] + flipped_bit + bits[bit_pos+1:]
        flipped = bits_to_float(new_bits)
        
        flat[idx] = flipped
        print(f"Injected fault in {layer_name}, copy {copy_num}, bit {bit_pos}")
        print(f"  Original: {original:.6e}, Flipped: {flipped:.6e}")

    def inject_fault_2_in_same_weight(self, layer_name, copy_num, bit_pos_standard, weight_idx=None):
        """
        Inject a single-bit fault using STANDARD bit numbering.
        
        Args:
            layer_name: Name of the layer (e.g., 'lifting.fcs.0.weight')
            copy_num: Which copy to corrupt (1, 2, or 3)
            bit_pos_standard: Standard bit position (0=LSB, 31=MSB/sign bit)
            weight_idx: Specific weight index to corrupt (if None, random)
        
        Returns:
            weight_idx: The index that was corrupted (for reuse)
        """
        if layer_name not in self.tmr_weights:
            print(f"Warning: {layer_name} not TMR-protected")
            return None
        
        # Convert standard bit position to string index
        bit_pos_string = 31 - bit_pos_standard
        
        copy_name = self.tmr_weights[layer_name][f'copy{copy_num}']
        weight_copy = getattr(self, copy_name)
        
        flat = weight_copy.flatten()
        
        if weight_idx is None:
            weight_idx = torch.randint(0, flat.numel(), (1,)).item()
        
        original = flat[weight_idx].item()
        bits = float_to_bits(original)
        
        # Flip the bit using string index
        flipped_bit = '0' if bits[bit_pos_string] == '1' else '1'
        new_bits = bits[:bit_pos_string] + flipped_bit + bits[bit_pos_string+1:]
        flipped = bits_to_float(new_bits)
        
        flat[weight_idx] = flipped
        print(f"Injected fault in {layer_name}, copy {copy_num}, bit {bit_pos_standard} (standard), idx {weight_idx}")
        print(f"  Original: {original:.10e} (bits: {bits})")
        print(f"  Flipped:  {flipped:.10e} (bits: {new_bits})")
        print(f"  Sign changed: {(original < 0) != (flipped < 0)}")
        print(f"  Magnitude changed: {abs(abs(original) - abs(flipped)) > 1e-10}")
        
        return weight_idx 

#==================
# Model Creation and Saving
#------------------
def create_tmr_model(original_model_path, tmr_model_path):
    """
    Load original FNO model, create TMR-protected version, and save.
    """
    print("Loading original FNO model...")
    base_model = FNO(n_modes=(32, 32), in_channels=1, out_channels=1,
                     hidden_channels=32, projection_channel_ratio=2)
    base_model = base_model.to(device)
    base_model.load_state_dict(torch.load(original_model_path, map_location=device, weights_only=False))
    base_model.eval()
    
    print("Creating TMR-protected model...")
    tmr_model = TMR_FNO(base_model)
    tmr_model.eval()
    
    # Count parameters
    original_params = sum(p.numel() for p in base_model.parameters())
    tmr_params = sum(p.numel() for p in tmr_model.parameters()) + \
                 sum(b.numel() for b in tmr_model.buffers())
    
    print(f"\nOriginal model parameters: {original_params:,}")
    print(f"TMR model parameters + buffers: {tmr_params:,}")
    print(f"Overhead: {tmr_params - original_params:,} ({(tmr_params/original_params - 1)*100:.2f}%)")
    
    # Save TMR model
    print(f"\nSaving TMR-protected model to {tmr_model_path}...")
    torch.save(tmr_model.state_dict(), tmr_model_path)
    
    print("TMR model saved successfully!")
    
    return tmr_model

def load_tmr_model(base_model_path, tmr_model_path):
    """
    Load a saved TMR model.
    """
    print("Loading base FNO architecture...")
    base_model = FNO(n_modes=(32, 32), in_channels=1, out_channels=1,
                     hidden_channels=32, projection_channel_ratio=2)
    base_model = base_model.to(device)
    base_model.load_state_dict(torch.load(base_model_path, map_location=device, weights_only=False))
    
    print("Loading TMR-protected weights...")
    tmr_model = TMR_FNO(base_model)
    tmr_model.load_state_dict(torch.load(tmr_model_path, map_location=device, weights_only=False), 
                              strict=False)
    tmr_model.eval()
    
    print("TMR model loaded successfully!")
    return tmr_model

#==================
# Inference Time Benchmark
#------------------
def benchmark_inference_time(original_model, tmr_model, test_loader, num_samples=100, warmup=10):
    """
    Benchmark inference time for original vs TMR model.
    
    Args:
        original_model: Original FNO model
        tmr_model: TMR-protected FNO model
        test_loader: DataLoader for test data
        num_samples: Number of samples to benchmark
        warmup: Number of warmup iterations
    """
    print("\n" + "="*80)
    print("INFERENCE TIME BENCHMARK")
    print("="*80)
    
    from neuralop.data.datasets.darcy import DarcyDataset
    dataset = DarcyDataset(root_dir="../../../Data/Darcy",
                          n_train=1000, n_tests=[100, 50],
                          batch_size=32, test_batch_sizes=[32, 32],
                          train_resolution=32, test_resolutions=[32, 64],
                          download=False)
    data_processor = dataset.data_processor
    
    # Collect test inputs
    test_inputs = []
    for idx, data in enumerate(test_loader):
        if idx >= num_samples:
            break
        data = data_processor.preprocess(data, batched=True)
        x = data['x'].to(device)
        test_inputs.append(x)
    
    print(f"\nBenchmarking on {len(test_inputs)} samples (with {warmup} warmup iterations)")
    
    # Warmup - Original Model
    print("\nWarming up original model...")
    for i in range(warmup):
        with torch.no_grad():
            _ = original_model(test_inputs[i % len(test_inputs)])
    
    # Benchmark - Original Model
    print("Benchmarking original model...")
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
    
    # Warmup - TMR Model
    print("Warming up TMR model...")
    for i in range(warmup):
        with torch.no_grad():
            _ = tmr_model(test_inputs[i % len(test_inputs)])
    
    # Benchmark - TMR Model
    print("Benchmarking TMR model...")
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.time()
    for x in test_inputs:
        with torch.no_grad():
            _ = tmr_model(x)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    tmr_time = time.time() - start_time
    tmr_time_per_sample = tmr_time / len(test_inputs)
    
    # Calculate overhead
    time_overhead = tmr_time - original_time
    time_overhead_pct = (tmr_time / original_time - 1) * 100
    
    # Print results
    print("\n" + "-"*80)
    print("BENCHMARK RESULTS")
    print("-"*80)
    print(f"Original Model:")
    print(f"  Total time:       {original_time:.4f} seconds")
    print(f"  Time per sample:  {original_time_per_sample*1000:.2f} ms")
    print(f"  Throughput:       {len(test_inputs)/original_time:.2f} samples/sec")
    
    print(f"\nTMR Model:")
    print(f"  Total time:       {tmr_time:.4f} seconds")
    print(f"  Time per sample:  {tmr_time_per_sample*1000:.2f} ms")
    print(f"  Throughput:       {len(test_inputs)/tmr_time:.2f} samples/sec")
    
    print(f"\nOverhead:")
    print(f"  Additional time:  {time_overhead:.4f} seconds")
    print(f"  Time overhead:    {time_overhead_pct:.2f}%")
    print(f"  Slowdown factor:  {tmr_time/original_time:.2f}x")
    print("-"*80)
    
    return {
        'original_time': original_time,
        'tmr_time': tmr_time,
        'overhead_pct': time_overhead_pct,
        'original_per_sample': original_time_per_sample,
        'tmr_per_sample': tmr_time_per_sample
    }

#==================
# Testing TMR Functionality
#------------------
def test_tmr_correction():
    """
    Test that TMR can correct single-bit errors.
    """
    from neuralop.data.datasets.darcy import DarcyDataset
    from torch.utils.data import DataLoader
    
    print("\n" + "="*80)
    print("TESTING TMR CORRECTION CAPABILITY")
    print("="*80)
    
    # Reset seed for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    
    # Load dataset
    dataset = DarcyDataset(root_dir="../../../Data/Darcy",
                           n_train=1000, n_tests=[100, 50],
                           batch_size=32, test_batch_sizes=[32, 32],
                           train_resolution=32, test_resolutions=[32, 64],
                           download=False)
    data_processor = dataset.data_processor
    test_loader = DataLoader(dataset.test_dbs[32], batch_size=1, shuffle=False)
    
    # Get one test sample
    data = next(iter(test_loader))
    data = data_processor.preprocess(data, batched=True)
    x = data['x'].to(device)
    y = data['y'].to(device)
    
    # Create TMR model
    original_path = '../../../Checkpoints/Darcy/darcy_fno_state_dict.pt'
    tmr_path = 'checkpoints/darcy_fno_tmr.pth'
    
    tmr_model = create_tmr_model(original_path, tmr_path)
    
    # Test 0: Original model inference (baseline)
    print("\nTest 0: Original model inference (baseline)")
    original_model = FNO(n_modes=(32, 32), in_channels=1, out_channels=1,
                         hidden_channels=32, projection_channel_ratio=2)
    original_model = original_model.to(device)
    original_model.load_state_dict(torch.load(original_path, map_location=device, weights_only=False))
    original_model.eval()
    
    with torch.no_grad():
        output_original = original_model(x)
        mse_original = torch.mean((output_original - y) ** 2).item()
    print(f"MSE (original): {mse_original:.6e}")
    
    # Test 1: Normal inference
    print("\nTest 1: TMR model inference (no faults)")
    with torch.no_grad():
        output_clean = tmr_model(x)
        mse_clean = torch.mean((output_clean - y) ** 2).item()
    print(f"MSE (TMR - no faults injected): {mse_clean:.6e}")
    print(f"Delta MSE vs original: {abs(mse_clean - mse_original):.6e}")
    
    # Test 2: Single-bit error in one copy (should be corrected)
    print("\nTest 2: Single-bit error in one copy (TMR corrects)")
    tmr_model.inject_fault('lifting.fcs.0.weight', copy_num=1, bit_pos=30)
    with torch.no_grad():
        output_corrected = tmr_model(x)
        mse_corrected = torch.mean((output_corrected - y) ** 2).item()
    
    delta_mse_2 = abs(mse_corrected - mse_clean)
    print(f"MSE (TMR - fault injected in 1 copy - corrected): {mse_corrected:.6e}")
    print(f"Delta MSE vs clean: {delta_mse_2:.6e}")
    
    # Test 3: Two-bit errors in SAME weight (TMR fails)
    print("\nTest 3: Two-bit errors in same weight (TMR fails)")
    weight_idx = tmr_model.inject_fault_2_in_same_weight(
        'lifting.fcs.0.weight', copy_num=1, bit_pos_standard=30)
    tmr_model.inject_fault_2_in_same_weight(
        'lifting.fcs.0.weight', copy_num=2, bit_pos_standard=30, weight_idx=weight_idx)
    
    with torch.no_grad():
        output_failed = tmr_model(x)
        mse_failed = torch.mean((output_failed - y) ** 2).item()
    
    delta_mse_3 = abs(mse_failed - mse_clean)
    print(f"\nMSE (TMR - 2 faults in same copy - cannot correct): {mse_failed:.6e}")
    print(f"Delta MSE vs clean: {delta_mse_3:.6e}")
    print(f"MSE increased by: {((mse_failed/mse_clean - 1) * 100):.2f}%")
    
    # Benchmark inference time
    benchmark_inference_time(original_model, tmr_model, test_loader, num_samples=100, warmup=10)

#==================
# Main Execution
#------------------
if __name__ == "__main__":
    original_model_path = '../../../Checkpoints/Darcy/darcy_fno_state_dict.pt'
    tmr_model_path = 'checkpoints/darcy_fno_tmr.pth'
    
    # Create and save TMR model
    tmr_model = create_tmr_model(original_model_path, tmr_model_path)
    
    # Test TMR correction
    test_tmr_correction()
    
    print("\n" + "="*80)
    print("TMR Implementation Complete")
    print(f"TMR model saved to: {tmr_model_path}")
    print("="*80)

