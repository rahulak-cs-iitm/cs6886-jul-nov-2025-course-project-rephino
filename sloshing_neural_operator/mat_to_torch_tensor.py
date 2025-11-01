import scipy.io as sio
import torch
import glob
import os

# --- Configuration ---
DATA_PATH = './FNO_Dataset/'
NEW_PATH = './FNO_Dataset_PT/' # New folder for PyTorch files
DATA_KEY = 'velocity_field_5D' 
# ---------------------

file_paths = glob.glob(f"{DATA_PATH}/FNO_dataset_run_*.mat")
os.makedirs(NEW_PATH, exist_ok=True) # Create the new directory

print(f"Found {len(file_paths)} files to convert.")

for path in file_paths:
    try:
        filename = os.path.basename(path)
        new_filename = os.path.splitext(filename)[0] + '.pt' # New extension
        new_full_path = os.path.join(NEW_PATH, new_filename)

        # 1. Load the .mat file
        mat_data = sio.loadmat(path)
        tensor_data = torch.tensor(mat_data[DATA_KEY]).float() # Convert to tensor
        
        # 2. Save as a .pt file
        torch.save(tensor_data, new_full_path)
        
        print(f"Converted {filename} -> {new_filename}")

    except Exception as e:
        print(f"Error converting {path}: {e}")

print("Conversion complete!")