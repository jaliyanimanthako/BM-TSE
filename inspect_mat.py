import scipy.io
import sys
import os
import numpy as np

def inspect_mat(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"--- Inspecting {os.path.basename(file_path)} ---")
    try:
        mat = scipy.io.loadmat(file_path)
    except NotImplementedError:
        print("Using h5py...")
        import h5py
        mat = h5py.File(file_path, 'r')
    
    if 'trials' in mat:
        trials = mat['trials']
        print(f"Type of 'trials': {type(trials)}")
        print(f"Shape of 'trials': {trials.shape}")
        
        # Access the first element to see fields
        if hasattr(trials, 'dtype'):
             print("\nFields in 'trials' (scipy.io):")
             names = trials.dtype.names
             print(names)
             
             # Print details of first trial
             if trials.size > 0:
                 first_trial = trials[0,0] if trials.ndim > 1 else trials[0]
                 print("\nContent types in first trial:")
                 for name in names:
                     val = first_trial[name]
                     print(f"  {name}: {type(val)} - Shape: {val.shape if hasattr(val, 'shape') else 'scalar'}")
        else:
            # h5py structure
            print("\nKeys in 'trials' (h5py group):")
            print(list(trials.keys()))
            
    else:
        print("'trials' key not found.")
        print("Keys found:", list(mat.keys()))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_mat.py <path_to_mat_file>")
    else:
        inspect_mat(sys.argv[1])
