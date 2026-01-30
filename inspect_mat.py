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
        
        # Check if it's an object array (common for structs in scipy.io)
        if trials.dtype.names is None:
            print("Array does not have fields at top level. checking elements...")
            if trials.size > 0:
                first_elem = trials.flat[0]
                print(f"Type of first element: {type(first_elem)}")
                
                if hasattr(first_elem, 'dtype') and first_elem.dtype.names:
                    print(f"Fields found in first trial: {first_elem.dtype.names}")
                    
                    for name in first_elem.dtype.names:
                        val = first_elem[name]
                        shape_info = val.shape if hasattr(val, 'shape') else 'scalar'
                        print(f"  {name}: {shape_info} (Type: {type(val)})")
                else:
                    print("First element does not have named fields.")
                    print(first_elem)
        else:
             names = trials.dtype.names
             print(f"Fields in 'trials': {names}")
             
             if trials.size > 0:
                 first_trial = trials[0,0] if trials.ndim > 1 else trials[0]
                 for name in names:
                     val = first_trial[name]
                     print(f"  {name}: {val.shape if hasattr(val, 'shape') else 'scalar'}")
            
    else:
        print("'trials' key not found.")
        print("Keys found:", list(mat.keys()))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_mat.py <path_to_mat_file>")
    else:
        inspect_mat(sys.argv[1])
