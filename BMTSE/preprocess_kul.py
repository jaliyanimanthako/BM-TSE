
import os
import argparse
import numpy as np
import scipy.io
import scipy.signal
import torch
import torchaudio
import glob
from tqdm import tqdm

def load_mat_file(file_path):
    """
    Load .mat file, handling potentially different versions or structures.
    """
    try:
        return scipy.io.loadmat(file_path)
    except NotImplementedError:
        try:
            import h5py
            return h5py.File(file_path, 'r')
        except ImportError:
            print("Error: Could not load .mat file. If it's v7.3, please install h5py.")
            return None

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Apply Butterworth bandpass filter.
    data: (Time, Channels) or (Channels, Time) - Code assumes axis=-1 is time if not specified, 
    but we will transpose to (..., Time) before filtering to be safe.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = scipy.signal.filtfilt(b, a, data, axis=-1)
    return y

def notch_filter(data, fs, freq=50.0, q=30.0):
    """
    Apply Notch filter to remove power-line interference.
    """
    b, a = scipy.signal.iirnotch(freq, q, fs)
    y = scipy.signal.filtfilt(b, a, data, axis=-1)
    return y

def preprocess_eeg(eeg_data, fs_original):
    """
    Apply KUL preprocessing steps:
    1. Re-referencing (Average Reference)
    2. Band-pass filter (0.1 - 45 Hz)
    3. Notch filter (50 Hz)
    4. Downsample to 128 Hz
    
    eeg_data: numpy array of shape (Channels, Time)
    fs_original: Sampling rate of the input EEG
    """
    # Ensure shape is (Channels, Time)
    if eeg_data.shape[0] > eeg_data.shape[1]:
         # If more rows than cols, assume (Time, Channels) -> transpose
         eeg_data = eeg_data.T

    # 1. Re-referencing (Average Reference)
    # Subtract average of all channels from each channel
    eeg_data = eeg_data - np.mean(eeg_data, axis=0, keepdims=True)

    # 2. Band-pass Filter (0.1 - 45 Hz)
    eeg_data = butter_bandpass_filter(eeg_data, 0.1, 45.0, fs_original, order=4)

    # 3. Notch Filter (50 Hz)
    eeg_data = notch_filter(eeg_data, fs_original, freq=50.0)

    # 4. Downsample to 128 Hz
    target_fs = 128
    if fs_original != target_fs:
        num_samples = int(eeg_data.shape[1] * float(target_fs) / fs_original)
        eeg_data = scipy.signal.resample(eeg_data, num_samples, axis=1)

    return eeg_data

def normalize_audio(audio_tensor):
    """
    Normalize audio to have RMS = 1.0 (Unit Energy).
    This ensures that when A and B are mixed, if both are normalized, 
    they are mixed at approx 0dB SNR (assuming similar crest factors).
    """
    rms = torch.sqrt(torch.mean(audio_tensor**2))
    if rms > 0:
        return audio_tensor / rms
    return audio_tensor

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    
    mat_files = sorted(glob.glob(os.path.join(args.mat_dir, '*.mat')))
    print(f"Found {len(mat_files)} .mat files in {args.mat_dir}")

    for mat_path in tqdm(mat_files, desc="Processing Subjects"):
        filename = os.path.basename(mat_path)
        subject_name = os.path.splitext(filename)[0] # e.g. "S01"
        
        # Load MAT file
        mat_data = load_mat_file(mat_path)
        if mat_data is None: continue

        # --- EXTRACT VARIABLES FROM MAT ---
        # NOTE: You may need to adjust these key names based on your actual MAT file structure.
        # Common KUL keys: 'trials', 'data', 'subject'
        # Trying to guess standard structure or look for likely candidates
        
        # Placeholder for extraction logic. 
        # We assume a structure where we can iterate over trials.
        # For KUL dataset (typically):
        # mat_data['trials'] is a struct array.
        
        # Heuristic to find the main data structure
        possible_keys = [k for k in mat_data.keys() if not k.startswith('__')]
        if len(possible_keys) == 0:
            print(f"Skipping {filename}: No data keys found.")
            continue
            
        print(f"Keys in {filename}: {possible_keys}")
        
        # User Logic: Iterate trials
        # This part heavily depends on the specific MAT structure.
        # We will assume 'trials' exists or use the first available key.
        data_struct = mat_data[possible_keys[0]] 
        
        # Assuming structure: data_struct is a dictionary or structured array
        # This loop is generic; USER MUST VERIFY
        
        all_eeg_segments = []
        all_audio_segments = [] # Should contain [AudioA, AudioB]
        all_labels = []   # 0 or 1
        
        # Hardcoded constraints
        EEG_FS_RAW = 8192 # Example assumption, user must verify
        # If sampling rate is in the file, load it.
        if 'fs' in mat_data:
            EEG_FS_RAW = float(mat_data['fs'][0][0])
            
        # Example processing loop (Assuming 'trials' list structure)
        # You might need to change this loop if your data is just one big array.
        
        # --- DEMO / PLACEHOLDER LOOP ---
        # Modify below to match your structure.
        # Example: assuming 'trials' has 'eeg', 'stimuli_name', 'attention_label'
        
        # If we cannot parse, we warn.
        print(f"WARNING: Script needs to be tailored to internal MAT structure. Inspecting {filename}...")
        
        # Simulated extraction for the script to be valid python
        # Replace this block with actual extraction logic
        continue_processing = False # Set to true if you implement extraction
        
        if not continue_processing:
            print(f"Please inspect {filename} and update the extraction logic in lines 100-150.")
            print("Skipping processing for now to avoid errors.")
            continue

        # --- END PLACEHOLDER ---

        # Once you have:
        # raw_eeg (Time, Channels)
        # audio_file_path_A
        # audio_file_path_B
        # label (0 or 1)
        
        # 1. Process EEG
        # processed_eeg = preprocess_eeg(raw_eeg, EEG_FS_RAW) # -> (64, Time_128Hz)

        # 2. Load and Process Audio
        # wavA, srA = torchaudio.load(os.path.join(args.stimuli_dir, audio_file_path_A))
        # wavB, srB = torchaudio.load(os.path.join(args.stimuli_dir, audio_file_path_B))
        
        # Resample to 16k
        # resampler = torchaudio.transforms.Resample(srA, 16000)
        # wavA = resampler(wavA)
        # wavB = resampler(wavB)
        
        # Normalize
        # wavA = normalize_audio(wavA)
        # wavB = normalize_audio(wavB)
        
        # 3. Segment into 1s windows
        # win_eeg = 128
        # win_audio = 16000
        # n_wins = processed_eeg.shape[1] // win_eeg
        
        # for i in range(n_wins):
        #     eeg_seg = processed_eeg[:, i*win_eeg : (i+1)*win_eeg]
        #     aud_segA = wavA[:, i*win_audio : (i+1)*win_audio]
        #     aud_segB = wavB[:, i*win_audio : (i+1)*win_audio]
        #     
        #     # Stack Audio (2, 16000)
        #     aud_stacked = torch.cat([aud_segA, aud_segB], dim=0)
        #
        #     all_eeg_segments.append(eeg_seg)
        #     all_audio_segments.append(aud_stacked.numpy())
        #     all_labels.append(label)

        # 4. Save
        # eeg_final = np.array(all_eeg_segments)
        # audio_final = np.array(all_audio_segments)
        # labels_final = np.array(all_labels)

        # save_path = os.path.join(args.output_dir, f"{subject_name}.npy")
        # np.savez(save_path, eeg=eeg_final, audio=audio_final, ear=labels_final)
        # print(f"Saved {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess KUL Dataset to .npy format')
    parser = argparse.ArgumentParser()
    parser.add_argument('--mat_dir', type=str, required=True, help='Directory containing EEG .mat files')
    parser.add_argument('--stimuli_dir', type=str, required=True, help='Directory containing Audio stimuli files')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for .npy files')
    
    args = parser.parse_args()
    main(args)
