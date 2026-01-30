
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
        
        # Get trials array
        if 'trials' not in mat_data:
            print(f"Skipping {filename}: 'trials' key not found.")
            continue
            
        trials = mat_data['trials'].reshape(-1) # Flatten to 1D array of trials
        
        print(f"Processing {len(trials)} trials in {filename}...")
        
        all_eeg_segments = []
        all_audio_segments = []
        all_labels = []

        for i, trial in enumerate(trials):
            # trial is a numpy element (void). Access fields by name.
            # Helper to safely get field
            def get_field(struct, field):
                if field in struct.dtype.names:
                    return struct[field]
                return None
            
            # --- 1. Extract EEG ---
            # Try RawData -> EegData
            raw_data = get_field(trial, 'RawData')
            eeg = None
            if raw_data is not None and raw_data.size > 0:
                inner = raw_data[0,0] if raw_data.ndim > 1 else raw_data.flat[0]
                if 'EegData' in inner.dtype.names:
                    eeg = inner['EegData'] # Shape (Channels, Time) or (Time, Channels)

            if eeg is None:
                print(f"Skipping trial {i}: EEG not found in RawData.")
                continue
            
            # Unwrap nested arrays (often happens with mat files)
            while eeg.size == 1 and isinstance(eeg.flat[0], np.ndarray):
                eeg = eeg.flat[0]
            
            print(f"  Trial {i}: EEG Shape raw: {eeg.shape}")
            
            # Check for minimal length
            if max(eeg.shape) < 100:
                print(f"Skipping trial {i}: EEG data too short ({max(eeg.shape)} samples).")
                continue
                
            # --- 2. Extract Audio Filenames ---
            stimuli = get_field(trial, 'stimuli')
            audio_files = [] # [PathA, PathB]
            if stimuli is not None and stimuli.size > 0:
                inner_stim = stimuli[0,0] if stimuli.ndim > 1 else stimuli.flat[0]
                
                # Check if we have named fields
                if hasattr(inner_stim, 'dtype') and inner_stim.dtype.names is not None:
                    for name in inner_stim.dtype.names:
                         val = inner_stim[name]
                         if val.size > 0:
                             str_val = str(val[0]) if isinstance(val[0], (str, np.str_)) else ""
                             if len(str_val) == 0 and val.dtype.kind in 'SU': # String or Unicode
                                  str_val = str(val.flat[0])
                             
                             if str_val.endswith('.wav'):
                                 audio_files.append(str_val)
                else:
                    # No named fields, might be an array of strings/objects
                    # Iterate over flat elements if they look like strings
                    if inner_stim.size > 0:
                        flat_stim = inner_stim.flatten()
                        for item in flat_stim:
                            str_val = ""
                            if isinstance(item, (str, np.str_)):
                                str_val = str(item)
                            elif isinstance(item, np.ndarray) and item.size > 0:
                                # Sometimes it's nested array of strings
                                sub_item = item.flat[0]
                                if isinstance(sub_item, (str, np.str_)):
                                    str_val = str(sub_item)
                            
                            if str_val.endswith('.wav'):
                                audio_files.append(str_val)
            
            # Sort to ensure order (usually Stream A then Stream B)
            # Or reliance on specific keys if known. For now, sorting assumes naming convention.
            audio_files = sorted(audio_files) 
            
            if len(audio_files) < 2:
                print(f"Skipping trial {i}: Found {len(audio_files)} audio files, needed 2.")
                continue
                
            # --- 3. Extract Label ---
            # attended_ear: 'R' or 'L' or 0/1
            attended_ear = get_field(trial, 'attended_ear')
            label = 0 # Default
            if attended_ear is not None and attended_ear.size > 0:
                val = attended_ear.flat[0]
                if isinstance(val, (str, np.str_)):
                     # Assume 'L' = 0 (left), 'R' = 1 (right) or dependent on audio order
                     # If audio files are [TrackA, TrackB], we need to know which one was attended.
                     # Usually KUL provides 'attended_track' ?
                     pass
                elif isinstance(val, (int, float, np.number)):
                     label = int(val) 
            
            # Trying 'attended_track' if available (more reliable for A vs B)
            attended_track = get_field(trial, 'attended_track') 
            if attended_track is not None and attended_track.size > 0:
                 try:
                     val_str = str(attended_track.flat[0])
                     # If matches first audio file -> 0, else 1
                     if val_str in audio_files[0]:
                         label = 0
                     elif val_str in audio_files[1]:
                         label = 1
                 except:
                     pass

            # --- PROCESS DATA ---
            
            # 1. Process EEG
            # EEG FS is usually in RawData header or known.
            EEG_FS_RAW = 128.0 # If already downsampled? Or 8192?
            # KUL standard raw is often high, but let's assume raw was passed here. 
            # Ideally try to read fs from struct.
            
            processed_eeg = preprocess_eeg(eeg, EEG_FS_RAW) # -> (64, Time)
            processed_eeg = processed_eeg.astype(np.float32)

            # 2. Process Audio
            wavs = []
            for af in audio_files[:2]:
                 path = os.path.join(args.stimuli_dir, af)
                 if not os.path.exists(path):
                     print(f"Warning: Audio file not found {path}")
                     # Try finding mostly matching file?
                     basename = os.path.basename(af)
                     path = os.path.join(args.stimuli_dir, basename)
                 
                 if os.path.exists(path):
                     w, sr = torchaudio.load(path)
                     # Resample
                     if sr != 16000:
                         w = torchaudio.transforms.Resample(sr, 16000)(w)
                     
                     # Normalize (Ref: "mix them 0 db")
                     w = normalize_audio(w)
                     wavs.append(w)
                 else:
                     wavs.append(torch.zeros(1, 16000)) # Dummy if missing

            if len(wavs) < 2: continue
            
            wavA = wavs[0]
            wavB = wavs[1]
            
            # Truncate to min length
            min_len = min(wavA.shape[1], wavB.shape[1])
            wavA = wavA[:, :min_len]
            wavB = wavB[:, :min_len]
            
            # Align EEG
            # EEG time = audio time
            # audio len / 16000 = seconds
            # eeg len / 128 = seconds
            # Crop to match the shorter duration
            dur_audio = min_len / 16000.0
            dur_eeg = processed_eeg.shape[1] / 128.0
            dur = min(dur_audio, dur_eeg)
            
            eeg_samples = int(dur * 128)
            audio_samples = int(dur * 16000)
            
            processed_eeg = processed_eeg[:, :eeg_samples]
            wavA = wavA[:, :audio_samples]
            wavB = wavB[:, :audio_samples]

            # 3. Segment into 1s windows
            win_eeg = 128
            win_audio = 16000
            n_wins = int(dur) # Full seconds only
            
            for w_idx in range(n_wins):
                eeg_seg = processed_eeg[:, w_idx*win_eeg : (w_idx+1)*win_eeg]
                aud_segA = wavA[:, w_idx*win_audio : (w_idx+1)*win_audio]
                aud_segB = wavB[:, w_idx*win_audio : (w_idx+1)*win_audio]
                
                # Stack Audio (2, 16000)
                aud_stacked = torch.cat([aud_segA, aud_segB], dim=0)
    
                all_eeg_segments.append(eeg_seg)
                all_audio_segments.append(aud_stacked.numpy())
                all_labels.append(label)

        if len(all_eeg_segments) > 0:
            # 4. Save
            eeg_final = np.array(all_eeg_segments)
            audio_final = np.array(all_audio_segments)
            labels_final = np.array(all_labels)
    
            save_path = os.path.join(args.output_dir, f"{filename}.npy")
            # Usually we remove .mat extension
            save_path = os.path.join(args.output_dir, f"{subject_name}.npy")
            
            # Save as dictionary
            np.savez(save_path, eeg=eeg_final, audio=audio_final, ear=labels_final)
            print(f"Saved {save_path}: EEG {eeg_final.shape}, Audio {audio_final.shape}")
        else:
            print(f"No valid segments found for {filename}")

        continue # Finished this file

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess KUL Dataset to .npy format')
    parser = argparse.ArgumentParser()
    parser.add_argument('--mat_dir', type=str, required=True, help='Directory containing EEG .mat files')
    parser.add_argument('--stimuli_dir', type=str, required=True, help='Directory containing Audio stimuli files')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for .npy files')
    
    args = parser.parse_args()
    main(args)
