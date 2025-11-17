import os
import librosa
import numpy as np
from os import listdir
from os.path import isdir, join
import random

# Optimized settings for efficiency
SAMPLE_RATE = 16000  # Increased from 8kHz for better quality
N_MFCC = 13  # Reduced from 16 - no deltas needed
SECONDS = 1
FRAME_LENGTH = 512  # Smaller frame for efficiency
HOP_LENGTH = 160   # 10ms hop
N_FFT = 512

def extract_minimal_mfcc(audio_path, target_length=None):
    """Extract minimal MFCC features for efficiency"""
    try:
        # Load audio
        signal, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=SECONDS)
        
        # Ensure exactly 1 second
        if len(signal) > SAMPLE_RATE:
            signal = signal[:SAMPLE_RATE]
        elif len(signal) < SAMPLE_RATE:
            signal = np.pad(signal, (0, SAMPLE_RATE - len(signal)), mode='constant')
        
        # Pre-emphasis filter (slight improvement)
        signal = np.append(signal[0], signal[1:] - 0.97 * signal[:-1])
        
        # Extract MFCCs - just 13 coefficients, no deltas
        mfccs = librosa.feature.mfcc(
            y=signal,
            sr=sr,
            n_mfcc=N_MFCC,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=26,
            fmin=0,
            fmax=sr//2
        )
        
        # Target length for consistency (pad or truncate)
        if target_length is not None:
            if mfccs.shape[1] > target_length:
                mfccs = mfccs[:, :target_length]
            elif mfccs.shape[1] < target_length:
                mfccs = np.pad(mfccs, ((0, 0), (0, target_length - mfccs.shape[1])), mode='constant')
        
        return mfccs.T  # Return as (time, features)
        
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def create_optimized_dataset(dataset_path, output_file='optimized_wake_word_data.npz'):
    """Create optimized dataset with minimal features"""
    
    # Get all target directories
    all_targets = [name for name in listdir(dataset_path) if isdir(join(dataset_path, name))]
    print(f"Found targets: {all_targets}")
    
    # Collect all files
    filenames = []
    labels = []
    
    for target_idx, target in enumerate(all_targets):
        target_dir = join(dataset_path, target)
        target_files = [f for f in listdir(target_dir) if f.endswith('.wav')]
        
        filenames.extend([join(target_dir, f) for f in target_files])
        labels.extend([target_idx] * len(target_files))
        
        print(f"{target}: {len(target_files)} samples")
    
    print(f"Total samples: {len(filenames)}")
    
    # Shuffle data
    combined = list(zip(filenames, labels))
    random.shuffle(combined)
    filenames, labels = zip(*combined)
    
    # Extract features from a sample to determine consistent length
    sample_mfcc = extract_minimal_mfcc(filenames[0])
    if sample_mfcc is None:
        raise ValueError("Could not extract features from sample file")
    
    target_length = sample_mfcc.shape[0]  # Use first sample's length as target
    print(f"Target MFCC length: {target_length}")
    
    # Extract all features
    features = []
    valid_labels = []
    failed_count = 0
    
    for i, (filename, label) in enumerate(zip(filenames, labels)):
        if i % 100 == 0:
            print(f"Processing {i}/{len(filenames)}")
        
        mfcc = extract_minimal_mfcc(filename, target_length)
        if mfcc is not None:
            features.append(mfcc)
            valid_labels.append(label)
        else:
            failed_count += 1
    
    print(f"Successfully processed: {len(features)}")
    print(f"Failed: {failed_count}")
    
    features = np.array(features)
    valid_labels = np.array(valid_labels)
    
    # Split into train/val/test
    n_samples = len(features)
    n_test = int(0.1 * n_samples)
    n_val = int(0.1 * n_samples)
    n_train = n_samples - n_test - n_val
    
    # Convert to binary classification (assuming positive class is index 1)
    binary_labels = (valid_labels == 1).astype(np.float32)
    
    # Split data
    x_test = features[:n_test]
    y_test = binary_labels[:n_test]
    
    x_val = features[n_test:n_test + n_val]
    y_val = binary_labels[n_test:n_test + n_val]
    
    x_train = features[n_test + n_val:]
    y_train = binary_labels[n_test + n_val:]
    
    print(f"Train: {len(x_train)}, Val: {len(x_val)}, Test: {len(x_test)}")
    print(f"Positive samples - Train: {y_train.sum()}, Val: {y_val.sum()}, Test: {y_test.sum()}")
    
    # Save dataset
    np.savez(output_file,
             x_train=x_train, y_train=y_train,
             x_val=x_val, y_val=y_val,
             x_test=x_test, y_test=y_test,
             sample_rate=SAMPLE_RATE,
             n_mfcc=N_MFCC,
             target_length=target_length)
    
    print(f"Dataset saved to {output_file}")
    return output_file

# Example usage
if __name__ == "__main__":
    dataset_path = "/media/trax/Elements/machine-learning/training-data/Brisingr wakeword/"  # Update this path
    create_optimized_dataset(dataset_path)