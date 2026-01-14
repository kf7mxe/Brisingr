#!/usr/bin/env python3
"""
Detailed MFCC comparison script for debugging Android vs Linux differences.
Uses the same audio files to compare MFCC extraction and model output.
"""

import librosa
import numpy as np
import torch
import json

# MFCC parameters (must match Android)
SAMPLE_RATE = 16000
N_MFCC = 13
N_FILT = 26  # n_mels
N_FFT = 512
HOP_LENGTH = 160
FMIN = 80
FMAX = 8000

def extract_mfcc_librosa(audio, sr=SAMPLE_RATE):
    """Extract MFCC using librosa (matching the Linux inference script exactly)"""
    # Pad or trim to exactly 1 second
    target_length = sr
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
    elif len(audio) > target_length:
        audio = audio[:target_length]

    # IMPORTANT: Pre-emphasis filter (must match Linux inference)
    audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_FILT,
        fmin=FMIN,
        fmax=FMAX
    )

    return mfcc.T  # Return as (frames, coefficients)

def load_model(model_path='tiny_wake_word_optimized.pt'):
    """Load the TorchScript model"""
    model = torch.jit.load(model_path, map_location='cpu')
    model.eval()
    return model

def run_inference(model, mfcc_features):
    """Run inference on MFCC features"""
    # Pad/truncate to 101 frames
    expected_frames = 101
    if mfcc_features.shape[0] < expected_frames:
        padding = np.zeros((expected_frames - mfcc_features.shape[0], mfcc_features.shape[1]))
        mfcc_features = np.vstack([mfcc_features, padding])
    elif mfcc_features.shape[0] > expected_frames:
        mfcc_features = mfcc_features[:expected_frames]

    # Convert to tensor: (1, frames, mfcc) - NOT (1, 1, frames, mfcc)
    tensor = torch.FloatTensor(mfcc_features).unsqueeze(0)
    print(f"  Tensor shape: {tensor.shape}")

    with torch.no_grad():
        output = model(tensor)
        # Apply softmax to get probabilities
        probs = torch.softmax(output, dim=1)
        wake_word_prob = probs[0, 1].item()

    return output.numpy()[0], probs.numpy()[0], wake_word_prob

def analyze_file(filename, model):
    """Analyze a single audio file"""
    print(f"\n{'='*60}")
    print(f"Analyzing: {filename}")
    print('='*60)

    # Load audio
    audio, sr = librosa.load(filename, sr=SAMPLE_RATE)
    print(f"\nAudio stats:")
    print(f"  Length: {len(audio)} samples ({len(audio)/sr:.2f}s)")
    print(f"  Min: {audio.min():.6f}")
    print(f"  Max: {audio.max():.6f}")
    print(f"  RMS: {np.sqrt(np.mean(audio**2)):.6f}")

    # Extract MFCC
    mfcc = extract_mfcc_librosa(audio)
    print(f"\nMFCC stats:")
    print(f"  Shape: {mfcc.shape} (frames x coefficients)")
    print(f"  Min: {mfcc.min():.2f}")
    print(f"  Max: {mfcc.max():.2f}")
    print(f"  Mean: {mfcc.mean():.2f}")
    print(f"  Std: {mfcc.std():.2f}")

    # Print first few frames for Android comparison
    print(f"\nMFCC first 3 frames (for Android comparison):")
    for i in range(min(3, len(mfcc))):
        coeffs = ", ".join([f"{c:.2f}" for c in mfcc[i]])
        print(f"  Frame {i}: [{coeffs}]")

    # Print middle frame
    mid_idx = len(mfcc) // 2
    coeffs = ", ".join([f"{c:.2f}" for c in mfcc[mid_idx]])
    print(f"  Frame {mid_idx} (middle): [{coeffs}]")

    # Run inference
    raw_output, probs, wake_prob = run_inference(model, mfcc)
    print(f"\nModel output:")
    print(f"  Raw logits: [{raw_output[0]:.4f}, {raw_output[1]:.4f}]")
    print(f"  Probabilities: [not_wake={probs[0]:.4f}, wake={probs[1]:.4f}]")
    print(f"  Wake word probability: {wake_prob:.4f} ({wake_prob*100:.1f}%)")

    # Return data for JSON export
    return {
        'filename': filename,
        'audio': {
            'length': len(audio),
            'min': float(audio.min()),
            'max': float(audio.max()),
            'rms': float(np.sqrt(np.mean(audio**2)))
        },
        'mfcc': {
            'shape': list(mfcc.shape),
            'min': float(mfcc.min()),
            'max': float(mfcc.max()),
            'mean': float(mfcc.mean()),
            'std': float(mfcc.std()),
            'frames': mfcc.tolist()  # Full MFCC data for comparison
        },
        'model': {
            'raw_logits': raw_output.tolist(),
            'probabilities': probs.tolist(),
            'wake_probability': wake_prob
        }
    }

def main():
    print("Loading model...")
    model = load_model('tiny_wake_word_optimized.pt')
    print("Model loaded successfully")

    results = {}

    # Analyze both files
    for filename in ['negative.wav', 'positive.wav']:
        try:
            results[filename] = analyze_file(filename, model)
        except Exception as e:
            print(f"Error analyzing {filename}: {e}")

    # Save results to JSON for Android comparison
    with open('mfcc_reference.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n{'='*60}")
    print("Reference data saved to mfcc_reference.json")
    print("Use this to compare with Android MFCC output")
    print('='*60)

if __name__ == '__main__':
    main()
