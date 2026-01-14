#!/usr/bin/env python3
"""
Debug script to show expected MFCC values for comparison with Android.

Run this while saying the wake word into your Linux microphone,
then compare the MFCC statistics with Android's debug output.
"""

import numpy as np
import librosa
import torch
import pyaudio
import time
from datetime import datetime, timezone
import os

# Load model
print("Loading model...")
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir , 'tiny_wake_word_optimized.pt')
model = torch.jit.load(model_path, map_location='cpu')
model.eval()

# Audio settings (same as Android)
SAMPLE_RATE = 16000
BUFFER_SIZE = SAMPLE_RATE  # 1 second
N_MFCC = 13
N_FFT = 512
HOP_LENGTH = 160

def extract_mfcc(audio_signal):
    """Extract MFCC exactly as Linux inference does"""
    # Ensure exactly 1 second
    if len(audio_signal) > SAMPLE_RATE:
        audio_signal = audio_signal[:SAMPLE_RATE]
    elif len(audio_signal) < SAMPLE_RATE:
        audio_signal = np.pad(audio_signal, (0, SAMPLE_RATE - len(audio_signal)))

    # Pre-emphasis
    audio_signal = np.append(audio_signal[0], audio_signal[1:] - 0.97 * audio_signal[:-1])

    # MFCC extraction
    mfccs = librosa.feature.mfcc(
        y=audio_signal,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=26,
        fmin=80,
        fmax=8000
    )

    return mfccs.T  # Shape: [frames, 13]

def run_inference(mfcc_features):
    """Run model inference"""
    # Pad/truncate to 101 frames
    if len(mfcc_features) < 101:
        mfcc_features = np.pad(mfcc_features, ((0, 101 - len(mfcc_features)), (0, 0)))
    else:
        mfcc_features = mfcc_features[:101]

    x = torch.FloatTensor(mfcc_features).unsqueeze(0)
    with torch.no_grad():
        output = model(x)
        probs = torch.softmax(output, dim=1)  # Same as Linux code

    return probs[0][1].item()  # Wake word probability

def main():
    print("="*60)
    print("MFCC DEBUG - Compare these values with Android logcat")
    print("="*60)

    # Initialize audio
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=1024
    )

    print("\nListening... Speak your wake word and compare stats with Android")
    print("Press Ctrl+C to stop\n")

    audio_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)

    try:
        while True:
            # Read audio
            data = stream.read(1024, exception_on_overflow=False)
            new_samples = np.frombuffer(data, dtype=np.float32)

            # Shift buffer
            audio_buffer[:-len(new_samples)] = audio_buffer[len(new_samples):]
            audio_buffer[-len(new_samples):] = new_samples

            # Calculate audio stats
            audio_min = audio_buffer.min()
            audio_max = audio_buffer.max()
            audio_rms = np.sqrt(np.mean(audio_buffer ** 2))

            # Skip if too quiet
            if audio_rms < 0.001:
                continue

            # Extract MFCC
            mfcc = extract_mfcc(audio_buffer.copy())

            # MFCC stats
            mfcc_min = mfcc.min()
            mfcc_max = mfcc.max()
            mfcc_mean = mfcc.mean()
            first_frame = mfcc[0][:5]

            # Run inference
            prob = run_inference(mfcc)

            # Print stats (format matches Android debug output)
            now_utc = datetime.now(timezone.utc)
            timeFormated = now_utc.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            print("=== MFCC Debug (Python/Linux) ===")
            print(f"Timestamp: {timeFormated}")
            print(f"Audio: min={audio_min:.4f}, max={audio_max:.4f}, rms={audio_rms:.4f}")
            print(f"MFCC: frames={len(mfcc)}, min={mfcc_min:.2f}, max={mfcc_max:.2f}, mean={mfcc_mean:.2f}")
            print(f"MFCC first frame (c0-c4): {[f'{x:.2f}' for x in first_frame]}")
            print(f"Wake word probability: {prob:.4f}")
            print()

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

if __name__ == "__main__":
    main()
