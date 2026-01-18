#!/usr/bin/env python3
"""
Synthetic Wake Word Data Generator for "Brisingr" (BRIS-ing-gr)

This script generates synthetic positive training samples for wake word detection
using Text-to-Speech (TTS) engines and audio augmentation techniques.

Pronunciation: BRIS-ing-gr
- "BRIS" rhymes with "Swiss" (short 'i' as in "it", sharp 's')
- "ing" like in "sing"
- "gr" with soft 'er' sound (like "her"), similar to "ger" or "grr"
- Hard 'g' (as in "go")

Requirements:
    pip install TTS pyttsx3 librosa soundfile numpy scipy

    For higher quality voices (optional):
    pip install edge-tts  # Microsoft Edge TTS (free, high quality)
"""

import os
import sys
import argparse
import random
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Audio parameters (matching your training pipeline)
SAMPLE_RATE = 16000
DURATION_SECONDS = 1.0
TARGET_SAMPLES = int(SAMPLE_RATE * DURATION_SECONDS)


def check_dependencies():
    """Check and report available TTS backends."""
    available_backends = []

    # Check pyttsx3 (offline, works everywhere)
    try:
        import pyttsx3
        available_backends.append('pyttsx3')
    except ImportError:
        pass

    # Check edge-tts (Microsoft voices, high quality, requires internet)
    try:
        import edge_tts
        available_backends.append('edge_tts')
    except ImportError:
        pass

    # Check Coqui TTS (neural TTS, high quality)
    try:
        from TTS.api import TTS
        available_backends.append('coqui_tts')
    except ImportError:
        pass

    # Check gTTS (Google TTS, requires internet)
    try:
        import gtts
        available_backends.append('gtts')
    except ImportError:
        pass

    return available_backends


def generate_with_pyttsx3(text: str, output_path: str, rate: int = 150,
                          voice_idx: int = 0) -> bool:
    """Generate speech using pyttsx3 (offline, system voices)."""
    try:
        import pyttsx3
        import tempfile
        import subprocess

        engine = pyttsx3.init()
        voices = engine.getProperty('voices')

        if voice_idx < len(voices):
            engine.setProperty('voice', voices[voice_idx].id)

        engine.setProperty('rate', rate)

        # pyttsx3 saves as wav directly
        temp_path = output_path.replace('.wav', '_temp.wav')
        engine.save_to_file(text, temp_path)
        engine.runAndWait()

        # Resample to target sample rate using ffmpeg or scipy
        if os.path.exists(temp_path):
            audio, sr = sf.read(temp_path)
            if sr != SAMPLE_RATE:
                import scipy.signal
                audio = scipy.signal.resample(audio, int(len(audio) * SAMPLE_RATE / sr))
            sf.write(output_path, audio, SAMPLE_RATE)
            os.remove(temp_path)
            return True
        return False
    except Exception as e:
        print(f"pyttsx3 error: {e}")
        return False


async def generate_with_edge_tts(text: str, output_path: str,
                                  voice: str = "en-US-GuyNeural") -> bool:
    """Generate speech using Microsoft Edge TTS (high quality, requires internet)."""
    try:
        import edge_tts
        import tempfile

        communicate = edge_tts.Communicate(text, voice)
        temp_path = output_path.replace('.wav', '_temp.mp3')
        await communicate.save(temp_path)

        # Convert to wav at correct sample rate
        if os.path.exists(temp_path):
            import subprocess
            subprocess.run([
                'ffmpeg', '-y', '-i', temp_path,
                '-ar', str(SAMPLE_RATE), '-ac', '1',
                output_path
            ], capture_output=True)
            os.remove(temp_path)
            return os.path.exists(output_path)
        return False
    except Exception as e:
        print(f"edge_tts error: {e}")
        return False


def generate_with_coqui(text: str, output_path: str,
                        model_name: str = "tts_models/en/ljspeech/tacotron2-DDC") -> bool:
    """Generate speech using Coqui TTS (neural TTS, high quality)."""
    try:
        from TTS.api import TTS

        tts = TTS(model_name=model_name, progress_bar=False)
        tts.tts_to_file(text=text, file_path=output_path)

        # Resample if needed
        audio, sr = sf.read(output_path)
        if sr != SAMPLE_RATE:
            import scipy.signal
            audio = scipy.signal.resample(audio, int(len(audio) * SAMPLE_RATE / sr))
            sf.write(output_path, audio, SAMPLE_RATE)
        return True
    except Exception as e:
        print(f"Coqui TTS error: {e}")
        return False


def apply_augmentation(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Apply random augmentation to audio sample."""
    augmented = audio.copy()

    # Random gain adjustment (volume variation)
    if random.random() < 0.7:
        gain = random.uniform(0.5, 1.5)
        augmented = augmented * gain

    # Add background noise
    if random.random() < 0.5:
        noise_level = random.uniform(0.001, 0.02)
        noise = np.random.randn(len(augmented)) * noise_level
        augmented = augmented + noise

    # Time shift (move audio slightly left or right)
    if random.random() < 0.5:
        shift = random.randint(-int(sr * 0.1), int(sr * 0.1))
        augmented = np.roll(augmented, shift)
        if shift > 0:
            augmented[:shift] = 0
        elif shift < 0:
            augmented[shift:] = 0

    # Pitch shift (using simple resampling - for better quality use librosa)
    if random.random() < 0.3:
        try:
            import librosa
            # Shift by up to 2 semitones
            n_steps = random.uniform(-2, 2)
            augmented = librosa.effects.pitch_shift(augmented, sr=sr, n_steps=n_steps)
        except:
            pass

    # Time stretch (speed up or slow down)
    if random.random() < 0.3:
        try:
            import librosa
            rate = random.uniform(0.9, 1.1)
            augmented = librosa.effects.time_stretch(augmented, rate=rate)
        except:
            pass

    # Add reverb (simple convolution with impulse response)
    if random.random() < 0.2:
        try:
            import scipy.signal
            # Simple room impulse response
            decay = random.uniform(0.1, 0.3)
            ir_length = int(sr * 0.1)
            ir = np.exp(-np.arange(ir_length) / (sr * decay))
            ir = ir / np.sum(ir)
            augmented = scipy.signal.convolve(augmented, ir, mode='same')
        except:
            pass

    # Normalize to prevent clipping
    max_val = np.max(np.abs(augmented))
    if max_val > 0.99:
        augmented = augmented * 0.95 / max_val

    return augmented


def pad_or_trim(audio: np.ndarray, target_length: int = TARGET_SAMPLES) -> np.ndarray:
    """Pad or trim audio to target length."""
    if len(audio) > target_length:
        # Center crop
        start = (len(audio) - target_length) // 2
        audio = audio[start:start + target_length]
    elif len(audio) < target_length:
        # Center pad
        pad_left = (target_length - len(audio)) // 2
        pad_right = target_length - len(audio) - pad_left
        audio = np.pad(audio, (pad_left, pad_right), mode='constant')
    return audio


def get_pronunciation_variants() -> List[str]:
    """
    Get various text representations of "Brisingr" to help TTS engines
    produce the correct pronunciation: BRIS-ing-gr
    """
    return [
        "Brisingr",           # Standard spelling
        "Brissingr",          # Double 's' for sharper sound
        "Briss-inger",        # Hyphenated for clarity
        "Brissinger",         # Combined spelling
        "Bris inger",         # Space separated
        "Brissingger",        # Emphasize hard 'g'
        "Briss-ing-ger",      # Full syllable separation
        "Brissingur",         # Alternate ending
        "Bris-sing-er",       # Another variant
        "Brissing-er",        # Partial hyphenation
    ]


def get_edge_tts_voices() -> List[str]:
    """Get a variety of English voices from Edge TTS."""
    return [
        "en-US-GuyNeural",
        "en-US-JennyNeural",
        "en-US-AriaNeural",
        "en-US-DavisNeural",
        "en-US-AmberNeural",
        "en-US-AnaNeural",
        "en-US-AndrewNeural",
        "en-US-BrandonNeural",
        "en-US-ChristopherNeural",
        "en-US-CoraNeural",
        "en-US-ElizabethNeural",
        "en-US-EricNeural",
        "en-US-JacobNeural",
        "en-US-JaneNeural",
        "en-US-JasonNeural",
        "en-US-MichelleNeural",
        "en-US-MonicaNeural",
        "en-US-NancyNeural",
        "en-US-RogerNeural",
        "en-US-SaraNeural",
        "en-US-SteffanNeural",
        "en-US-TonyNeural",
        "en-GB-RyanNeural",
        "en-GB-SoniaNeural",
        "en-GB-LibbyNeural",
        "en-AU-NatashaNeural",
        "en-AU-WilliamNeural",
        "en-CA-ClaraNeural",
        "en-CA-LiamNeural",
        "en-IN-NeerjaNeural",
        "en-IN-PrabhatNeural",
    ]


async def generate_edge_tts_samples(output_dir: str, num_samples: int = 100,
                                     augment_per_sample: int = 5):
    """Generate samples using Edge TTS with multiple voices."""
    import edge_tts

    os.makedirs(output_dir, exist_ok=True)

    voices = get_edge_tts_voices()
    pronunciations = get_pronunciation_variants()

    generated = 0
    sample_idx = 0

    print(f"Generating {num_samples} base samples with Edge TTS...")

    for i in range(num_samples):
        voice = random.choice(voices)
        text = random.choice(pronunciations)

        # Vary speech rate
        rate = random.choice(["-10%", "-5%", "+0%", "+5%", "+10%"])
        pitch = random.choice(["-5Hz", "+0Hz", "+5Hz"])

        try:
            communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
            temp_path = os.path.join(output_dir, f"temp_{sample_idx}.mp3")
            await communicate.save(temp_path)

            if os.path.exists(temp_path):
                # Convert to wav at correct sample rate
                import subprocess
                wav_path = os.path.join(output_dir, f"brisingr_{sample_idx:04d}.wav")
                result = subprocess.run([
                    'ffmpeg', '-y', '-i', temp_path,
                    '-ar', str(SAMPLE_RATE), '-ac', '1',
                    wav_path
                ], capture_output=True)
                os.remove(temp_path)

                if os.path.exists(wav_path):
                    # Load and create augmented versions
                    audio, _ = sf.read(wav_path)
                    audio = pad_or_trim(audio)
                    sf.write(wav_path, audio, SAMPLE_RATE)
                    sample_idx += 1
                    generated += 1

                    # Generate augmented versions
                    for aug_idx in range(augment_per_sample):
                        aug_audio = apply_augmentation(audio)
                        aug_audio = pad_or_trim(aug_audio)
                        aug_path = os.path.join(output_dir, f"brisingr_{sample_idx:04d}.wav")
                        sf.write(aug_path, aug_audio, SAMPLE_RATE)
                        sample_idx += 1
                        generated += 1

        except Exception as e:
            print(f"Error generating sample {i}: {e}")
            continue

        if i % 10 == 0:
            print(f"Progress: {i}/{num_samples} base samples ({generated} total with augmentation)")

    print(f"Generated {generated} total samples in {output_dir}")
    return generated


def generate_pyttsx3_samples(output_dir: str, num_samples: int = 50,
                             augment_per_sample: int = 5):
    """Generate samples using pyttsx3 (offline, system voices)."""
    import pyttsx3

    os.makedirs(output_dir, exist_ok=True)

    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    pronunciations = get_pronunciation_variants()

    generated = 0
    sample_idx = 0

    print(f"Generating {num_samples} base samples with pyttsx3...")
    print(f"Available voices: {len(voices)}")

    for i in range(num_samples):
        voice_idx = i % len(voices) if voices else 0
        if voices:
            engine.setProperty('voice', voices[voice_idx].id)

        # Vary speech rate
        rate = random.randint(120, 180)
        engine.setProperty('rate', rate)

        text = random.choice(pronunciations)

        try:
            wav_path = os.path.join(output_dir, f"brisingr_pyttsx3_{sample_idx:04d}.wav")
            temp_path = wav_path.replace('.wav', '_temp.wav')

            engine.save_to_file(text, temp_path)
            engine.runAndWait()

            if os.path.exists(temp_path):
                # Resample to target sample rate
                audio, sr = sf.read(temp_path)
                if len(audio.shape) > 1:
                    audio = audio[:, 0]  # Take first channel if stereo

                if sr != SAMPLE_RATE:
                    import scipy.signal
                    audio = scipy.signal.resample(audio, int(len(audio) * SAMPLE_RATE / sr))

                audio = pad_or_trim(audio)
                sf.write(wav_path, audio, SAMPLE_RATE)
                os.remove(temp_path)
                sample_idx += 1
                generated += 1

                # Generate augmented versions
                for aug_idx in range(augment_per_sample):
                    aug_audio = apply_augmentation(audio)
                    aug_audio = pad_or_trim(aug_audio)
                    aug_path = os.path.join(output_dir, f"brisingr_pyttsx3_{sample_idx:04d}.wav")
                    sf.write(aug_path, aug_audio, SAMPLE_RATE)
                    sample_idx += 1
                    generated += 1

        except Exception as e:
            print(f"Error generating sample {i}: {e}")
            continue

        if i % 10 == 0:
            print(f"Progress: {i}/{num_samples} base samples ({generated} total with augmentation)")

    print(f"Generated {generated} total samples in {output_dir}")
    return generated


def generate_from_existing_samples(input_dir: str, output_dir: str,
                                    augmentations_per_sample: int = 10):
    """
    Generate augmented samples from existing positive recordings.
    This is very effective if you have even a few real recordings.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Find existing wav files
    input_path = Path(input_dir)
    wav_files = list(input_path.glob("*.wav"))

    if not wav_files:
        print(f"No .wav files found in {input_dir}")
        return 0

    print(f"Found {len(wav_files)} existing samples")
    print(f"Generating {augmentations_per_sample} augmentations per sample...")

    generated = 0

    for wav_file in wav_files:
        try:
            audio, sr = sf.read(str(wav_file))

            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = audio[:, 0]

            # Resample if needed
            if sr != SAMPLE_RATE:
                import scipy.signal
                audio = scipy.signal.resample(audio, int(len(audio) * SAMPLE_RATE / sr))

            audio = pad_or_trim(audio)

            # Generate augmented versions
            base_name = wav_file.stem
            for aug_idx in range(augmentations_per_sample):
                aug_audio = apply_augmentation(audio)
                aug_audio = pad_or_trim(aug_audio)

                output_path = os.path.join(output_dir, f"{base_name}_aug{aug_idx:03d}.wav")
                sf.write(output_path, aug_audio, SAMPLE_RATE)
                generated += 1

        except Exception as e:
            print(f"Error processing {wav_file}: {e}")
            continue

    print(f"Generated {generated} augmented samples in {output_dir}")
    return generated


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic wake word training data for 'Brisingr'",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate with Edge TTS (recommended, high quality)
  python generate_synthetic_data.py --backend edge_tts --output ./synthetic_positive --num-samples 100

  # Generate with pyttsx3 (offline, works without internet)
  python generate_synthetic_data.py --backend pyttsx3 --output ./synthetic_positive --num-samples 50

  # Augment existing recordings (most effective!)
  python generate_synthetic_data.py --augment-existing ./my_recordings --output ./augmented_positive --aug-per-sample 20

  # Generate with all available backends
  python generate_synthetic_data.py --backend all --output ./synthetic_positive
        """
    )

    parser.add_argument('--backend', choices=['edge_tts', 'pyttsx3', 'coqui', 'all'],
                        default='all', help='TTS backend to use')
    parser.add_argument('--output', '-o', type=str, default='./synthetic_positive',
                        help='Output directory for generated samples')
    parser.add_argument('--num-samples', '-n', type=int, default=50,
                        help='Number of base samples to generate per backend')
    parser.add_argument('--aug-per-sample', type=int, default=5,
                        help='Number of augmented versions per base sample')
    parser.add_argument('--augment-existing', type=str,
                        help='Directory with existing recordings to augment')
    parser.add_argument('--list-backends', action='store_true',
                        help='List available TTS backends and exit')

    args = parser.parse_args()

    # Check available backends
    available = check_dependencies()
    print(f"Available TTS backends: {available}")

    if args.list_backends:
        print("\nBackend details:")
        print("  edge_tts   - Microsoft Edge TTS (high quality, requires internet)")
        print("  pyttsx3    - System TTS (offline, lower quality)")
        print("  coqui_tts  - Coqui Neural TTS (high quality, requires model download)")
        print("  gtts       - Google TTS (requires internet)")
        return

    total_generated = 0

    # Augment existing samples if provided
    if args.augment_existing:
        print("\n" + "="*60)
        print("Augmenting existing recordings...")
        print("="*60)
        count = generate_from_existing_samples(
            args.augment_existing,
            os.path.join(args.output, 'augmented'),
            args.aug_per_sample * 2  # More augmentations for real recordings
        )
        total_generated += count

    # Generate with selected backends
    if args.backend in ['edge_tts', 'all'] and 'edge_tts' in available:
        print("\n" + "="*60)
        print("Generating with Edge TTS...")
        print("="*60)
        import asyncio
        count = asyncio.run(generate_edge_tts_samples(
            os.path.join(args.output, 'edge_tts'),
            args.num_samples,
            args.aug_per_sample
        ))
        total_generated += count

    if args.backend in ['pyttsx3', 'all'] and 'pyttsx3' in available:
        print("\n" + "="*60)
        print("Generating with pyttsx3...")
        print("="*60)
        count = generate_pyttsx3_samples(
            os.path.join(args.output, 'pyttsx3'),
            args.num_samples,
            args.aug_per_sample
        )
        total_generated += count

    if args.backend in ['coqui', 'all'] and 'coqui_tts' in available:
        print("\n" + "="*60)
        print("Generating with Coqui TTS...")
        print("="*60)
        print("Note: Coqui TTS generation is slower but higher quality")
        # Coqui generation would go here - similar pattern

    print("\n" + "="*60)
    print(f"TOTAL GENERATED: {total_generated} samples")
    print(f"Output directory: {args.output}")
    print("="*60)

    if total_generated == 0:
        print("\nNo samples generated! Try installing a TTS backend:")
        print("  pip install edge-tts    # Recommended - high quality")
        print("  pip install pyttsx3     # Offline option")
        print("\nOr augment existing recordings:")
        print(f"  python {sys.argv[0]} --augment-existing ./your_recordings --output ./augmented")


if __name__ == "__main__":
    main()
