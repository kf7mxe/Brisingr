import torch
import numpy as np
import librosa
import pyaudio
import threading
import time
from collections import deque
import gc
import os
from datetime import datetime, timezone


class SileroVAD:
    """Silero VAD integration with more lenient settings"""
    
    def __init__(self, model_name='silero_vad', device='cpu'):
        self.device = device
        self.model = None
        self.sample_rate = 16000
        self.window_size_samples = 512
        self.threshold = 0.3  # LOWERED from 0.5 - less aggressive
        
        self._load_vad_model()
        self.vad_history = deque(maxlen=3)  # Reduced history for faster response
        self.overlap_samples = 256
        
    def _load_vad_model(self):
        """Load Silero VAD model with error handling"""
        try:
            model_path = 'silero_vad.jit'
            if not os.path.exists(model_path):
                print("Downloading Silero VAD model...")
                torch.hub.download_url_to_file(
                    'https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.jit',
                    model_path
                )
            
            self.model = torch.jit.load(model_path, map_location=self.device)
            self.model.eval()
            print(f"Silero VAD loaded successfully")
            
        except Exception as e:
            print(f"Warning: Could not load Silero VAD: {e}")
            print("Disabling VAD - using simple energy detection")
            self.model = None
    
    def detect_speech(self, audio_chunk):
        """More lenient speech detection"""
        if self.model is None:
            return self._fallback_vad(audio_chunk)
        
        try:
            # Use larger chunks for better context
            if len(audio_chunk) < 1024:  # Use at least 1024 samples
                audio_chunk = np.pad(audio_chunk, 
                                   (0, 1024 - len(audio_chunk)),
                                   mode='constant')
            
            # Take the last 1024 samples for analysis
            audio_chunk = audio_chunk[-1024:]
            
            speech_probabilities = []
            
            # Process in 512-sample windows with overlap
            for start_idx in range(0, len(audio_chunk) - 512 + 1, 256):
                window = audio_chunk[start_idx:start_idx + 512]
                
                if len(window) == 512:
                    audio_tensor = torch.FloatTensor(window).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        speech_prob = self.model(audio_tensor, self.sample_rate).item()
                        speech_probabilities.append(speech_prob)
            
            if speech_probabilities:
                # Use average instead of max for smoother detection
                aggregated_prob = np.mean(speech_probabilities)
            else:
                return self._fallback_vad(audio_chunk)
            
            # Lighter temporal smoothing
            self.vad_history.append(aggregated_prob)
            if len(self.vad_history) >= 2:
                smoothed_prob = np.mean(self.vad_history)
            else:
                smoothed_prob = aggregated_prob
            
            is_speech = smoothed_prob > self.threshold
            return is_speech, smoothed_prob
            
        except Exception as e:
            print(f"VAD error: {e}, falling back to energy detection")
            return self._fallback_vad(audio_chunk)
    
    def _fallback_vad(self, audio_chunk):
        """More lenient energy-based VAD"""
        energy = np.sum(audio_chunk ** 2) / len(audio_chunk)
        is_speech = energy > 0.005  # Lowered threshold
        return is_speech, energy


class LenientVolumeDetector:
    """More lenient volume detection"""
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.noise_floor = 0.0005  # Lowered
        self.volume_history = deque(maxlen=10)
        
        # Much more lenient thresholds
        self.min_volume_threshold = 0.001  # Significantly lowered
        
    def analyze_audio_quality(self, audio_chunk):
        """More lenient audio quality analysis"""
        rms_volume = np.sqrt(np.mean(audio_chunk ** 2))
        peak_volume = np.max(np.abs(audio_chunk))
        
        self.volume_history.append(rms_volume)
        
        # Update noise floor more conservatively
        if len(self.volume_history) >= 10:
            sorted_volumes = sorted(self.volume_history)
            self.noise_floor = max(0.0005, np.mean(sorted_volumes[:2]))  # Less aggressive
        
        # Much more lenient criteria
        volume_sufficient = rms_volume > max(self.min_volume_threshold, self.noise_floor * 2)  # Reduced multiplier
        not_clipped = peak_volume < 0.98
        
        # Simplified quality scoring - focus on volume only
        quality_score = 0.8 if volume_sufficient else 0.2
        
        return {
            'volume_ok': volume_sufficient,
            'not_clipped': not_clipped,
            'quality_score': quality_score,
            'rms_volume': rms_volume,
            'should_process': volume_sufficient  # Simplified decision
        }


class FixedEnhancedWakeWordDetector:
    def __init__(self, model_path='tiny_wake_word_optimized.pt', 
                 sample_rate=16000, 
                 n_mfcc=13,
                 chunk_size=1024,
                 detection_threshold=0.7,
                 smoothing_window=5,
                 power_save_mode=True,
                 gpu_batch_size=4,
                 use_vad=True,
                 vad_threshold=0.3):  # Lowered default
        
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.chunk_size = chunk_size
        self.detection_threshold = detection_threshold
        self.smoothing_window = smoothing_window
        self.power_save_mode = power_save_mode
        self.gpu_batch_size = gpu_batch_size
        self.use_vad = use_vad
        self.vad_threshold = vad_threshold
        
        # Audio settings
        self.frame_length = 512
        self.hop_length = 160
        self.buffer_size = sample_rate
        
        # Device selection
        self.device = self._select_optimal_device()
        print(f"Using device: {self.device}")
        
        # Initialize VAD with more lenient settings
        if self.use_vad:
            print("Initializing Silero VAD with lenient settings...")
            self.vad = SileroVAD(device='cpu')
            self.vad.threshold = vad_threshold
        else:
            self.vad = None
        
        # Initialize lenient volume detector
        self.volume_detector = LenientVolumeDetector(sample_rate)
        
        # Load model
        print(f"Loading model from {model_path}")
        self.model = self._load_and_optimize_model(model_path)
        print("Model loaded and optimized successfully")
        
        # Buffers and queues
        self.audio_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.buffer_lock = threading.Lock()
        self.detection_history = deque(maxlen=smoothing_window)
        
        # Audio setup
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        # Performance monitoring
        self.inference_times = deque(maxlen=50)
        
        # Simplified statistics
        self.stats = {
            'total_chunks': 0,
            'processed': 0,
            'detections': 0,
            'filtered_volume': 0,
            'filtered_vad': 0
        }
        
        # More aggressive processing for better detection
        self.processing_interval = 0.05 if power_save_mode else 0.03
        
        # Disable adaptive skipping initially
        self.skip_counter = 0
        self.skip_threshold = 10  # Much higher threshold
        
    def _select_optimal_device(self):
        """Select device"""
        if torch.cuda.is_available():
            return 'cuda:0'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    
    def _load_and_optimize_model(self, model_path):
        """Load model"""
        model = torch.jit.load(model_path, map_location='cpu')
        model.eval()
        model = model.to(self.device)
        
        if self.device.startswith('cuda'):
            model = torch.jit.optimize_for_inference(model)
            torch.cuda.empty_cache()
        elif self.device == 'mps':
            model = torch.jit.optimize_for_inference(model)
        else:
            torch.set_num_threads(2)
            model = torch.jit.optimize_for_inference(model)
        
        return model
    
    def should_process_audio(self, audio_signal):
        # """Much more lenient audio filtering"""
        # self.stats['total_chunks'] += 1
        
        # # Stage 1: Very lenient volume check
        # volume_analysis = self.volume_detector.analyze_audio_quality(audio_signal)
        
        # if not volume_analysis['should_process']:
        #     self.stats['filtered_volume'] += 1
        #     # Only skip if volume is REALLY low
        #     if volume_analysis['rms_volume'] < 0.0005:  # Very low threshold
        #         return False, f"volume_too_low_{volume_analysis['rms_volume']:.6f}"
        
        # # Stage 2: Lenient VAD check (only if enabled and working)
        # if self.use_vad and self.vad and self.vad.model is not None:
        #     try:
        #         is_speech, vad_prob = self.vad.detect_speech(audio_signal[-1024:])  # Use last 1024 samples
                
        #         if not is_speech and vad_prob < 0.1:  # Only skip if VAD is very confident it's not speech
        #             self.stats['filtered_vad'] += 1
        #             return False, f"no_speech_vad_{vad_prob:.3f}"
        #     except:
        #         # If VAD fails, don't filter - process anyway
        #         pass
        
        # # Stage 3: Minimal adaptive skipping
        # if volume_analysis['rms_volume'] < 0.001:  # Only for very quiet audio
        #     self.skip_counter += 1
        #     if self.skip_counter < self.skip_threshold:
        #         return False, "adaptive_skip"
        #     else:
        #         self.skip_counter = 0
        # else:
        #     self.skip_counter = 0
        
        return True, "processing"
    
    def extract_mfcc_optimized(self, audio_signal):
        """MFCC extraction (same as original)"""
        try:
            if len(audio_signal) != self.sample_rate:
                if len(audio_signal) > self.sample_rate:
                    audio_signal = audio_signal[:self.sample_rate]
                else:
                    audio_signal = np.pad(audio_signal, 
                                        (0, self.sample_rate - len(audio_signal)), 
                                        mode='constant')
            
            # Pre-emphasis filter
            audio_signal = np.append(audio_signal[0], 
                                   audio_signal[1:] - 0.97 * audio_signal[:-1])
            
            # Extract MFCCs
            mfccs = librosa.feature.mfcc(
                y=audio_signal,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=self.frame_length,
                hop_length=self.hop_length,
                n_mels=26,
                fmin=80,
                fmax=8000
            )

            # mfccs shape is (n_mfcc, frames), need to transpose for correct stats
            mfccs_T = mfccs.T  # Now shape is (frames, n_mfcc)
            mfcc_min = mfccs.min()
            mfcc_max = mfccs.max()
            mfcc_mean = mfccs.mean()
            # Get first frame's coefficients c0-c4 (first row of transposed matrix)
            first_frame = mfccs_T[0][:5] if len(mfccs_T) > 0 else []
            num_frames = mfccs.shape[1]  # Number of frames is second dimension

            audio_min = audio_signal.min()
            audio_max = audio_signal.max()
            audio_rms = np.sqrt(np.mean(audio_signal ** 2))
            now_utc = datetime.now(timezone.utc)
            timeFormated = now_utc.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            print("=== MFCC Debug (Python/Linux) ===")
            print(f"Timestamp: {timeFormated}")
            print(f"Audio: min={audio_min:.4f}, max={audio_max:.4f}, rms={audio_rms:.4f}")
            print(f"MFCC: frames={num_frames}, min={mfcc_min:.2f}, max={mfcc_max:.2f}, mean={mfcc_mean:.2f}")
            print(f"MFCC first frame (c0-c4): {[f'{x:.2f}' for x in first_frame]}")


            
            return mfccs.T
            
        except Exception as e:
            print(f"MFCC extraction error: {e}")
            return None
    
    def detect_wake_word_single(self, mfcc_features):
        """Single inference"""
        start_time = time.time()
        
        try:
            x = torch.FloatTensor(mfcc_features).unsqueeze(0).to(self.device, non_blocking=True)
            
            with torch.no_grad():
                if self.device.startswith('cuda') and torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        output = self.model(x)
                else:
                    output = self.model(x)
                
                probabilities = torch.softmax(output, dim=1)
                wake_word_prob = probabilities[0][1].item()
                print(f"Wake word probability: {wake_word_prob:.4f}")
                print()
            inference_time = (time.time() - start_time) * 1000
            self.inference_times.append(inference_time)
            
            return wake_word_prob
            
        except Exception as e:
            print(f"Inference error: {e}")
            return 0.0
    
    def smooth_detection(self, probability):
        """Temporal smoothing"""
        self.detection_history.append(probability)
        
        if len(self.detection_history) >= self.smoothing_window:
            return np.mean(self.detection_history)
        else:
            return probability
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Audio callback"""
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        
        with self.buffer_lock:
            shift_amount = len(audio_data)
            self.audio_buffer[:-shift_amount] = self.audio_buffer[shift_amount:]
            self.audio_buffer[-shift_amount:] = audio_data
        
        return (in_data, pyaudio.paContinue)
    
    def process_audio_continuously(self):
        """Main processing loop with debugging"""
        print("Starting Fixed Wake Word Detection...")
        print(f"Device: {self.device}")
        print(f"VAD enabled: {self.use_vad}")
        print(f"Detection threshold: {self.detection_threshold}")
        print("Say your wake word!")
        
        last_detection_time = 0
        detection_cooldown = 2.0
        last_stats_time = time.time()
        
        try:
            while self.stream.is_active():
                with self.buffer_lock:
                    current_buffer = self.audio_buffer.copy()
                
                # Apply filtering
                should_process, reason = self.should_process_audio(current_buffer)
                
                if should_process:
                    self.stats['processed'] += 1
                    
                    # Extract features
                    mfcc_features = self.extract_mfcc_optimized(current_buffer)
                    
                    if mfcc_features is not None:
                        # Single inference (no batching for now to simplify)
                        probability = self.detect_wake_word_single(mfcc_features)
                        smoothed_prob = self.smooth_detection(probability)

                        print("Probablility confidence  {smoothed_prob:.3f}")
                        
                        # Check for detection
                        current_time = time.time()
                        if (smoothed_prob > self.detection_threshold and 
                            current_time - last_detection_time > detection_cooldown):
                            
                            print(f"\n🎤 WAKE WORD DETECTED! (confidence: {smoothed_prob:.3f})")
                            self.stats['detections'] += 1
                            last_detection_time = current_time
                        
                        # Status display
                        if len(self.inference_times) > 0:
                            avg_time = np.mean(list(self.inference_times)[-5:])
                            print(f"\rProb: {smoothed_prob:.3f} | "
                                  f"Time: {avg_time:.1f}ms | "
                                  f"Reason: {reason}", end='', flush=True)
                
                # Show periodic statistics
                if time.time() - last_stats_time > 15:  # Every 15 seconds
                    self._print_debug_stats()
                    last_stats_time = time.time()
                
                time.sleep(self.processing_interval)
                
        except KeyboardInterrupt:
            print("\nStopping...")
        except Exception as e:
            print(f"Processing error: {e}")
    
    def _print_debug_stats(self):
        """Print debugging statistics"""
        total = max(1, self.stats['total_chunks'])
        processing_rate = (self.stats['processed'] / total) * 100
        volume_filter_rate = (self.stats['filtered_volume'] / total) * 100
        vad_filter_rate = (self.stats['filtered_vad'] / total) * 100
        
        print(f"\n--- Debug Stats ---")
        print(f"Processing rate: {processing_rate:.1f}%")
        print(f"Volume filtered: {volume_filter_rate:.1f}%")
        print(f"VAD filtered: {vad_filter_rate:.1f}%")
        print(f"Total detections: {self.stats['detections']}")
        
        # Warning if processing rate is too low
        if processing_rate < 30:
            print(f"WARNING: Low processing rate ({processing_rate:.1f}%) - filters may be too strict!")
        
        # Reset stats
        for key in self.stats:
            self.stats[key] = 0
    
    def start_detection(self):
        """Start detection"""
        try:
            # Warm up
            dummy_input = torch.randn(1, 100, self.n_mfcc, device=self.device)
            with torch.no_grad():
                _ = self.model(dummy_input)
            del dummy_input
            
            # Open audio stream
            self.stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self.audio_callback
            )
            
            self.stream.start_stream()
            self.process_audio_continuously()
            
        except Exception as e:
            print(f"Error starting detection: {e}")
        finally:
            self.stop_detection()
    
    def stop_detection(self):
        """Stop detection"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        print("Detection stopped")
    
    def disable_filtering(self):
        """Disable all filtering for debugging"""
        print("Disabling all filtering for debugging...")
        self.use_vad = False
        self.volume_detector.min_volume_threshold = 0.0001
        self.skip_threshold = 1000  # Effectively disable adaptive skipping
        print("All filters disabled - processing all audio")


def main():
    """Main function with debugging options"""
    print("Fixed Enhanced Wake Word Detector")
    print("This version uses more lenient filtering to avoid missing wake words")
    
    # More lenient default settings
    # Get the model path relative to this script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, '..', 'tiny_wake_word_optimized.pt')

    detector = FixedEnhancedWakeWordDetector(
        model_path=model_path,
        detection_threshold=0.65,  # Slightly lower threshold
        smoothing_window=3,        # Shorter smoothing for faster response
        power_save_mode=True,
        use_vad=True,              # But with lenient settings
        vad_threshold=0.3          # Lower VAD threshold
    )
    
    print("\nPress 'd' during detection to disable all filtering if wake word isn't detected")
    
    try:
        detector.start_detection() 
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        detector.stop_detection()


if __name__ == "__main__":
    main()