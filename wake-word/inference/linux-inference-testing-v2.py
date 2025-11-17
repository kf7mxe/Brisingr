import torch
import numpy as np
import librosa
import pyaudio
import threading
import time
from collections import deque
import matplotlib.pyplot as plt

class OptimizedWakeWordDetector:
    def __init__(self, model_path='tiny_wake_word_optimized.pt', 
                 sample_rate=16000, 
                 n_mfcc=13,
                 chunk_size=1024,
                 detection_threshold=0.7,
                 smoothing_window=5):
        
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.chunk_size = chunk_size
        self.detection_threshold = detection_threshold
        self.smoothing_window = smoothing_window
        
        # Audio settings
        self.frame_length = 512
        self.hop_length = 160  # 10ms hops
        self.buffer_size = sample_rate  # 1 second buffer
        
        # Load model
        print(f"Loading model from {model_path}")
        self.model = torch.jit.load(model_path, map_location='cpu')
        self.model.eval()
        print("Model loaded successfully")
        
        # Audio buffer
        self.audio_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.buffer_lock = threading.Lock()
        
        # Detection smoothing
        self.detection_history = deque(maxlen=smoothing_window)
        
        # PyAudio setup
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        # Performance monitoring
        self.inference_times = deque(maxlen=100)
        
    def extract_mfcc_optimized(self, audio_signal):
        """Optimized MFCC extraction"""
        try:
            # Ensure correct length
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
                fmin=0,
                fmax=self.sample_rate//2
            )
            
            # Transpose to (time, features)
            return mfccs.T
            
        except Exception as e:
            print(f"MFCC extraction error: {e}")
            return None
    
    def detect_wake_word(self, mfcc_features):
        """Run inference with performance monitoring"""
        start_time = time.time()
        
        try:
            # Convert to tensor
            x = torch.FloatTensor(mfcc_features).unsqueeze(0)  # Add batch dimension
            
            # Inference
            with torch.no_grad():
                output = self.model(x)
                probabilities = torch.exp(output)  # Convert from log probabilities
                wake_word_prob = probabilities[0][1].item()  # Probability of wake word class
            
            # Record inference time
            inference_time = (time.time() - start_time) * 1000  # ms
            self.inference_times.append(inference_time)
            
            return wake_word_prob
            
        except Exception as e:
            print(f"Inference error: {e}")
            return 0.0
    
    def smooth_detection(self, probability):
        """Apply temporal smoothing to reduce false positives"""
        self.detection_history.append(probability)
        
        if len(self.detection_history) >= self.smoothing_window:
            # Use weighted average with more weight on recent detections
            weights = np.exp(np.linspace(0, 1, len(self.detection_history)))
            smoothed_prob = np.average(self.detection_history, weights=weights)
            return smoothed_prob
        else:
            return probability
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback for continuous audio capture"""
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        
        with self.buffer_lock:
            # Shift buffer and add new data
            shift_amount = len(audio_data)
            self.audio_buffer[:-shift_amount] = self.audio_buffer[shift_amount:]
            self.audio_buffer[-shift_amount:] = audio_data
        
        return (in_data, pyaudio.paContinue)
    
    def process_audio_continuously(self):
        """Main processing loop"""
        print("Starting wake word detection...")
        print(f"Detection threshold: {self.detection_threshold}")
        print(f"Say your wake word!")
        
        last_detection_time = 0
        detection_cooldown = 2.0  # seconds
        
        try:
            while self.stream.is_active():
                # Get current buffer
                with self.buffer_lock:
                    current_buffer = self.audio_buffer.copy()
                
                # Extract features
                mfcc_features = self.extract_mfcc_optimized(current_buffer)
                
                if mfcc_features is not None:
                    # Detect wake word
                    probability = self.detect_wake_word(mfcc_features)
                    smoothed_prob = self.smooth_detection(probability)
                    
                    # Check for detection
                    current_time = time.time()
                    if (smoothed_prob > self.detection_threshold and 
                        current_time - last_detection_time > detection_cooldown):
                        
                        print(f"\n🎤 WAKE WORD DETECTED! (confidence: {smoothed_prob:.3f})")
                        last_detection_time = current_time
                        
                        # Optional: Save detection audio for analysis
                        # self.save_detection_audio(current_buffer, smoothed_prob)
                    
                    # Performance monitoring
                    if len(self.inference_times) > 0:
                        avg_inference_time = np.mean(list(self.inference_times)[-10:])
                        print(f"\rProb: {smoothed_prob:.3f} | "
                              f"Inference: {avg_inference_time:.1f}ms | "
                              f"Threshold: {self.detection_threshold}", end='', flush=True)
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.05)
                
        except KeyboardInterrupt:
            print("\nStopping wake word detection...")
        except Exception as e:
            print(f"Processing error: {e}")
    
    def save_detection_audio(self, audio_data, confidence):
        """Save audio when wake word is detected for analysis"""
        try:
            import soundfile as sf
            timestamp = int(time.time())
            filename = f"detection_{timestamp}_conf_{confidence:.3f}.wav"
            sf.write(filename, audio_data, self.sample_rate)
            print(f"Detection audio saved: {filename}")
        except ImportError:
            pass  # soundfile not available
        except Exception as e:
            print(f"Error saving audio: {e}")
    
    def start_detection(self):
        """Start the wake word detection system"""
        try:
            # Open audio stream
            self.stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self.audio_callback
            )
            
            # Start streaming
            self.stream.start_stream()
            
            # Start processing
            self.process_audio_continuously()
            
        except Exception as e:
            print(f"Error starting detection: {e}")
        finally:
            self.stop_detection()
    
    def stop_detection(self):
        """Stop the wake word detection system"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        print("Wake word detection stopped")
    
    def adjust_threshold(self, new_threshold):
        """Dynamically adjust detection threshold"""
        self.detection_threshold = max(0.1, min(0.9, new_threshold))
        print(f"Detection threshold adjusted to: {self.detection_threshold}")
    
    def get_performance_stats(self):
        """Get performance statistics"""
        if len(self.inference_times) > 0:
            avg_time = np.mean(self.inference_times)
            max_time = np.max(self.inference_times)
            min_time = np.min(self.inference_times)
            
            print(f"\nPerformance Statistics:")
            print(f"Average inference time: {avg_time:.2f}ms")
            print(f"Min inference time: {min_time:.2f}ms")
            print(f"Max inference time: {max_time:.2f}ms")
            print(f"Target real-time: < 50ms (Current: {'✓' if avg_time < 50 else '✗'})")
        
        return self.inference_times

# Interactive detection with threshold adjustment
class InteractiveWakeWordDetector(OptimizedWakeWordDetector):
    """Interactive version with keyboard controls"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.running = True
    
    def keyboard_listener(self):
        """Listen for keyboard commands"""
        print("\nKeyboard Controls:")
        print("  'q' - Quit")
        print("  '+' - Increase threshold")
        print("  '-' - Decrease threshold")
        print("  's' - Show stats")
        
        while self.running:
            try:
                key = input().strip().lower()
                if key == 'q':
                    self.running = False
                    break
                elif key == '+':
                    self.adjust_threshold(self.detection_threshold + 0.05)
                elif key == '-':
                    self.adjust_threshold(self.detection_threshold - 0.05)
                elif key == 's':
                    self.get_performance_stats()
            except:
                break
    
    def start_detection(self):
        """Start interactive detection"""
        # Start keyboard listener in separate thread
        keyboard_thread = threading.Thread(target=self.keyboard_listener, daemon=True)
        keyboard_thread.start()
        
        # Start detection
        super().start_detection()

def main():
    """Main function to run the wake word detector"""
    # Initialize detector
    detector = InteractiveWakeWordDetector(
        model_path='tiny_wake_word_optimized.pt',
        detection_threshold=0.7,
        smoothing_window=3
    )
    
    try:
        detector.start_detection()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        detector.stop_detection()

if __name__ == "__main__":
    main()