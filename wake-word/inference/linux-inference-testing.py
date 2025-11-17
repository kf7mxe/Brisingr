import numpy as np
import torch
import sounddevice as sd
import queue
import python_speech_features
import wavio
import os
import librosa

# Constants
MODEL_PATH = 'wake-word/model-convolution-16-16-android_lite_model.pt1'
SAMPLE_RATE = 8000
sample_rate = SAMPLE_RATE  # For compatibility with existing code
CHUNK_SIZE = 1024
CHANNELS = 1
BUFFER_SIZE = SAMPLE_RATE  # 1 second buffer
MFCC_WINLEN = 0.025  # 25ms window length
MFCC_WINSTEP = 0.01  # 10ms window step
MFCC_NUMCEP = 13     # Number of cepstral coefficients
MFCC_NFILT = 26     # Number of filters
MFCC_NFFT = 512     # FFT size
MFCC_LOWFREQ = 0    # Low frequency cutoff
MFCC_HIGHFREQ = None  # High frequency cutoff
MFCC_PREEMPH = 0.97  # Pre-emphasis coefficient

# Load model
model = torch.jit.load(MODEL_PATH)
model.eval()  # Set model to evaluation mode

# Initialize buffer
full_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)
buffer_index = 0
audio_queue = queue.Queue()

def callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

# Start recording stream
stream = sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=CHANNELS,
    dtype='float32',
    callback=callback,
    blocksize=CHUNK_SIZE
)

with stream:
    while True:
        # Get new audio chunk
        audio_data = audio_queue.get()
        audio_data = audio_data.squeeze()
        
        # Update circular buffer
        if buffer_index + CHUNK_SIZE <= BUFFER_SIZE:
            full_buffer[buffer_index:buffer_index + CHUNK_SIZE] = audio_data
        else:
            split_point = BUFFER_SIZE - buffer_index
            full_buffer[buffer_index:] = audio_data[:split_point]
            full_buffer[:CHUNK_SIZE - split_point] = audio_data[split_point:]
            
        buffer_index = (buffer_index + CHUNK_SIZE) % BUFFER_SIZE

        # Process audio in memory
        mfcc = python_speech_features.mfcc(
            full_buffer,
            samplerate=SAMPLE_RATE,
            winlen=MFCC_WINLEN,
            winstep=MFCC_WINSTEP,
            numcep=MFCC_NUMCEP,
            nfilt=MFCC_NFILT,
            nfft=MFCC_NFFT,
            lowfreq=MFCC_LOWFREQ,
            highfreq=MFCC_HIGHFREQ,
            preemph=MFCC_PREEMPH
        )
        
        # Normalize MFCC features
        mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
        
        # Prepare input tensor
        input_tensor = torch.tensor(mfcc, dtype=torch.float32)
        input_tensor = input_tensor.unsqueeze(0)
        input_tensor = input_tensor.unsqueeze(0)  # Add channel dimension
        
        # Convert to float tensor
        input_tensor = input_tensor.type(torch.FloatTensor)

        # Run inference
        with torch.no_grad():
            output = model(input_tensor)
            
        # Get prediction
        prediction = torch.argmax(output, dim=1)
        
        # Print result
        print("output ", output[0][0].item())
        if output[0][0] > 0.5:
            print("Wake word detected!")
        
        # Print result
        if prediction.item() == 1:
            print("Wake word detected!")
            print("Confidence: ", output[0][0].item())

        # Save buffer to file for debugging
        wavio.write("buffer.wav", full_buffer, SAMPLE_RATE, sampwidth=2)
        #     start = full_buffer_size - buffer_size
        #     stop = start + full_buffer_size
        #     wake_word_audio = full_buffer[start:stop]
    
        #     # save the wake word audio to a file
        #     import scipy.io.wavfile as wavfile
        #     wavfile.write('wake-word.wav', sample_rate, wake_word_audio)
    
        #     # stop the stream
        #     stream.stop_stream()