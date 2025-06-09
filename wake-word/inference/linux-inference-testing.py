from os import listdir
from os.path import isdir, join
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import python_speech_features
import sounddevice as sd
import queue
import librosa
import wavio
# load the pytorch model for making a hotword or wakeword detector in the directory /wake-word named model.pth


width_size = "16"
height_size = "16"

model_path = os.getcwd() + '/wake-word/model-convolution-'+width_size+'-'+height_size+'-android_lite_model.pt1'
print("model_path")
print(model_path)

input_size = 13



# load the jit model
model = torch.jit.load(model_path)

seconds = 1
sample_rate = 8000
num_mfcc = 20 * seconds
winlen = num_mfcc / sample_rate
chunk_size = 1024
channels = 1

# buffer size for 1 second of audio
full_buffer_size = sample_rate

# get data from the microphone every 0.5 seconds and put the data in a the full_buffer array putting the most recent data at the end of the array
full_buffer = np.zeros(full_buffer_size, dtype=np.float32)
buffer_index = 0

# Create a queue to store audio chunks
audio_queue = queue.Queue()

def callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

# Start recording stream
stream = sd.InputStream(
    samplerate=sample_rate,
    channels=channels,
    dtype='float32',
    callback=callback,
    blocksize=chunk_size
)

with stream:
    while stream.active:
        # shift the buffer down and new data in
        audio_data = audio_queue.get()
        audio_data = audio_data.squeeze()  # Remove extra dimension if present
        
        # Handle circular buffer wrap-around
        if buffer_index + chunk_size <= full_buffer_size:
            # Normal case: write directly
            full_buffer[buffer_index:buffer_index + chunk_size] = audio_data
        else:
            # Wrap-around case: split the write into two parts
            split_point = full_buffer_size - buffer_index
            full_buffer[buffer_index:] = audio_data[:split_point]
            full_buffer[:chunk_size - split_point] = audio_data[split_point:]
            
        buffer_index = (buffer_index + chunk_size) % full_buffer_size
        
        # Print buffer index and chunk position for debugging
        # print(f"Buffer index: {buffer_index}, chunk position: {buffer_index + chunk_size}")

        # print out the full_buffer size
        # print("full_buffer.shape")
        # print(full_buffer.shape)


        # save buffer to a file to wav file for debugging
        wavio.write("buffer.wav", full_buffer, sample_rate, sampwidth=2)


        # read buffer.wav file
        path = os.path.abspath(os.path.join(os.getcwd(), 'buffer.wav'))
        # print("does the path exist")
        # print(os.path.exists(path))
        signal, file_sample_rate = librosa.load(path, sr=sample_rate)
      




        mfcc = python_speech_features.base.mfcc(signal,samplerate=sample_rate,
                                             winlen=winlen,
                                            numcep=13,
                                            nfilt=26,
                                            nfft=512,
                                            lowfreq=0,
                                            highfreq=sample_rate/2)
        
        # print("mfcc.shape")
        # print(mfcc.shape)


        # convert from numpy array to torch tensor
        x = torch.from_numpy(mfcc)

        # Reshape to match expected input shape (1, 16, 16)
        x = x[:16, :16]  # Take first 16x16 features
        x = x.unsqueeze(0)  # Add batch dimension
        x = x.unsqueeze(0)  # Add channel dimension
        
        # Convert to float tensor
        x = x.type(torch.FloatTensor)

        # Print tensor shape for debugging
        # print("Input tensor shape:", x.shape)

        # import matplotlib.pyplot as plt
        # # save the mfcc to 
        # fig = plt.figure()
        # plt.imshow(x.squeeze(), cmap='hot', interpolation='nearest')

        # print("stuff")
        # print(x.shape)

        # # run the model on the buffer
        output = model(x)
        # print("Output")
        # print(output)
    
        # # if the model found the wake word][])
        # print("Percentage:", output[0][0].item())
        if output[0][0] > 0.5:
            print("Percentage ", output[0][0].item())
            print("Wake word detected")
        #     # extract the 1 second of audio that contains thewake  word
        #     start = full_buffer_size - buffer_size
        #     stop = start + full_buffer_size
        #     wake_word_audio = full_buffer[start:stop]
    
        #     # save the wake word audio to a file
        #     import scipy.io.wavfile as wavfile
        #     wavfile.write('wake-word.wav', sample_rate, wake_word_audio)
    
        #     # stop the stream
        #     stream.stop_stream()