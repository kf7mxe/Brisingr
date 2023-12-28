from os import listdir
from os.path import isdir, join
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import python_speech_features
import pyaudio
cwd = os.getcwd()
import wavio
# load the pytorch model for making a hotword or wakeword detector in the directory /wake-word named model.pth


width_size = "13"
height_size = "101"

model_path = cwd + '/wake-word/model-convolution-'+width_size+'-'+height_size+'-android_lite_model.pt1'
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
format = pyaudio.paFloat32

# create a stream object to get data from the microphone
microphone = pyaudio.PyAudio()


# buffer size for 1 second of audio
full_buffer_size = sample_rate

# get data from the microphone every 0.5 seconds and put the data in a the full_buffer array putting the most recent data at the end of the array
full_buffer = np.zeros(full_buffer_size, dtype=np.float32)
buffer_index = 0

stream = microphone.open(format=format, channels=channels, rate=sample_rate, input=True, frames_per_buffer=chunk_size)

# create a window to apply to the audio signal
# window = 0.5 * (1 - np.cos(np.linspace(0, 2 * np.pi, buffer_size, False)))

stream.start_stream()

# loop to process audio and look for the wake word
while stream.is_active():
    
        # shift the buffer down and new data in
        audio_data = np.frombuffer(stream.read(chunk_size), dtype=np.float32)

        full_buffer[buffer_index:buffer_index + chunk_size] = audio_data

        buffer_index = (buffer_index + chunk_size) % full_buffer_size

        # print out the full_buffer size
        print("full_buffer.shape")
        print(full_buffer.shape)


        # save buffer to a file to wav file for debugging
        wavio.write("buffer.wav", full_buffer, sample_rate, sampwidth=2)


        # read buffer.wav file
        import librosa
        path = os.path.abspath(os.path.join(cwd, 'buffer.wav'))
        print("does the path exist")
        print(os.path.exists(path))
        signal, file_sample_rate = librosa.load(path, sr=sample_rate)
      




        mfcc = python_speech_features.base.mfcc(signal,samplerate=sample_rate,
                                             winlen=winlen,
                                            numcep=13,
                                            nfilt=26,
                                            nfft=512,
                                            lowfreq=0,
                                            highfreq=sample_rate/2)
        
        print("mfcc.shape")
        print(mfcc.shape)


        # convert from numpy array to torch tensor
        x = torch.from_numpy(mfcc)

        # switch the axis of the tensor
        x = x.transpose(0, 1)


        import matplotlib.pyplot as plt
        # save the mfcc to 
        fig = plt.figure()
        plt.imshow(x, cmap='hot', interpolation='nearest')
        plt.savefig('mfcc-inference-record.png')


    
    

        x = x.unsqueeze(0)
        # x = x.unsqueeze(1)

        x = x.type(torch.FloatTensor)

        print("stuff")
        print(x.shape)

        # # run the model on the buffer
        # output = model(x)
    
        # # if the model found the wake word
        # if output[0][0] > 0.5:
        #     print("Wake word detected")
    
        #     # extract the 1 second of audio that contains the wake word
        #     start = full_buffer_size - buffer_size
        #     stop = start + full_buffer_size
        #     wake_word_audio = full_buffer[start:stop]
    
        #     # save the wake word audio to a file
        #     import scipy.io.wavfile as wavfile
        #     wavfile.write('wake-word.wav', sample_rate, wake_word_audio)
    
        #     # stop the stream
        #     stream.stop_stream()