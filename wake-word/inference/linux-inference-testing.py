from os import listdir
from os.path import isdir, join
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import python_speech_features

# load the pytorch model for making a hotword or wakeword detector in the directory /wake-word named model.pth
model_path = Path (os.path.dirname(__file__)).parent  / 'model.ckpt'
print("model_path")
print(model_path)

input_size = 16

class Net(nn.Module):
    def __init__(self):   
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, input_size, kernel_size=3, stride=1, padding=1) # 16, 16, 16
        self.convnorm1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2) # 16, 8, 8
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1) # 32, 8, 8
        self.convnorm2 = nn.BatchNorm2d(32)
        self.pool2 = nn.AvgPool2d(kernel_size=2) # 32, 4, 4
        self.linear1 = nn.Linear(32*4*4, 400) # 32*4*4, 400
        self.linear1_bn = nn.BatchNorm1d(400)
        self.drop = nn.Dropout(0.5)
        self.linear2 = nn.Linear(400, 2) # 400, 2
        
    def forward(self, x):

        x = self.conv1(x)
        x = self.convnorm1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.pool2(F.relu(self.convnorm2(self.conv2(x))))
        x = x.view(-1, 32*4*4)
        x = F.relu(self.linear1(x))
        x = self.linear1_bn(x)
        x = self.drop(x)
        x = self.linear2(x)
        return F.log_softmax(x, dim=1)
    



model = Net()

model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))


model.eval()

sample_rate = 8000

# buffer size for 1 second of audio
full_buffer_size = sample_rate

# get data from the microphone every 0.5 seconds and put the data in a the full_buffer array putting the most recent data at the end of the array
full_buffer = np.zeros(full_buffer_size, dtype=np.float32)
buffer_size = int(sample_rate / 2)
buffer = np.zeros(buffer_size, dtype=np.float32)

# create a stream object to get data from the microphone
import pyaudio
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32, channels=1, rate=sample_rate, input=True, frames_per_buffer=buffer_size)

# create a window to apply to the audio signal
# window = 0.5 * (1 - np.cos(np.linspace(0, 2 * np.pi, buffer_size, False)))

stream.start_stream()

# loop to process audio and look for the wake word
while stream.is_active():
    
        # shift the buffer down and new data in
        full_buffer[:-buffer_size] = full_buffer[buffer_size:]
        full_buffer[-buffer_size:] = buffer
    
        # read the microphone data into buffer
        buffer = np.fromstring(stream.read(buffer_size), dtype=np.float32)
    
        # apply the hamming window
        # buffer *= window

        print(buffer.shape)


        mfcc = python_speech_features.base.mfcc(buffer,samplerate=sample_rate,
                                            winlen=0.35,
                                            numcep=16)


    
    
        # convert from numpy array to torch tensor
        x = torch.from_numpy(mfcc)

        x = x.unsqueeze(1)
        x = x.unsqueeze(1)

        x = x.type(torch.FloatTensor)

        print("stuff")
        print(x.shape)

        # run the model on the buffer
        output = model(x)
    
        # if the model found the wake word
        if output[0][0] > 0.5:
            print("Wake word detected")
    
            # extract the 1 second of audio that contains the wake word
            start = full_buffer_size - buffer_size
            stop = start + full_buffer_size
            wake_word_audio = full_buffer[start:stop]
    
            # save the wake word audio to a file
            import scipy.io.wavfile as wavfile
            wavfile.write('wake-word.wav', sample_rate, wake_word_audio)
    
            # stop the stream
            stream.stop_stream()