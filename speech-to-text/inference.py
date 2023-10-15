
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

import numpy as np

import pyaudio

import time














# pyaudio = pyaudio.PyAudio()
# stream = pyaudio.open(format=pyaudio.paFloat32, channels=1, rate=8000, input=True, frames_per_buffer=4000)

# stream.start_stream()

# while stream.is_active():
#     data = stream.read(4000)
#     data = torch.from_numpy(np.fromstring(data, dtype=np.float32))
#     data = data.view(1, 1, -1)
#     data = data.to(device)
#     output = model(data)
#     print(output)
#     print(decoder(output.cpu()))
#     time.sleep(0.5)

# stream.stop_stream()
# stream.close()
# pyaudio.terminate()



