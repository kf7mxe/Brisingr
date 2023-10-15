from os import listdir
from os.path import isdir, join
import os
import numpy as np
from torch.utils.mobile_optimizer import optimize_for_mobile

# load wake-word-settings.json 
import json


# get the parent directory 
parent_dir = os.path.dirname(os.path.dirname(__file__))

with open(join(os.path.dirname(os.path.dirname(__file__)),'wake-word-settings.json')) as data:
    data = json.load(data)
    len_mfcc = data["num_mfcc"]
    winlen = data["winlen"]


# Create list of all targets (minus background noise)
# dataset_path = "/media/kf7mxe/Elements/machine-learning/training-data/Brisingr wakeword/"
# all_targets = all_targets = [name for name in listdir(dataset_path) if isdir(join(dataset_path, name))]
# # all_targets.remove('_background_noise_')
# print("all_targets")
# print(all_targets)


# Settings
feature_sets_filename = 'all_targets_mfcc_sets.npz'
model_filename = 'brisingr-30epochs-1s-no-copies-no-augmentation-removed-downloaded-backgroundnoise.h5'
wake_word = 'positive'


# Load feature sets
feature_sets = np.load(join(os.path.dirname(__file__), feature_sets_filename))
print(feature_sets.files)


# Assign feature sets
x_train = feature_sets['x_train']
y_train = feature_sets['y_train']
x_val = feature_sets['x_val']
y_val = feature_sets['y_val']
x_test = feature_sets['x_test']
y_test = feature_sets['y_test']

# Look at tensor dimensions
print("x_train.shape")
print(x_train.shape)


# Convert ground truth arrays to one wake word (1) and 'other' (0)
# wake_word_index = all_targets.index(wake_word)
wake_word_index = 1
print("wake word index")
print(wake_word_index)
y_train = np.equal(y_train, wake_word_index).astype('float64')
y_val = np.equal(y_val, wake_word_index).astype('float64')
y_test = np.equal(y_test, wake_word_index).astype('float64')

# What percentage of 'stop' appear in validation labels
print(sum(y_val) / len(y_val))
print(1 - sum(y_val) / len(y_val))
# View the dimensions of our input data
print(x_train.shape)






# # balance x_train and y_train
# # get the count of each class in y_train
# unique, counts = np.unique(y_train, return_counts=True)

# # get the index of each class in y_train
# class0_index = np.where(y_train == 0)[0]
# class1_index = np.where(y_train == 1)[0]

# small_class = None
# largest_count = None
# if len(class0_index) > len(class1_index):
#     small_class = class1_index
#     largest_count = len(class1_index)
# elif len(class0_index) < len(class1_index):
#     small_class = class0_index
#     largest_count = len(class0_index)
# else:
#     print("the classes are already balanced")

# # get the number of samples to remove from class0
# num_samples_to_remove = len(class0_index) - largest_count
# print("num_samples_to_remove")
# print(num_samples_to_remove)

# # randomly choose num_samples_to_remove samples from class0
# random_indices = np.random.choice(class0_index, num_samples_to_remove, replace=False)

# # remove the samples from class0
# x_train = np.delete(x_train, random_indices, axis=0)
# y_train = np.delete(y_train, random_indices, axis=0)

# # get the count of each class in y_train
# unique, counts = np.unique(y_train, return_counts=True)

# print("unique")
# print(unique)
# print("counts")
# print(counts)






# normalize x_train, x_test
x_train = x_train / 255
x_test = x_test / 255






# Create a pytorch model for makeing a hotword or wakeword detector
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Create a pytorch dataset
class HotwordDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
# Create a pytorch dataloader
def get_dataloader(x, y, batch_size):
    dataset = HotwordDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

batch_size = 3
# batch_size = 1


# get the count of each class in y_train
unique, counts = np.unique(y_train, return_counts=True)
# get the largest unique count label
if counts[0] > counts[1]:
    largest_count = counts[0]
else:
    largest_count = counts[1]

# calculate the accuracy if we always predict the largest count label
print("accuracy if we always predict the largest count label")
print(largest_count / len(y_train))





train_loader = get_dataloader(x_train, y_train, batch_size)
test_loader = get_dataloader(x_test, y_test, 1)



input_size = x_train.shape[1]

class Net(nn.Module):
    def __init__(self):   
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, input_size, kernel_size=3, stride=1, padding=1) # 16, 16, 16
        self.convnorm1 = nn.BatchNorm2d(input_size)
        self.pool1 = nn.MaxPool2d(kernel_size=2) # 16, 8, 8
        self.conv2 = nn.Conv2d(input_size, 32, kernel_size=3, stride=1, padding=1) # 32, 8, 8
        self.convnorm2 = nn.BatchNorm2d(32)
        self.pool2 = nn.AvgPool2d(kernel_size=2) # 32, 4, 4
        self.linear1 = nn.Linear(2400, 400) # 32*4*4, 400
        self.linear1_bn = nn.BatchNorm1d(400)
        self.drop = nn.Dropout(0.1)
        self.linear2 = nn.Linear(400, 2) # 400, 2
        
    def forward(self, x):

        x = self.conv1(x)
        x = self.convnorm1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.convnorm2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear1_bn(x) 
        x = self.drop(x)
        x = self.linear2(x)
        return F.log_softmax(x, dim=1)


    
model = Net().to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.001)
criterion = nn.NLLLoss()


# print input size and output sizes of the model


# Training loop
for epoch in range(1):
    train_loss = 0
    for i, data in enumerate(train_loader):
        inputs, labels = data

        # unsqueeze the inputs to add a channel dimension
        inputs = inputs.unsqueeze(1)

        # convert to FloatTensor 
        inputs = inputs.type(torch.FloatTensor)

        inputs, labels = Variable(inputs), Variable(labels)
 
        # print("inputs")
        # print(inputs.shape)
        optimizer.zero_grad()
        # print("input size")
        # print(inputs.shape)

        if inputs.shape[0] != batch_size:
            break

        outputs = model(inputs)
        loss = criterion(outputs, labels.long())
        # print("right here test")
        # print(loss.data)
        # print(loss.item())
        # train_loss += loss.data[0]
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    print('epoch: {}, Loss: {:.4f}'.format(epoch+1, train_loss / len(train_loader)))
    # print accuracy on test set
    model.eval()
    correct = 0
    total = 0
    for data in test_loader:
        inputs, labels = data
        # print("inputs")
        # print(inputs.shape)
        inputs = inputs.unsqueeze(1)
        inputs = inputs.type(torch.FloatTensor)
        inputs, labels = Variable(inputs), Variable(labels)
        # print("in the test loop")
        # print(inputs.shape)
        outputs = model(inputs)
        # print("test output")
        # print(outputs.shape)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.long()).sum()

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

# Test the model
model.eval()
correct = 0
total = 0
for data in test_loader:
    inputs, labels = data
    # print("inputs")
    # print(inputs.shape)
    inputs = inputs.unsqueeze(1)
    inputs = inputs.type(torch.FloatTensor)
    inputs, labels = Variable(inputs), Variable(labels)
    # print("in the test loop")
    # print(inputs.shape)
    outputs = model(inputs)
    # print("test output")
    # print(outputs.shape)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels.long()).sum()

print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

# Save the model and plot
torch.save(model.state_dict(), 'wake-word/model-convolution-{len_mfcc}.ckpt')

# load model
# model = torch.load('wake-word/model-convolution-{len_mfcc}.ckpt')

# infer on a random input
x = torch.randn(1, 1, 16, 16, requires_grad=True)
torch_out = model(x)

model.eval()

model_trace_script = torch.jit.trace(model, x)
model_for_android = optimize_for_mobile(model_trace_script)
model_for_android._save_for_lite_interpreter("wake-word/model-convolution-{len_mfcc}-android_lite_model.pt1")




#####################################################################

# simple pytorch neural network that accepts a tensor of size of 1 x num_mfcc x num_mfcc and outputs the probability of the input being the wake word
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(len_mfcc * len_mfcc, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = x.view(-1, len_mfcc * len_mfcc)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x
    

simple_model = SimpleNet()
optimizer = optim.Adam(simple_model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# Training loop
for epoch in range(1):
    train_loss = 0
    for i, data in enumerate(train_loader):
        inputs, labels = data

        # unsqueeze the inputs to add a channel dimension
        inputs = inputs.unsqueeze(1)

        # convert to FloatTensor 
        inputs = inputs.type(torch.FloatTensor)

        inputs, labels = Variable(inputs), Variable(labels)
 
        optimizer.zero_grad()
        outputs = simple_model(inputs)
        loss = criterion(outputs, labels.long())
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    print('epoch: {}, Loss: {:.4f}'.format(epoch+1, train_loss / len(train_loader)))

# Test the model
simple_model.eval()
correct = 0
total = 0
for data in test_loader:
    inputs, labels = data
    print("inputs")
    print(inputs.shape)
    inputs = inputs.unsqueeze(1)
    inputs = inputs.type(torch.FloatTensor)
    inputs, labels = Variable(inputs), Variable(labels)
    print("in the test loop")
    print(inputs.shape)
    outputs = simple_model(inputs)
    print("test output")
    print(outputs.shape)
    print(outputs.data)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels.long()).sum()

print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

# Save the model and plot
torch.save(simple_model.state_dict(), 'wake-word/simple-model-{len_mfcc}.ckpt')


