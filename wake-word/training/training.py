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
# x_train = x_train / 255
# x_test = x_test / 255






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
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        # Get the sample and convert to tensor
        x = torch.FloatTensor(self.x_data[idx])
        # Add channel dimension
        x = x.unsqueeze(0)  # Shape becomes (1, 16, 16)
        y = torch.FloatTensor([self.y_data[idx]])
        return x, y

# Create a pytorch dataloader
def get_dataloader(x, y, batch_size):
    dataset = HotwordDataset(x, y)
    # Custom collate function to handle input shapes
    def collate_fn(batch):
        inputs = torch.stack([item[0] for item in batch])  # Stack inputs
        labels = torch.stack([item[1] for item in batch])  # Stack labels
        return inputs, labels
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

batch_size = 3
# batch_size = 1


# get the count of each class in y_train
unique, counts = np.unique(y_train, return_counts=True)
# get the largest unique count label
if counts[0] > counts[1]:
    largest_count = counts[0]
else:
    largest_count = counts[1]

# count of each classes
print("count of each classes")
print(counts)

# calculate the accuracy if we always predict the largest count label
print("accuracy if we always predict the largest count label")
print(largest_count / len(y_train))

# Create train, validation, and test loaders
train_loader = get_dataloader(x_train, y_train, batch_size)
val_loader = get_dataloader(x_val, y_val, batch_size)
test_loader = get_dataloader(x_test, y_test, batch_size)



input_size = x_train.shape[1]

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input shape is (batch_size, 256) after flattening
        self.linear1 = nn.Linear(256, 128)
        self.linear1_bn = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(0.1)
        self.linear2 = nn.Linear(128, 1)  # Binary classification
        self.linear1_bn = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(0.1)
        self.linear2 = nn.Linear(128, 1)  # Binary classification
        
    def forward(self, x):
        # Input shape should be (batch_size, 1, 16, 16)
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 256)
        x = self.linear1(x)
        x = self.linear1_bn(x)
        x = F.relu(x)
        x = self.drop(x)
        x = self.linear2(x)
        return torch.sigmoid(x)  # Binary classification needs sigmoid at the end


    
input_size = x_train.shape[1]
model = Net()
model = model.to(device)

# Print model summary to verify input dimensions
print("Model Summary:")
# print(model)

# Test input tensor shape
batch_size = 3
x_test = torch.randn(batch_size, 1, input_size, input_size).to(device)
print("\nTest input shape:", x_test.shape)

# Forward pass to verify output shape
with torch.no_grad():
    test_output = model(x_test)
    print("Test output shape:", test_output.shape)

optimizer = optim.AdamW(model.parameters(), lr=0.001)
criterion = nn.BCELoss()


# print input size and output sizes of the model


# Training loop
epochs = 1
for epoch in range(epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        # Skip if batch size is not as expected
        if inputs.shape[0] != batch_size:
            continue
            
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(inputs)
        # BCELoss requires float targets
        loss = criterion(outputs, labels.float())
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print('epoch: {}, Loss: {:.4f}'.format(epoch+1, running_loss / len(train_loader)))
    
    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            if inputs.shape[0] != batch_size:
                continue
            
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            val_loss += loss.item()
            
            # Convert sigmoid output to binary prediction
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Validation Loss: {:.4f}, Accuracy: {:.2f}%'.format(
        val_loss / len(val_loader), 100 * correct / total))
    
    # Switch back to training mode
    model.train()

# Test the model
model.eval()
test_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        if inputs.shape[0] != batch_size:
            continue
            
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels.float())
        test_loss += loss.item()
        
        # Convert sigmoid output to binary prediction
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Test Loss: {:.4f}, Accuracy: {:.2f}%'.format(
    test_loss / len(test_loader), 100 * correct / total))

# Save the model and plot
torch.save(model.state_dict(), '/home/trax/Projects/Brisingr/wake-word/model-convolution.ckpt')

# load model
# model = torch.load('wake-word/model-convolution-{len_mfcc}.ckpt')

# infer on a random input
x = torch.randn(1, 1, 16, 16, requires_grad=True)
torch_out = model(x)

model.eval()

model_trace_script = torch.jit.trace(model, x)
model_for_android = optimize_for_mobile(model_trace_script)
model_for_android._save_for_lite_interpreter('/home/trax/Projects/Brisingr/wake-word/model-convolution-'+str(x_train.shape[1])+'-'+str(x_train.shape[2]) + '-android_lite_model.pt1')




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
        loss = criterion(outputs, labels.float())
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
    # print("inputs")
    # print(inputs.shape)
    inputs = inputs.unsqueeze(1)
    inputs = inputs.type(torch.FloatTensor)
    inputs, labels = Variable(inputs), Variable(labels)
    # print("in the test loop")
    # print(inputs.shape)
    outputs = simple_model(inputs)
    # print("test output")
    # print(outputs.shape)
    # print(outputs.data)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)

    correct += (predicted == labels.long()).sum()

print("Correct: ", correct)
print("Total: ", total)
print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

# Save the model and plot
torch.save(simple_model.state_dict(), '/home/trax/Projects/Brisingr/wake-word/simple-model-{len_mfcc}.ckpt')


