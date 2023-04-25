from os import listdir
from os.path import isdir, join
import os
import numpy as np
from torch.utils.mobile_optimizer import optimize_for_mobile

# Create list of all targets (minus background noise)
dataset_path = os.path.join(os.path.dirname(__file__),'training-data/')
all_targets = all_targets = [name for name in listdir(dataset_path) if isdir(join(dataset_path, name))]
# all_targets.remove('_background_noise_')
print(all_targets)


# Settings
feature_sets_filename = 'all_targets_mfcc_sets.npz'
model_filename = 'brisingr-30epochs-1s-no-copies-no-augmentation-removed-downloaded-backgroundnoise.h5'
wake_word = 'brisingr'


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
print("y_train.shape")
print(x_val.shape)
print("x_test.shape")
print(x_test.shape)

# Peek at labels
print("y_val")
print(y_val)

# Convert ground truth arrays to one wake word (1) and 'other' (0)
wake_word_index = all_targets.index(wake_word)
y_train = np.equal(y_train, wake_word_index).astype('float64')
y_val = np.equal(y_val, wake_word_index).astype('float64')
y_test = np.equal(y_test, wake_word_index).astype('float64')

# What percentage of 'stop' appear in validation labels
print(sum(y_val) / len(y_val))
print(1 - sum(y_val) / len(y_val))
# View the dimensions of our input data
print(x_train.shape)





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

# batch_size = 3
batch_size = 21

train_loader = get_dataloader(x_train, y_train, batch_size)
test_loader = get_dataloader(x_test, y_test, batch_size)

input_size = x_train.shape[1]
print("c16")

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
    
model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.NLLLoss()

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

# Test the model
model.eval()
correct = 0
total = 0
for data in test_loader:
    inputs, labels = data
    print("inputs")
    print(inputs)
    inputs = inputs.unsqueeze(1)
    inputs = inputs.type(torch.FloatTensor)
    inputs, labels = Variable(inputs), Variable(labels)
    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels.long()).sum()

print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

# Save the model and plot
torch.save(model.state_dict(), 'model.ckpt')
torch.save(model, 'model.pth')

# load model
model = torch.load('model.pth')
model.eval()

# infer on a random input
x = torch.randn(1, 1, 16, 16, requires_grad=True)
torch_out = model(x)

print("stuff")
print(torch_out)

print("test")
print (torch.max(torch_out,1))


# save model so it can be ran on a android device
torch.save(model, 'wake-word/model.pth')


model.eval()

model_trace_script = torch.jit.trace(model, x)
model_for_android = optimize_for_mobile(model_trace_script)
model_for_android._save_for_lite_interpreter("wake-word/android_lite_model.pt1")




# exit script
exit()




import torch.utils.mobile_optimizer as mobile_optimizer


from torch.utils.mobile_optimizer import optimize_for_mobile
def optimizeSave():
    # Load in the model
    model = Model()
    model.load_state_dict(torch.load("model.pkl", \
        map_location=torch.device("cpu")))
    model.eval() # Put the model in inference mode
    
    # Generate some random noise
    X = torch.distributions.uniform.Uniform(-10000, \
        10000).sample((4, 2))
    
    # Generate the optimized model
    traced_script_module = torch.jit.trace(model, X)
    traced_script_module_optimized = optimize_for_mobile(\
        traced_script_module)
    
    # Save the optimzied model
    traced_script_module_optimized._save_for_lite_interpreter(\
        "model.pt")

#######################################











# pytorch convulutional neural network

second_modle_save_name = 'second_model.pt'
learning_rate = 0.001
num_epochs = 10

class ConvNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential( # 16, 8, 8
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2), # 32, 8, 8
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)) # 32, 4, 4
        self.fc = nn.Linear(32*4*4, num_classes) # 512, 2
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out) # 32, 4, 4
        out = out.reshape(out.size(0), -1) # 512
        out = self.fc(out)
        return out
    
model = ConvNet(num_classes=2).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(x_train)
for epoch in range(num_epochs):
    for i in range(len(x_train)):
        images = x_train[i].to(device)
        labels = y_train[i].to(device)
        
        # Forward pass
        outputs = model(images.unsqueeze(0))
        loss = criterion(outputs, labels.long())
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            
# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = len(x_val)
    for i in range(len(x_val)):
        images = x_val[i].to(device)
        labels = y_val[i].to(device)
        outputs = model(images.unsqueeze(0))
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels.long()).sum().item()
    print('Accuracy of the network on the {} test images: {} %'.format(total, 100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), second_modle_save_name+'.ckpt')

# Load the model checkpoint
model = ConvNet(num_classes=2).to(device)
model.load_state_dict(torch.load(second_modle_save_name+'.ckpt'))

# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = len(x_test)
    for i in range(len(x_test)):
        images = x_test[i].to(device)
        labels = y_test[i].to(device)
        outputs = model(images.unsqueeze(0))
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels.long()).sum().item()
    print('Accuracy of the network on the {} test images: {} %'.format(total, 100 * correct / total))




