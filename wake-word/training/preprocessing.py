from os import listdir
import os
from os.path import isdir, join
import librosa
import random
import numpy as np
import matplotlib 
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import python_speech_features

# Original Dataset path and view possible targets 
# dataset_path = os.path.join(os.path.dirname(__file__),'training-data/')
dataset_path = "/media/trax/Elements/machine-learning/training-data/Brisingr wakeword/"
for name in listdir(dataset_path):
    if isdir(join(dataset_path, name)):
        print(name)



# Create an all targets list
all_targets = [name for name in listdir(dataset_path) if isdir(join(dataset_path, name))]
print(all_targets)


num_samples = 0
for target in all_targets:
    print(len(listdir(join(dataset_path, target))))
    num_samples += len(listdir(join(dataset_path, target)))
print('Total samples:', num_samples)



# Settings original
target_list = all_targets
feature_sets_file = 'wake-word/training/all_targets_mfcc_sets.npz'
perc_keep_samples = 1 # 1.0 is keep all samples
val_ratio = 0.1
test_ratio = 0.1
seconds = 1
sample_rate = 8000
num_mfcc = 16 * seconds
len_mfcc = 16 * seconds


# Create list of filenames along with ground truth vector (y)
filenames = []
y = []
for index, target in enumerate(target_list):
    print(join(dataset_path, target))
    filenames.append(listdir(join(dataset_path, target)))
    y.append(np.ones(len(filenames[index])) * index)




# Check ground truth Y vector
print(y)
for item in y:
    print(len(item))


# Flatten filename and y vectors
filenames = [item for sublist in filenames for item in sublist]
y = [item for sublist in y for item in sublist]


# Associate filenames with true output and shuffle
filenames_y = list(zip(filenames, y))
random.shuffle(filenames_y)
filenames, y = zip(*filenames_y)


# Only keep the specified number of samples (shorter extraction/training)
print(len(filenames))
filenames = filenames[:int(len(filenames) * perc_keep_samples)]
print(len(filenames))


# Calculate validation and test set sizes
val_set_size = int(len(filenames) * val_ratio)
test_set_size = int(len(filenames) * test_ratio)


# Break dataset apart into train, validation, and test sets
filenames_val = filenames[:val_set_size]
filenames_test = filenames[val_set_size:(val_set_size + test_set_size)]
filenames_train = filenames[(val_set_size + test_set_size):]



# Break y apart into train, validation, and test sets
y_orig_val = y[:val_set_size]
y_orig_test = y[val_set_size:(val_set_size + test_set_size)]
y_orig_train = y[(val_set_size + test_set_size):]



def calc_mfcc(path):
    try:
        # Load wavefile
        signal, fs = librosa.load(path, sr=sample_rate)
        
        # Check if signal is empty
        if len(signal) == 0:
            print(f"Warning: Empty audio file: {path}")
            return None
            
        # Create MFCCs from sound clip
        mfccs = python_speech_features.base.mfcc(signal, 
                                                samplerate=fs,
                                                winlen=0.256,
                                                winstep=0.050,
                                                numcep=num_mfcc,
                                                nfilt=26,
                                                nfft=2048,
                                                preemph=0.0,
                                                ceplifter=0,
                                                appendEnergy=False,
                                                winfunc=np.hanning)
        return mfccs.transpose()
    except Exception as e:
        print(f"Error processing {path}: {str(e)}")
        return None



# TEST: Construct test set by computing MFCC of each WAV file
prob_cnt = 0
x_test = []
y_test = []
for index, filename in enumerate(filenames_train):
    
    # Stop after 500
    if index >= 500:
        break
    
    # Create path from given filename and target item
    path = join(dataset_path, target_list[int(y_orig_train[index])], 
                filename)
    
    # Create MFCCs
    mfccs = calc_mfcc(path)
    if mfccs is not None:
        if mfccs.shape[1] == len_mfcc:
            x_test.append(mfccs)
            y_test.append(y_orig_train[index])
        else:
            print('Dropped:', index, mfccs.shape)
            prob_cnt += 1
    else:
        print('Dropped:', index, 'Empty or invalid file')
        prob_cnt += 1


print('% of problematic samples:', prob_cnt / 500)




# TEST: Test shorter MFCC (commented out for non-GUI environment)
# !pip install playsound
# from playsound import playsound

idx = 4
path = join(dataset_path, target_list[int(y_orig_train[idx])], 
            filenames_train[idx])
mfccs = calc_mfcc(path)
# fig = plt.figure(figsize=(10, 4))
# plt.imshow(mfccs, cmap='inferno', origin='lower')
# plt.colorbar()
# plt.title('MFCC')
# plt.tight_layout()
# plt.show()
print(target_list[int(y_orig_train[idx])])
# playsound(path)



# Function: Create MFCCs, keeping only ones of desired length
def extract_features(in_files, in_y):
    prob_cnt = 0
    out_x = []
    out_y = []
    for index, filename in enumerate(in_files):
        
        # Create path from given filename and target item
        path = join(dataset_path, target_list[int(in_y[index])], 
                    filename)
        
        # Create MFCCs
        mfccs = calc_mfcc(path)
        if mfccs is not None:
            if mfccs.shape[1] >= len_mfcc:
                # If MFCC is larger than len_mfcc, truncate it
                if mfccs.shape[1] > len_mfcc:
                    mfccs = mfccs[:, :len_mfcc]
                out_x.append(mfccs)
                out_y.append(in_y[index])
            else:
                print('Dropped:', index, mfccs.shape)
                prob_cnt += 1
        else:
            print('Dropped:', index, 'Empty or invalid file')
            prob_cnt += 1
    
    return out_x, out_y, prob_cnt



# Create train, validation, and test sets
x_train, y_train, prob = extract_features(filenames_train, 
                                          y_orig_train)
print('Removed percentage:', prob / len(y_orig_train))
x_val, y_val, prob = extract_features(filenames_val, y_orig_val)
print('Removed percentage:', prob / len(y_orig_val))
x_test, y_test, prob = extract_features(filenames_test, y_orig_test)
print('Removed percentage:', prob / len(y_orig_test))


# Save features and truth vector (y) sets to disk
np.savez(feature_sets_file, 
         x_train=x_train, 
         y_train=y_train, 
         x_val=x_val, 
         y_val=y_val, 
         x_test=x_test, 
         y_test=y_test)


# TEST: Load features
feature_sets = np.load(feature_sets_file)
feature_sets.files

# positive wake word amount
positive_amount = len(feature_sets['y_train'])
negative_amount = len(feature_sets['y_val'])
print("positive wake word amount")
print(positive_amount)
print("negative wake word amount")
print(negative_amount)


print("len fefature set")

print(len(feature_sets['x_train']))

print("len y_train")
print(len(feature_sets['y_train']))