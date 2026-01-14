# Wake Word Training UI - User Guide

A graphical user interface for training wake word detection models with easy dataset selection and real-time progress monitoring.

## Features

- 🎯 **Easy Dataset Selection**: Browse for training data files or select positive/negative sample directories
- 🔧 **Configurable Training**: Adjust epochs, batch size, learning rate, and model architecture
- 📊 **Real-time Progress**: Live training progress bar and log output
- 📈 **Results Dashboard**: View training metrics and test accuracy
- 💾 **Automatic Saving**: Best model automatically saved during training
- 🎨 **User-Friendly**: Clean tabbed interface with validation and error handling

## Installation

### 1. Ensure Dependencies are Installed

```bash
cd wake-word
source ../venv/bin/activate  # Or activate your venv

# Install required packages
pip install torch torchvision numpy scikit-learn matplotlib
```

### 2. Verify Training Script Exists

The UI depends on `training/training-v3.py`:

```bash
ls training/training-v3.py
```

If it doesn't exist, the UI will use built-in minimal model definitions.

## Usage

### Starting the Training UI

```bash
cd wake-word
source ../venv/bin/activate  # Activate your venv
python training_ui.py
```

Or make it executable and run directly:

```bash
chmod +x training_ui.py
./training_ui.py
```

## UI Tabs

### 1. Configuration Tab

#### Dataset Configuration

**Option A: Use Existing .npz File**
- Click "Browse" next to "Training Data File (.npz)"
- Select your pre-processed training data file
- Default: `optimized_wake_word_data.npz`

**Option B: Select Individual Directories** (Future feature)
- Browse for "Positive Samples Dir" (wake word recordings)
- Browse for "Negative Samples Dir" (background noise, other words)
- The UI will process these into MFCC features

#### Model Configuration

Choose from available model architectures:

| Model | Parameters | Speed | Accuracy | Best For |
|-------|-----------|-------|----------|----------|
| **TinyWakeWordCNN** | ~10K | Medium | Good | General use, balanced |
| **UltraTinyWakeWordCNN** | ~2-3K | Fast | Good | Mobile, battery-critical |
| **MobileNetStyleWakeWord** | ~4K | Fast | Good | Mobile optimization |
| **QuantizedFriendlyWakeWord** | ~5K | Medium | Very Good | Quantization target |

**Recommended**: UltraTinyWakeWordCNN (Efficient) for best balance of size and accuracy.

#### Training Parameters

- **Epochs** (default: 50)
  - How many times to iterate through the training data
  - More epochs = better training but longer time
  - Typical range: 30-100

- **Batch Size** (default: 64)
  - Number of samples processed together
  - Larger = faster training but more memory
  - Reduce if you get out-of-memory errors
  - Typical range: 32-128

- **Learning Rate** (default: 0.002)
  - How quickly the model learns
  - Too high = unstable training
  - Too low = slow learning
  - Typical range: 0.0001-0.01

- **Output Model Path**
  - Where to save the trained model
  - Default: `ultra_tiny_wake_word.pth`
  - Automatically saves best model during training

### 2. Training Tab

#### Real-time Monitoring

- **Progress Bar**: Visual indication of training progress (0-100%)
- **Status**: Current epoch, loss, and validation accuracy
- **Training Log**: Detailed output of training process

#### Controls

- **Start Training**: Begin training with current configuration
- **Stop Training**: Safely stop training (saves progress)
- **Clear Log**: Clear the log output

#### During Training

The log displays:
```
Loading data from: optimized_wake_word_data.npz
Train: (1000, 101, 13), Val: (200, 101, 13), Test: (200, 101, 13)
Creating model: UltraTinyWakeWordCNN (Efficient)
Model parameters: 2,618
Starting training loop...
Epoch 5/50: Loss: 0.3421, Val Acc: 87.50%
Epoch 10/50: Loss: 0.2156, Val Acc: 92.00%
...
```

### 3. Results Tab

After training completes, view:
- Model architecture and parameter count
- Training configuration used
- Best validation accuracy achieved
- Final test accuracy
- Training loss history
- Model save location

## Preparing Your Data

### Method 1: Using Existing .npz File

If you already have a `.npz` file with training data:

```python
import numpy as np

# Your file should contain these arrays:
data = np.load('your_data.npz')
# Required keys:
# - 'x_train': Training features (N, 101, 13)
# - 'y_train': Training labels (N,) - 0 or 1
# - 'x_val': Validation features
# - 'y_val': Validation labels
# - 'x_test': Test features
# - 'y_test': Test labels
```

### Method 2: Creating Dataset from Audio Files

If you have separate directories of positive and negative samples:

```bash
wake-word/
  ├── positive_samples/
  │   ├── wake_word_001.wav
  │   ├── wake_word_002.wav
  │   └── ...
  └── negative_samples/
      ├── background_001.wav
      ├── other_words_001.wav
      └── ...
```

**Data preprocessing script** (create your own or use the UI's future feature):

```python
import librosa
import numpy as np
from pathlib import Path

def load_audio_and_extract_mfcc(file_path, sr=16000):
    """Load audio and extract MFCC features"""
    audio, _ = librosa.load(file_path, sr=sr, duration=1.0)

    # Ensure 1 second
    if len(audio) < sr:
        audio = np.pad(audio, (0, sr - len(audio)))
    else:
        audio = audio[:sr]

    # Pre-emphasis
    audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

    # Extract MFCCs
    mfccs = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=13, n_fft=512, hop_length=160,
        n_mels=26, fmin=80, fmax=8000
    )

    return mfccs.T  # (time, features)

# Load positive samples
positive_dir = Path("positive_samples")
positive_features = []
for audio_file in positive_dir.glob("*.wav"):
    mfcc = load_audio_and_extract_mfcc(audio_file)
    positive_features.append(mfcc)

# Load negative samples
negative_dir = Path("negative_samples")
negative_features = []
for audio_file in negative_dir.glob("*.wav"):
    mfcc = load_audio_and_extract_mfcc(audio_file)
    negative_features.append(mfcc)

# Combine and create labels
X = np.array(positive_features + negative_features)
y = np.array([1] * len(positive_features) + [0] * len(negative_features))

# Split into train/val/test (70/15/15)
from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Save
np.savez('my_wake_word_data.npz',
         x_train=X_train, y_train=y_train,
         x_val=X_val, y_val=y_val,
         x_test=X_test, y_test=y_test)

print(f"Dataset created: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test")
```

## Training Tips

### Getting Good Results

1. **Dataset Quality**
   - Need 100+ positive samples (wake word recordings)
   - Need 200+ negative samples (background noise, other words)
   - Record in various environments and with different speakers
   - Ensure 1-second duration for all samples

2. **Balanced Dataset**
   - Aim for 1:2 or 1:3 ratio (positive:negative)
   - More negative samples = fewer false positives
   - The UI automatically weights classes

3. **Training Configuration**
   - Start with defaults (50 epochs, batch size 64)
   - If validation accuracy plateaus, try more epochs
   - If training is slow, increase batch size
   - If overfitting (val acc drops), reduce epochs or add more data

4. **Model Selection**
   - **UltraTinyWakeWordCNN**: Best for mobile/embedded
   - **TinyWakeWordCNN**: More capacity, slightly better accuracy
   - **QuantizedFriendlyWakeWord**: If you plan to quantize

### Monitoring Training

Good training looks like:
```
Epoch 5:  Loss: 0.45, Val Acc: 75%
Epoch 10: Loss: 0.32, Val Acc: 85%
Epoch 20: Loss: 0.21, Val Acc: 92%
Epoch 30: Loss: 0.15, Val Acc: 94%
Epoch 40: Loss: 0.12, Val Acc: 95%
Epoch 50: Loss: 0.10, Val Acc: 95%
```

Bad training (overfitting):
```
Epoch 30: Loss: 0.15, Val Acc: 94%
Epoch 40: Loss: 0.08, Val Acc: 93%  ← Val acc dropping!
Epoch 50: Loss: 0.05, Val Acc: 91%
```

If overfitting occurs:
- Stop training earlier
- Add more training data
- Add data augmentation

### Target Metrics

For good wake word detection:
- **Validation Accuracy**: >90%
- **Test Accuracy**: >88%
- **Gap**: <5% (train acc - val acc)

If test accuracy is significantly lower than validation:
- Need more diverse test data
- Model may be overfitting to validation set

## After Training

### 1. Export to ExecuTorch (for Android)

```bash
cd wake-word
source ../venv/bin/activate

# Edit export_executorch_optimized.py to point to your model
# Then run:
python export_executorch_optimized.py
```

This creates:
- `wake_word_xnnpack.pte` - Optimized for mobile
- `wake_word_quantized.pte` - Even smaller/faster

### 2. Copy to Android App

```bash
cp wake_word_xnnpack.pte ../app/apps/src/androidMain/assets/wake_word.pte
```

### 3. Test Inference

Test your model before deploying:

```bash
python inference/linux-inference-testing-v3.py
```

Speak your wake word and verify it detects correctly!

## Troubleshooting

### "Data file not found"
- Ensure the .npz file path is correct
- Check that the file exists and is readable

### "Out of memory error"
- Reduce batch size (try 32 or 16)
- Close other applications
- Use a smaller model (UltraTinyWakeWordCNN)

### "Training stuck at low accuracy"
- Check dataset quality (might have corrupted samples)
- Verify labels are correct (1 = wake word, 0 = other)
- Try different learning rate (0.001 or 0.005)
- Ensure sufficient training data (>100 samples per class)

### "Model not improving"
- Increase learning rate
- Train for more epochs
- Use larger model (TinyWakeWordCNN)
- Check for data imbalance

### "Cannot import training modules"
- Ensure `training/training-v3.py` exists
- Activate virtual environment first
- Check Python path includes wake-word directory

## Keyboard Shortcuts

- **Ctrl+L**: Clear log (when log is focused)
- **Alt+T**: Focus Training tab
- **Alt+C**: Focus Configuration tab
- **Alt+R**: Focus Results tab

## Example Workflow

1. **Prepare Data**
   ```bash
   # Record samples or use existing dataset
   python preprocess_audio.py  # Your preprocessing script
   ```

2. **Launch UI**
   ```bash
   python training_ui.py
   ```

3. **Configure**
   - Select data file: `my_wake_word_data.npz`
   - Choose model: UltraTinyWakeWordCNN
   - Set epochs: 50
   - Set output: `my_trained_model.pth`

4. **Train**
   - Click "Start Training"
   - Monitor progress in Training tab
   - Wait for completion (5-30 minutes depending on data size)

5. **Review Results**
   - Check Results tab for metrics
   - Verify test accuracy >90%

6. **Export**
   ```bash
   python export_executorch_optimized.py
   # Edit script to use 'my_trained_model.pth'
   ```

7. **Deploy**
   ```bash
   cp wake_word_xnnpack.pte ../app/apps/src/androidMain/assets/
   ```

## Advanced Features

### Custom Model Architectures

To add your own model:

1. Define model in `training/training-v3.py`:
   ```python
   class MyCustomModel(nn.Module):
       def __init__(self, input_dim=13, sequence_length=101):
           # Your architecture
   ```

2. Restart the UI - it will appear in the dropdown

### Training Callbacks

The UI saves the best model automatically based on validation accuracy. The model file includes:
- Model state dict
- Optimizer state
- Training metrics
- Model configuration

### Resume Training

To resume from a checkpoint:

1. Load the saved model
2. Modify training script to load optimizer state
3. Continue training from that epoch

## Tips for Mobile Deployment

1. **Use UltraTinyWakeWordCNN**: Smallest, fastest model
2. **Export with XNNPACK**: 2-5x speedup on mobile
3. **Consider quantization**: Another 2-3x speedup
4. **Test on device**: Always validate performance on target device
5. **Tune threshold**: Adjust detection threshold in Android app settings

## Support & Documentation

- Main optimization guide: `../WAKE_WORD_OPTIMIZATION_GUIDE.md`
- Battery guide: `../BATTERY_OPTIMIZATION_GUIDE.md`
- Training script: `training/training-v3.py`
- Export script: `export_executorch_optimized.py`

## License

Same as parent project.
