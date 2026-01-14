# Wake Word Detection - Training & Inference

Complete toolkit for training and deploying wake word detection models for Android.

## Quick Start

### Training with GUI

```bash
cd wake-word
./launch_training_ui.sh
```

Or manually:

```bash
source ../venv/bin/activate
python training_ui.py
```

### Command-Line Training

```bash
cd training
python training-v3.py
```

## Project Structure

```
wake-word/
├── training_ui.py              # 🎨 Graphical training interface
├── launch_training_ui.sh       # 🚀 Quick launcher script
├── TRAINING_UI_GUIDE.md        # 📖 Complete UI documentation
│
├── training/
│   ├── training-v3.py         # Core training script
│   ├── training-v2.py         # Previous version
│   └── training.py            # Original version
│
├── inference/
│   ├── linux-inference-testing-v3.py  # Desktop inference testing
│   ├── linux-inference-testing-v2.py
│   └── linux-inference-testing.py
│
├── export_executorch.py        # Basic ExecuTorch export
├── export_executorch_optimized.py  # XNNPACK + quantization export
│
└── skills/                     # Development tools
    └── assistant_data_collector.py
```

## Features

### Training UI Features
- ✅ Browse for training data files (.npz format)
- ✅ Select positive/negative sample directories
- ✅ Choose from multiple model architectures
- ✅ Adjust training parameters (epochs, batch size, learning rate)
- ✅ Real-time training progress monitoring
- ✅ Automatic best model saving
- ✅ Results dashboard with metrics
- ✅ Multi-threaded training (UI stays responsive)

### Available Models

| Model | Size | Speed | Accuracy | Best For |
|-------|------|-------|----------|----------|
| TinyWakeWordCNN | 10KB | Medium | Good | General use |
| UltraTinyWakeWordCNN | 3KB | Fast | Good | Mobile/embedded |
| MobileNetStyleWakeWord | 5KB | Fast | Good | Mobile optimization |
| QuantizedFriendlyWakeWord | 6KB | Medium | Very Good | Quantization target |

## Workflows

### 1. Training a New Model

#### Using the GUI (Recommended)

```bash
# 1. Launch UI
./launch_training_ui.sh

# 2. In the Configuration tab:
#    - Select your .npz data file
#    - Choose model architecture (UltraTinyWakeWordCNN recommended)
#    - Set training parameters
#    - Specify output path

# 3. Switch to Training tab and click "Start Training"

# 4. Monitor progress and view results when complete
```

#### Using Command Line

```bash
cd training
python training-v3.py

# Edit the script to customize:
# - Model architecture
# - Epochs, batch size, learning rate
# - Data file path
```

### 2. Exporting for Android

After training:

```bash
# 1. Export to ExecuTorch with optimizations
python export_executorch_optimized.py

# This creates:
# - wake_word_xnnpack.pte (XNNPACK optimized)
# - wake_word_quantized.pte (INT8 quantized)

# 2. Copy to Android assets
cp wake_word_xnnpack.pte ../app/apps/src/androidMain/assets/wake_word.pte

# 3. Rebuild Android app
cd ../app
./gradlew assembleDebug
```

### 3. Testing Inference (Desktop)

```bash
cd inference
python linux-inference-testing-v3.py

# Speak your wake word to test detection
```

## Data Preparation

### Required Format

Training data should be a `.npz` file containing:

```python
{
    'x_train': (N_train, 101, 13),  # MFCC features
    'y_train': (N_train,),           # Labels (0 or 1)
    'x_val': (N_val, 101, 13),
    'y_val': (N_val,),
    'x_test': (N_test, 101, 13),
    'y_test': (N_test,)
}
```

- **Shape**: (num_samples, 101 frames, 13 MFCC coefficients)
- **Labels**: 0 = negative (not wake word), 1 = positive (wake word)
- **Split**: 70% train, 15% validation, 15% test

### Creating Dataset from Audio

See `TRAINING_UI_GUIDE.md` for complete data preprocessing example.

Quick version:

```python
import librosa
import numpy as np

# Load 1-second audio clip
audio, sr = librosa.load('audio.wav', sr=16000, duration=1.0)

# Pre-emphasis
audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

# Extract MFCCs (matches Android implementation)
mfccs = librosa.feature.mfcc(
    y=audio, sr=16000, n_mfcc=13,
    n_fft=512, hop_length=160,
    n_mels=26, fmin=80, fmax=8000
)

features = mfccs.T  # Shape: (101, 13)
```

## Training Tips

### Dataset Requirements

**Minimum:**
- 100+ positive samples (wake word recordings)
- 200+ negative samples (background noise, other words)

**Recommended:**
- 500+ positive samples
- 1000+ negative samples
- Multiple speakers
- Various environments (quiet, noisy, outdoor)
- Different recording devices

### Good Training Metrics

- **Validation Accuracy**: >90%
- **Test Accuracy**: >88%
- **Train-Val Gap**: <5%

If metrics are lower:
- Add more training data
- Train for more epochs
- Try different model architecture
- Check data quality

## Optimization Guides

- **Training UI Guide**: `TRAINING_UI_GUIDE.md` - Complete GUI documentation
- **Model Optimization**: `../WAKE_WORD_OPTIMIZATION_GUIDE.md` - Android optimization
- **Battery Optimization**: `../BATTERY_OPTIMIZATION_GUIDE.md` - Power efficiency

## Troubleshooting

### Training UI won't start

```bash
# Check Python version (3.8+)
python --version

# Install tkinter if missing (Ubuntu/Debian)
sudo apt-get install python3-tk

# Install dependencies
pip install torch numpy scikit-learn matplotlib
```

### ImportError: No module named 'training.training_v3'

The UI will use built-in minimal models if the import fails. This is normal and won't affect functionality.

### Out of memory during training

- Reduce batch size (try 32 or 16)
- Use smaller model (UltraTinyWakeWordCNN)
- Close other applications
- Use GPU if available

### Model not detecting on Android

1. **Check model export**: Ensure using XNNPACK export
2. **Verify MFCC parameters match**: fmin=80, fmax=8000, n_fft=512, hop_length=160
3. **Test on desktop first**: Use `linux-inference-testing-v3.py`
4. **Check Android logs**: Enable debug mode in WakeWordState
5. **Adjust threshold**: Try lowering to 0.5-0.6

## Advanced Usage

### Custom Model Architecture

1. Add your model to `training/training-v3.py`:

```python
class MyCustomModel(nn.Module):
    def __init__(self, input_dim=13, sequence_length=101):
        super().__init__()
        # Your architecture here

    def forward(self, x):
        # Your forward pass
        return F.log_softmax(output, dim=1)  # Must return log_softmax!
```

2. Restart the UI - it will appear in the model dropdown

### Hyperparameter Tuning

Try different combinations:

**Fast Training (Quick Test)**
- Epochs: 20
- Batch Size: 128
- Learning Rate: 0.005

**High Accuracy (Production)**
- Epochs: 100
- Batch Size: 64
- Learning Rate: 0.001

**Small Dataset (<500 samples)**
- Epochs: 30
- Batch Size: 32
- Learning Rate: 0.002
- Model: UltraTinyWakeWordCNN

## Performance Benchmarks

### Training Time (CPU)
- 1000 samples, 50 epochs: ~5-10 minutes
- 5000 samples, 50 epochs: ~20-30 minutes

### Training Time (GPU)
- 1000 samples, 50 epochs: ~2-3 minutes
- 5000 samples, 50 epochs: ~8-12 minutes

### Model Inference (Android)
- TinyWakeWordCNN: ~8-10ms per inference
- UltraTinyWakeWordCNN: ~4-6ms per inference
- With XNNPACK: ~2-3ms per inference
- Quantized: ~1-2ms per inference

## Dependencies

```bash
# Core
torch>=2.0.0
numpy>=1.20.0
scikit-learn>=1.0.0

# For training UI
tkinter (usually built-in)

# For data preprocessing
librosa>=0.9.0
soundfile>=0.10.0

# For export
executorch>=0.1.0
```

## Contributing

To add new features:

1. **New Model**: Add to `training/training-v3.py`
2. **UI Enhancement**: Modify `training_ui.py`
3. **Export Option**: Update `export_executorch_optimized.py`

## License

Same as parent Brisingr project.

## Quick Reference

```bash
# Launch training UI
./launch_training_ui.sh

# Train via command line
cd training && python training-v3.py

# Test inference
cd inference && python linux-inference-testing-v3.py

# Export for Android
python export_executorch_optimized.py

# Deploy to Android
cp wake_word_xnnpack.pte ../app/apps/src/androidMain/assets/wake_word.pte
```

## Support

For issues or questions:
1. Check `TRAINING_UI_GUIDE.md` for detailed documentation
2. Review optimization guides in parent directory
3. Enable debug logging for diagnostics
4. Check training logs for errors

Happy training! 🎯
