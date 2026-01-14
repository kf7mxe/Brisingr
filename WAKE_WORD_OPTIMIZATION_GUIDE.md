# Wake Word Detection Optimization Guide

## Recent Fixes Applied

### 1. Performance Improvements
- ✅ **Replaced naive DFT with FFT** (100x+ faster)
  - Before: O(n²) complexity using nested loops
  - After: O(n log n) using Cooley-Tukey FFT algorithm
  - Expected CPU reduction: 35% → 5-10%

- ✅ **Fixed MFCC parameters** to match desktop
  - `fmin`: 0 Hz → 80 Hz
  - `fmax`: 16000 Hz → 8000 Hz
  - This ensures features match training data

- ✅ **Fixed softmax handling**
  - Model outputs `log_softmax`, not logits
  - Now correctly applying `exp()` instead of double softmax

- ✅ **Added early volume filtering**
  - Skips expensive FFT for silent audio
  - Only checks new audio samples for faster RMS calculation

### 2. Testing Controls Added

You can now adjust these settings in `WakeWordState`:

```kotlin
// In your UI or test code:
WakeWordState.threshold.value = 0.5f           // Lower = more sensitive (0.0-1.0)
WakeWordState.cooldownSeconds.value = 1.0f     // Time between detections
WakeWordState.minVolumeThreshold.value = 0.0005f  // Lower = processes quieter audio
WakeWordState.processingIntervalMs.value = 50L    // Lower = more CPU, more responsive
WakeWordState.debugMode.value = true              // Enable debug logging
```

### 3. What Each Setting Means

**Threshold (default 0.65)**
- The minimum confidence score to trigger detection
- Lower = more false positives, catches wake word more easily
- Higher = fewer false positives, but might miss some wake words
- Desktop uses 0.65-0.70

**Cooldown Seconds (default 2.0)**
- Prevents the wake word from triggering multiple times in rapid succession
- After detection, waits this many seconds before allowing another detection
- For testing, set to 0.5 or 1.0 to trigger faster

**Min Volume Threshold (default 0.001)**
- Audio below this RMS volume is ignored (saves CPU)
- Lower = processes quieter audio (more CPU)
- Higher = only processes louder audio (less CPU)
- If wake word isn't detected when speaking quietly, lower this

**Processing Interval (default 50ms)**
- How often the detector processes audio
- Lower = more CPU, more responsive, better accuracy
- Higher = less CPU, slightly less responsive
- Desktop uses ~50ms

**Debug Mode (default false)**
- Enables detailed logging to logcat
- Shows MFCC frame counts, probabilities, volume levels
- Use `adb logcat | grep WakeWord` to see debug output

## Remaining Accuracy Issues

The Android implementation still may not match desktop accuracy. Here's why:

### Audio Buffering Difference

**Desktop (librosa):**
- Processes exactly 1 second (16000 samples)
- Uses `hop_length=160` which produces exactly 101 frames
- Each frame is computed consistently

**Android (current):**
- Uses sliding window with 0.5 second overlap
- Frame calculation might not align exactly with librosa
- May produce 99-100 frames instead of 101 (gets padded)

### Potential Fixes

1. **Match librosa's framing exactly** - Update MFCCExtractor to match librosa's window/hop strategy
2. **Ensure 101 frames always** - Current padding helps but might introduce artifacts

## GPU/NPU Acceleration Options

### Option 1: XNNPACK Backend (Recommended)

I've created `export_executorch_optimized.py` which exports with XNNPACK support:

```bash
cd wake-word
source ../venv/bin/activate  # Or activate your venv
pip install executorch
python export_executorch_optimized.py
```

This creates `wake_word_xnnpack.pte` which:
- Uses CPU SIMD instructions (ARM NEON on Android)
- Can leverage NPU/DSP on Snapdragon 8 Gen 2/3 and newer
- ~2-5x faster than basic ExecuTorch

**To use it:**
1. Copy `wake_word_xnnpack.pte` to `app/apps/src/androidMain/assets/`
2. Update model name in WakeWordDetector if needed
3. Rebuild and run

### Option 2: QNN Backend (Qualcomm NPU)

For Snapdragon devices with dedicated NPU:
- Requires Qualcomm's QNN SDK
- Much more complex setup
- Can achieve 10-20x speedup
- Only for production apps

### Option 3: CoreML Backend (iOS only)

For iOS devices:
- Uses Apple Neural Engine
- Would require separate export script
- Very efficient on iPhone

## Do You Need to Retrain?

**No, you don't need to retrain!** But you might want to re-export:

1. **Re-export with XNNPACK** ✅ (recommended)
   - Same model, better runtime
   - Run `export_executorch_optimized.py`

2. **Re-export with quantization** (optional)
   - Smaller model, faster inference
   - Slight accuracy loss (~1-2%)
   - Need to add quantization to export script

3. **Retrain completely** (only if accuracy is still poor)
   - Only needed if fundamental model architecture needs change
   - Current model should work fine

## Recommended Testing Procedure

1. **Enable debug mode:**
   ```kotlin
   WakeWordState.debugMode.value = true
   ```

2. **Run with default settings first:**
   - Threshold: 0.65
   - Cooldown: 2.0s
   - Processing interval: 50ms

3. **If not detecting:**
   - Lower threshold to 0.5
   - Check debug logs for probability values
   - If probabilities are very low (< 0.1), there's a feature extraction issue
   - If probabilities are moderate (0.4-0.6), just lower threshold

4. **If too many false positives:**
   - Raise threshold to 0.75
   - Increase cooldown to 3.0s

5. **If CPU is still too high:**
   - Increase processing interval to 100ms
   - Raise min volume threshold to 0.002
   - Use XNNPACK export

## Next Steps

1. **Try XNNPACK export** (highest priority)
   ```bash
   cd wake-word
   source ../venv/bin/activate
   python export_executorch_optimized.py
   cp wake_word_xnnpack.pte ../app/apps/src/androidMain/assets/wake_word.pte
   ```

2. **Enable debug mode and test**
   - Watch logcat output
   - Note probability values when saying wake word
   - Adjust threshold accordingly

3. **Profile CPU usage**
   - Before: ~35%
   - After FFT fix: should be ~10-15%
   - After XNNPACK: should be ~5%

4. **Compare MFCC output** (if accuracy still poor)
   - Log MFCC frame count on Android
   - Should be 101 frames consistently
   - If not, we need to fix the framing logic

## Example Test Code

Add this to your UI for easy testing:

```kotlin
// Testing controls
Button(onClick = {
    WakeWordState.threshold.value = 0.5f  // More sensitive
    WakeWordState.debugMode.value = true
}) {
    Text("Enable Debug + Lower Threshold")
}

Button(onClick = {
    WakeWordState.cooldownSeconds.value = 0.5f
}) {
    Text("Fast Cooldown (0.5s)")
}

Text("Current Probability: ${WakeWordState.probability.value}")
Text("Threshold: ${WakeWordState.threshold.value}")
```

## Common Issues

**"Wake word not detected at all"**
- Check debug logs - are probabilities ever > 0.1?
- If yes: lower threshold
- If no: MFCC extraction issue, check frame count

**"False positives on background noise"**
- Raise threshold to 0.75-0.80
- Increase min volume threshold

**"High CPU usage"**
- Use XNNPACK export
- Increase processing interval to 100ms
- Raise min volume threshold (skips more silent audio)

**"Works on desktop but not Android"**
- Most likely MFCC framing difference
- Check debug logs for frame count (should be 101)
- May need to exactly replicate librosa's framing logic
