package com.kf7mxe.brisingr.wakeword

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import androidx.core.content.ContextCompat
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.withContext
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import java.io.File
import java.io.FileOutputStream
import kotlin.math.sqrt
import kotlin.time.Clock
import kotlin.time.Instant

/**
 * Wake word detection engine for Android.
 * Handles audio recording, MFCC extraction, and ExecuTorch inference.
 * 
 * Based on linux-inference-testing-v3.py implementation.
 */
class WakeWordDetector(private val context: Context) {
    
    companion object {
        // Audio settings
        private const val SAMPLE_RATE = 16000
        private const val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
        private const val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_FLOAT
        private const val BUFFER_SECONDS = 1
        private const val BUFFER_SIZE = SAMPLE_RATE * BUFFER_SECONDS
        // Use smaller overlap for lower latency (1600 samples = 100ms at 16kHz)
        // Linux uses chunk_size=1024 (~64ms), but we need slightly more for stable processing
        private const val OVERLAP_SIZE = 1600

        // MFCC settings (matching Python/librosa implementation)
        private const val NUM_MFCC = 13
        private const val N_FILT = 26
        private const val NFFT = 512
        private const val HOP_LENGTH = 160
        private const val LOW_FREQ = 80
        private const val HIGH_FREQ = 8000

        // Detection settings
        private const val SMOOTHING_WINDOW = 5

        // Model file name - update this to match your exported model
        private const val MODEL_NAME = "wake_word_xnnpack.pte"

        // Set to true if your model outputs log_softmax, false for raw logits
        // The export script will tell you which one your model uses
        private const val MODEL_OUTPUTS_LOG_SOFTMAX = true
    }

    // MFCC extractor (matches librosa implementation)
    private val mfccExtractor = MFCCExtractor(
        sampleRate = SAMPLE_RATE,
        numCep = NUM_MFCC,
        nFilt = N_FILT,
        nfft = NFFT,
        lowFreq = LOW_FREQ,
        highFreq = HIGH_FREQ,
        hopLength = HOP_LENGTH
    )
    private var audioRecord: AudioRecord? = null
    private var module: Module? = null
    private var isRunning = false
    
    // Audio buffer (1 second)
    private val audioBuffer = FloatArray(BUFFER_SIZE)
    
    // Detection history for smoothing
    private val detectionHistory = ArrayDeque<Float>(SMOOTHING_WINDOW)

    private var lastDetectionTime = 0L
    private var consecutiveLowProbability = 0
    
    /**
     * Load the ExecuTorch model from assets.
     */
    private fun loadModel() {
        if (module != null) return
        
        try {
            val modelPath = assetFilePath(MODEL_NAME)
            module = Module.load(modelPath)
        } catch (e: Exception) {
            e.printStackTrace()
            WakeWordState.errorMessage.value = "Failed to load model: ${e.message}"
        }
    }
    
    /**
     * Copy asset to internal storage and return path.
     */
    private fun assetFilePath(assetName: String): String {
        val file = File(context.filesDir, assetName)
        if (file.exists() && file.length() > 0) {
            return file.absolutePath
        }
        
        context.assets.open(assetName).use { inputStream ->
            FileOutputStream(file).use { outputStream ->
                val buffer = ByteArray(4 * 1024)
                var read: Int
                while (inputStream.read(buffer).also { read = it } != -1) {
                    outputStream.write(buffer, 0, read)
                }
                outputStream.flush()
            }
        }
        return file.absolutePath
    }
    
    /**
     * Start continuous wake word detection.
     */
    suspend fun startDetection() = withContext(Dispatchers.Default) {
        if (isRunning) return@withContext
        
        // Check permission
        if (ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) 
            != PackageManager.PERMISSION_GRANTED) {
            WakeWordState.errorMessage.value = "Microphone permission required"
            return@withContext
        }
        
        try {
            loadModel()
            if (module == null) {
                WakeWordState.errorMessage.value = "Model not loaded"
                return@withContext
            }
            
            startAudioRecording()
            isRunning = true
            
            // Warm up the model
            warmUpModel()
            
            // Main detection loop
            // Note: processAudioBuffer uses READ_BLOCKING, so it naturally waits for audio
            // No additional delay needed - we process as fast as audio arrives
            while (isActive && isRunning) {
                processAudioBuffer()
                // Small yield to allow other coroutines to run, but no artificial delay
                delay(1)
            }
            
        } catch (e: Exception) {
            WakeWordState.errorMessage.value = "Detection error: ${e.message}"
            e.printStackTrace()
        } finally {
            stopAudioRecording()
        }
    }
    
    /**
     * Stop detection.
     */
    fun stopDetection() {
        isRunning = false
        stopAudioRecording()
        module?.destroy()
        module = null
    }
    
    private fun startAudioRecording() {
        val minBufferSize = AudioRecord.getMinBufferSize(
            SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT
        ) * 2
        
        // Use VOICE_RECOGNITION to disable Android's automatic gain control and noise suppression
        // This gives us raw audio similar to what Linux pyaudio provides
        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.VOICE_RECOGNITION,
            SAMPLE_RATE,
            CHANNEL_CONFIG,
            AUDIO_FORMAT,
            maxOf(minBufferSize, BUFFER_SIZE * 4)
        )
        
        audioRecord?.startRecording()
    }
    
    private fun stopAudioRecording() {
        audioRecord?.stop()
        audioRecord?.release()
        audioRecord = null
    }
    
    private fun warmUpModel() {
        // Dummy inference to warm up
        val dummyMfcc = Array(101) { FloatArray(NUM_MFCC) { 0f } }
        runInference(dummyMfcc)

        // Test with known input to verify model is working
        if (WakeWordState.debugMode.value) {
            testModelWithKnownInput()
            // Test with real wav files for MFCC comparison
            testWithWavFiles()
        }
    }

    /**
     * Test with real wav files to compare MFCC extraction with Python/librosa.
     * Loads positive.wav and negative.wav from assets and logs MFCC statistics.
     */
    private fun testWithWavFiles() {
        android.util.Log.d("WakeWord", "")
        android.util.Log.d("WakeWord", "=== WAV FILE MFCC TEST ===")

        for (filename in listOf("positive_16k.wav")) {  // Just test positive for detailed debug
            try {
                val audio = loadWavFromAssets(filename)
                if (audio != null) {
                    android.util.Log.d("WakeWord", "")
                    android.util.Log.d("WakeWord", "Testing: $filename")
                    android.util.Log.d("WakeWord", "Audio: samples=${audio.size}, min=${"%.6f".format(audio.minOrNull())}, max=${"%.6f".format(audio.maxOrNull())}")
                    android.util.Log.d("WakeWord", "Audio first 5: ${audio.take(5).map { "%.6f".format(it) }}")
                    android.util.Log.d("WakeWord", "Audio RMS: ${"%.6f".format(kotlin.math.sqrt(audio.map { it * it }.average().toFloat()))}")

                    // Expected Python values for positive_16k.wav:
                    android.util.Log.d("WakeWord", "Expected audio first 5: [0.007812, 0.007568, 0.001099, -0.002045, -0.005127]")

                    // Reset debug flag to get detailed intermediate logging
                    mfccExtractor.resetDebugFlag()

                    val mfcc = mfccExtractor.extractMFCC(audio)
                    val allMfcc = mfcc.flatMap { it.toList() }

                    android.util.Log.d("WakeWord", "MFCC: frames=${mfcc.size}, min=${"%.2f".format(allMfcc.minOrNull())}, max=${"%.2f".format(allMfcc.maxOrNull())}, mean=${"%.2f".format(allMfcc.average())}")

                    // Log first 3 frames for comparison with Python
                    for (i in 0 until minOf(3, mfcc.size)) {
                        val coeffs = mfcc[i].map { "%.2f".format(it) }
                        android.util.Log.d("WakeWord", "Frame $i: $coeffs")
                    }

                    // Run inference
                    val prob = runInference(mfcc)
                    android.util.Log.d("WakeWord", "Wake probability: ${"%.4f".format(prob)} (${"%.1f".format(prob * 100)}%)")

                    // Expected values from Python for positive_16k.wav
                    android.util.Log.d("WakeWord", "")
                    android.util.Log.d("WakeWord", "=== EXPECTED PYTHON VALUES ===")
                    android.util.Log.d("WakeWord", "Audio first 5: [0.007812, 0.007568, 0.001099, -0.002045, -0.005127]")
                    android.util.Log.d("WakeWord", "Pre-emph first 5: [0.007812, -0.000010, -0.006243, -0.003110, -0.003144]")
                    android.util.Log.d("WakeWord", "Frame 0 first 10: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] (center padding)")
                    android.util.Log.d("WakeWord", "Power bins 0-9: [0.000002, 0.000001, 0.000029, 0.000124, 0.000222, 0.000186, 0.000060, 0.000003, 0.000014, 0.000018]")
                    android.util.Log.d("WakeWord", "Mel energy 0-4: [0.00000284, 0.00000075, 0.00000557, 0.00000459, 0.00000411]")
                    android.util.Log.d("WakeWord", "Log mel 0-4: [-55.47, -61.27, -52.54, -53.38, -53.86]")
                    android.util.Log.d("WakeWord", "MFCC frame 0: [-259.49, 4.41, -21.95, -3.47, 1.47, -2.33, -8.70, 1.74, 7.13, 2.11, -4.30, -2.62, 0.07]")
                    android.util.Log.d("WakeWord", "Expected wake_prob: 99.6%")
                } else {
                    android.util.Log.d("WakeWord", "$filename: Failed to load")
                }
            } catch (e: Exception) {
                android.util.Log.e("WakeWord", "Error testing $filename: ${e.message}")
                e.printStackTrace()
            }
        }

        android.util.Log.d("WakeWord", "=== END WAV FILE TEST ===")
        android.util.Log.d("WakeWord", "")
    }

    /**
     * Load a WAV file from assets and convert to float array.
     * Assumes 16-bit PCM mono WAV at 16kHz.
     */
    private fun loadWavFromAssets(filename: String): FloatArray? {
        return try {
            val inputStream = context.assets.open(filename)
            val bytes = inputStream.readBytes()
            inputStream.close()

            // WAV header is 44 bytes for standard PCM
            // Find "data" chunk
            var dataOffset = 12
            while (dataOffset < bytes.size - 8) {
                val chunkId = String(bytes.sliceArray(dataOffset until dataOffset + 4))
                val chunkSize = bytes[dataOffset + 4].toInt() and 0xFF or
                    ((bytes[dataOffset + 5].toInt() and 0xFF) shl 8) or
                    ((bytes[dataOffset + 6].toInt() and 0xFF) shl 16) or
                    ((bytes[dataOffset + 7].toInt() and 0xFF) shl 24)

                if (chunkId == "data") {
                    dataOffset += 8
                    break
                }
                dataOffset += 8 + chunkSize
            }

            // Read 16-bit samples and convert to float [-1, 1]
            val numSamples = (bytes.size - dataOffset) / 2
            val audio = FloatArray(numSamples)

            for (i in 0 until numSamples) {
                val byteIdx = dataOffset + i * 2
                val sample = (bytes[byteIdx].toInt() and 0xFF) or
                    ((bytes[byteIdx + 1].toInt()) shl 8)
                audio[i] = sample.toShort() / 32768f
            }

            audio
        } catch (e: Exception) {
            android.util.Log.e("WakeWord", "Error loading $filename: ${e.message}")
            null
        }
    }

    /**
     * Test the model with known inputs to verify it's working correctly.
     * These tests help verify the ExecuTorch model produces same output as PyTorch.
     *
     * Expected outputs (from Python testing with librosa MFCC):
     * - Silence MFCC (c0~-500, mean~-40): Class1 ~4% (not wake word)
     * - Noise MFCC (c0~-100, mean~-12): Class1 ~37% (not wake word)
     * - Raw zeros (NOT MFCC): Class1 ~99% (model confused by wrong input range)
     */
    private fun testModelWithKnownInput() {
        android.util.Log.d("WakeWord", "=== MODEL VERIFICATION TEST ===")

        // Test 1: Silence-like MFCC (very negative c0, sparse other coefficients)
        // Real silence MFCC: c0 around -500, others near 0, mean ~-40
        val silenceLikeMfcc = Array(101) { FloatArray(NUM_MFCC) { coeffIdx ->
            if (coeffIdx == 0) -400f else 0f
        }}
        val silenceProb = runInference(silenceLikeMfcc)
        android.util.Log.d("WakeWord", "Test silence-like MFCC: WakeProb=$silenceProb (expected ~0.97)")

        // Test 2: Noise-like MFCC (c0 around -100, mean around -12)
        // Real noise MFCC: c0 ~-104, other coefficients ~-10 to +5, mean ~-12
        val noiseLikeMfcc = Array(101) { frameIdx ->
            FloatArray(NUM_MFCC) { coeffIdx ->
                when (coeffIdx) {
                    0 -> -100f + (frameIdx % 5) * 2f  // c0 around -100
                    else -> -12f + coeffIdx * 1.5f - (frameIdx % 3)  // others vary around -12
                }
            }
        }
        val noiseProb = runInference(noiseLikeMfcc)
        android.util.Log.d("WakeWord", "Test noise-like MFCC: WakeProb=$noiseProb (expected ~0.37)")

        // Test 3: Raw zeros (NOT valid MFCC - tests model behavior with wrong input)
        val zeros = Array(101) { FloatArray(NUM_MFCC) { 0f } }
        val zerosProb = runInference(zeros)
        android.util.Log.d("WakeWord", "Test zeros (invalid MFCC): WakeProb=$zerosProb (expected ~0.99 - model confused)")

        android.util.Log.d("WakeWord", "")
        android.util.Log.d("WakeWord", "If values match expected, model export is correct.")
        android.util.Log.d("WakeWord", "If values differ, there may be an issue with the .pte model.")
        android.util.Log.d("WakeWord", "=== END MODEL VERIFICATION ===")
    }
    
    /**
     * Read audio and process.
     */
    private fun processAudioBuffer() {
        val record = audioRecord ?: return

        // Read new audio chunk
        val partBuffer = FloatArray(OVERLAP_SIZE)
        val readResult = record.read(partBuffer, 0, OVERLAP_SIZE, AudioRecord.READ_BLOCKING)

        if (readResult < 0) return

        // Shift buffer and add new samples
        System.arraycopy(audioBuffer, OVERLAP_SIZE, audioBuffer, 0, BUFFER_SIZE - OVERLAP_SIZE)
        System.arraycopy(partBuffer, 0, audioBuffer, BUFFER_SIZE - OVERLAP_SIZE, OVERLAP_SIZE)

        // Early exit for very quiet audio (saves CPU on FFT computation)
        val rmsVolume = calculateRMS(partBuffer)  // Check only new samples for faster check
        val minThreshold = WakeWordState.minVolumeThreshold.value
        if (rmsVolume < minThreshold * 0.5f) {
            WakeWordState.probability.value = 0f
            if (WakeWordState.debugMode.value) {
                android.util.Log.d("WakeWord", "Volume too low: $rmsVolume < ${minThreshold * 0.5f}")
            }
            return
        }

        // Extract MFCC features
        val mfccFeatures = extractMFCC(audioBuffer)
        if (mfccFeatures == null) {
            if (WakeWordState.debugMode.value) {
                android.util.Log.e("WakeWord", "MFCC extraction failed")
            }
            return
        }

        if (WakeWordState.debugMode.value && System.currentTimeMillis() % 1000 < 100) {
            android.util.Log.d("WakeWord", "MFCC frames: ${mfccFeatures.size}, RMS: $rmsVolume")
        }

        // Run inference
        val probability = runInference(mfccFeatures)

        // Smooth detection
        val smoothedProb = smoothDetection(probability)
        WakeWordState.probability.value = smoothedProb

        // Track consecutive low probabilities for adaptive processing
        if (smoothedProb < 0.1f) {
            consecutiveLowProbability++
        } else {
            consecutiveLowProbability = 0
        }

        if (WakeWordState.debugMode.value && System.currentTimeMillis() % 500 < 100) {
            android.util.Log.d("WakeWord", "Prob: $probability, Smoothed: $smoothedProb")
        }

        // Check for detection
        checkDetection(smoothedProb)
    }

    /**
     * Calculate adaptive processing interval based on recent probabilities.
     * Lower probabilities = longer intervals (less CPU)
     * Higher probabilities = shorter intervals (more responsive)
     */
    private fun calculateAdaptiveInterval(): Long {
        val currentProb = WakeWordState.probability.value
        val baseInterval = WakeWordState.processingIntervalMs.value

        return when {
            // High probability - check frequently for responsiveness
            currentProb > 0.4f -> (baseInterval * 0.5f).toLong().coerceAtLeast(25L)

            // Medium probability - use normal interval
            currentProb > 0.15f -> baseInterval

            // Low probability but recent activity - use slightly longer interval
            consecutiveLowProbability < 5 -> (baseInterval * 1.5f).toLong()

            // Consistently low probability - save CPU with longer interval
            else -> (baseInterval * 2.0f).toLong().coerceAtMost(150L)
        }
    }
    
    /**
     * Calculate RMS volume.
     */
    private fun calculateRMS(samples: FloatArray): Float {
        var sum = 0.0
        for (sample in samples) {
            sum += sample * sample
        }
        return sqrt(sum / samples.size).toFloat()
    }

    /**
     * Normalize audio to a target RMS level.
     * This compensates for Android's typically quieter microphone input compared to Linux.
     */
    private fun normalizeAudio(samples: FloatArray, targetRms: Float): FloatArray {
        val currentRms = calculateRMS(samples)

        // Avoid division by zero and don't amplify very quiet signals (noise floor)
        if (currentRms < 0.0001f) {
            return samples
        }

        val gain = targetRms / currentRms

        // Limit gain to prevent excessive amplification of noise
        val clampedGain = gain.coerceIn(1f, 200f)

        if (WakeWordState.debugMode.value && System.currentTimeMillis() % 2000 < 100) {
            android.util.Log.d("WakeWord", "Audio normalization: RMS=${"%.4f".format(currentRms)}, gain=${"%.1f".format(clampedGain)}")
        }

        return FloatArray(samples.size) { i ->
            (samples[i] * clampedGain).coerceIn(-1f, 1f)
        }
    }

    /**
     * Extract MFCC features using custom MFCCExtractor.
     */
    private fun extractMFCC(audioSignal: FloatArray): Array<FloatArray>? {
        return try {
            // Don't normalize - use raw audio like Python does
            // The model was trained on librosa MFCCs from raw audio
            // Normalization changes the spectral characteristics and breaks detection
            val mfcc = mfccExtractor.extractMFCC(audioSignal)

            // Debug logging for MFCC analysis
            if (WakeWordState.debugMode.value) {
                // Log audio stats
                val audioRms = kotlin.math.sqrt(audioSignal.map { it * it }.average().toFloat())

                // Log MFCC stats
                val allMfcc = mfcc.flatMap { it.toList() }
                val mfccMin = allMfcc.minOrNull() ?: 0f
                val mfccMax = allMfcc.maxOrNull() ?: 0f
                val mfccMean = allMfcc.average().toFloat()

                // Log first frame for comparison with Python
                val firstFrame = if (mfcc.isNotEmpty()) mfcc[0].take(5).map { "%.2f".format(it) } else emptyList()

                // Log middle frame (around where wake word might be)
                val middleIdx = mfcc.size / 2
                val middleFrame = if (mfcc.size > middleIdx) mfcc[middleIdx].take(5).map { "%.2f".format(it) } else emptyList()
                android.util.Log.d("WakeWord","Android Timestamp: ${ Clock.System.now().toString()}")

                android.util.Log.d("WakeWord", "=== MFCC Debug ===")
                android.util.Log.d("WakeWord", "Audio rms=${"%.4f".format(audioRms)}")
                android.util.Log.d("WakeWord", "MFCC: frames=${mfcc.size}, min=${"%.2f".format(mfccMin)}, max=${"%.2f".format(mfccMax)}, mean=${"%.2f".format(mfccMean)}")
                android.util.Log.d("WakeWord", "MFCC frame[0] (c0-c4): $firstFrame")
                android.util.Log.d("WakeWord", "MFCC frame[$middleIdx] (c0-c4): $middleFrame")

                // IMPORTANT: Expected values for librosa MFCC (with pre-emphasis):
                // - Speech: mean around -20 to -25, range roughly [-350, +110]
                // - c0 typically -140 to -350 depending on energy
                android.util.Log.d("WakeWord", "Expected: mean~-20, range~[-350,+110], c0~-140 to -300")
            }

            mfcc
        } catch (e: Exception) {
            if (WakeWordState.debugMode.value) {
                android.util.Log.e("WakeWord", "MFCC extraction failed: ${e.message}")
            }
            null
        }
    }
    
    /**
     * Run ExecuTorch inference on MFCC features.
     */
    private fun runInference(mfccFeatures: Array<FloatArray>): Float {
        val model = module ?: return 0f

        return try {
            // Pad or truncate to expected sequence length (101 frames)
            val expectedFrames = 101
            val paddedFeatures = when {
                mfccFeatures.size == expectedFrames -> mfccFeatures
                mfccFeatures.size < expectedFrames -> {
                    // Pad with zeros
                    Array(expectedFrames) { i ->
                        if (i < mfccFeatures.size) mfccFeatures[i]
                        else FloatArray(NUM_MFCC) { 0f }
                    }
                }
                else -> {
                    // Truncate from the end to keep the most recent audio
                    mfccFeatures.takeLast(expectedFrames).toTypedArray()
                }
            }

            // Flatten MFCC array
            val flatMfcc = paddedFeatures.flatMap { it.toList() }.toFloatArray()

            // Create tensor: [1, frames, mfcc_coefficients]
            val shape = longArrayOf(1, paddedFeatures.size.toLong(), paddedFeatures[0].size.toLong())
            val tensor = Tensor.fromBlob(flatMfcc, shape)

            // Run inference
            val result = model.forward(EValue.from(tensor))
            val outputTensor = result[0].toTensor()
            val modelOutput = outputTensor.dataAsFloatArray

            // Convert to probabilities based on model output type
            // Class 0 = Not Wake Word, Class 1 = Wake Word
            // (verified: with real MFCC values, Class 1 has low probability for non-speech)
            val wakeWordProbability = if (MODEL_OUTPUTS_LOG_SOFTMAX) {
                // Model outputs log_softmax: apply exp to get probabilities
                val probabilities = modelOutput.map { kotlin.math.exp(it.toDouble()).toFloat() }
                probabilities.getOrElse(1) { 0f }  // Index 1 is wake word
            } else {
                // Model outputs raw logits: apply softmax
                // softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))
                val maxLogit = modelOutput.maxOrNull() ?: 0f
                val expValues = modelOutput.map { kotlin.math.exp((it - maxLogit).toDouble()).toFloat() }
                val sumExp = expValues.sum()
                val probabilities = expValues.map { it / sumExp }
                probabilities.getOrElse(1) { 0f }  // Index 1 is wake word
            }

            // Always log model output in debug mode for troubleshooting
            if (WakeWordState.debugMode.value) {
                android.util.Log.d("WakeWord","Android Timestamp: ${ Clock.System.now().toString()}")
                android.util.Log.d("WakeWord", "Model raw output: ${modelOutput.toList()}")
                android.util.Log.d("WakeWord", "Wake word prob: $wakeWordProbability")
            }

            wakeWordProbability

        } catch (e: Exception) {
            e.printStackTrace()
            0f
        }
    }
    
    /**
     * Apply temporal smoothing.
     */
    private fun smoothDetection(probability: Float): Float {
        detectionHistory.addLast(probability)
        while (detectionHistory.size > SMOOTHING_WINDOW) {
            detectionHistory.removeFirst()
        }
        
        return if (detectionHistory.size >= SMOOTHING_WINDOW) {
            detectionHistory.average().toFloat()
        } else {
            probability
        }
    }
    
    /**
     * Check if detection threshold is met.
     */
    private fun checkDetection(smoothedProbability: Float) {
        val currentTime = System.currentTimeMillis()
        val cooldownMs = (WakeWordState.cooldownSeconds.value * 1000).toLong()
        
        if (smoothedProbability > WakeWordState.threshold.value &&
            currentTime - lastDetectionTime > cooldownMs) {
            
            WakeWordState.detected.value = true
            lastDetectionTime = currentTime
            
            // Reset detected flag after a short delay
            android.os.Handler(android.os.Looper.getMainLooper()).postDelayed({
                WakeWordState.detected.value = false
            }, 1500)
        }
    }
}
