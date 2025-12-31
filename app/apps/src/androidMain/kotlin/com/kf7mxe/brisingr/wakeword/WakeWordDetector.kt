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
        private const val OVERLAP_SIZE = SAMPLE_RATE / 2  // 0.5 second overlap
        
        // MFCC settings (matching Python implementation)
        private const val NUM_MFCC = 13
        private const val N_FILT = 26
        private const val NFFT = 512
        private const val LOW_FREQ = 0
        private const val HIGH_FREQ = SAMPLE_RATE / 2
        
        // Detection settings
        private const val SMOOTHING_WINDOW = 5
        private const val MIN_VOLUME_THRESHOLD = 0.001f
        private const val PROCESSING_INTERVAL_MS = 50L
        
        private const val MODEL_NAME = "wake_word.pte"
    }
    
    // MFCC extractor (custom implementation)
    private val mfccExtractor = MFCCExtractor(
        sampleRate = SAMPLE_RATE,
        numCep = NUM_MFCC,
        nFilt = N_FILT,
        nfft = NFFT,
        lowFreq = LOW_FREQ,
        highFreq = HIGH_FREQ
    )
    private var audioRecord: AudioRecord? = null
    private var module: Module? = null
    private var isRunning = false
    
    // Audio buffer (1 second)
    private val audioBuffer = FloatArray(BUFFER_SIZE)
    
    // Detection history for smoothing
    private val detectionHistory = ArrayDeque<Float>(SMOOTHING_WINDOW)
    
    private var lastDetectionTime = 0L
    
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
            while (isActive && isRunning) {
                processAudioBuffer()
                delay(PROCESSING_INTERVAL_MS)
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
        
        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.MIC,
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
        val dummyMfcc = Array(100) { FloatArray(NUM_MFCC) { 0f } }
        runInference(dummyMfcc)
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
        
        // Check volume threshold
        val rmsVolume = calculateRMS(audioBuffer)
        if (rmsVolume < MIN_VOLUME_THRESHOLD) {
            WakeWordState.probability.value = 0f
            return
        }
        
        // Extract MFCC features
        val mfccFeatures = extractMFCC(audioBuffer)
        if (mfccFeatures == null) return
        
        // Run inference
        val probability = runInference(mfccFeatures)
        
        // Smooth detection
        val smoothedProb = smoothDetection(probability)
        WakeWordState.probability.value = smoothedProb
        
        // Check for detection
        checkDetection(smoothedProb)
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
     * Extract MFCC features using custom MFCCExtractor.
     */
    private fun extractMFCC(audioSignal: FloatArray): Array<FloatArray>? {
        return try {
            mfccExtractor.extractMFCC(audioSignal)
        } catch (e: Exception) {
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
                    // Truncate
                    mfccFeatures.take(expectedFrames).toTypedArray()
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
            val scores = outputTensor.dataAsFloatArray
            
            // Softmax and return wake word probability (index 1)
            val expScores = scores.map { kotlin.math.exp(it.toDouble()) }
            val sumExp = expScores.sum()
            val probabilities = expScores.map { (it / sumExp).toFloat() }
            
            probabilities.getOrElse(1) { 0f }
            
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
