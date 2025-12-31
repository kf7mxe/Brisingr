package com.kf7mxe.brisingr.wakeword

import kotlin.math.*

/**
 * Simple MFCC (Mel-Frequency Cepstral Coefficients) extractor.
 * Ported from python_speech_features for wake word detection.
 */
class MFCCExtractor(
    private val sampleRate: Int = 16000,
    private val numCep: Int = 13,
    private val nFilt: Int = 26,
    private val nfft: Int = 512,
    private val lowFreq: Int = 0,
    private val highFreq: Int = sampleRate / 2,
    private val winLen: Float = 0.025f,
    private val winStep: Float = 0.01f,
    private val preemph: Float = 0.97f
) {
    
    /**
     * Extract MFCC features from audio signal.
     */
    fun extractMFCC(signal: FloatArray): Array<FloatArray> {
        // Pre-emphasis filter
        val emphasized = preEmphasis(signal)
        
        // Frame the signal
        val frames = frameSignal(emphasized)
        
        // Apply Hamming window
        val windowedFrames = applyWindow(frames)
        
        // Compute power spectrum
        val powerSpectrum = computePowerSpectrum(windowedFrames)
        
        // Apply Mel filterbank
        val filterbanks = applyMelFilterbank(powerSpectrum)
        
        // Take log
        val logFilterbanks = filterbanks.map { frame ->
            frame.map { max(it, 1e-10f).let { v -> ln(v.toDouble()).toFloat() } }.toFloatArray()
        }
        
        // Apply DCT to get MFCCs
        return logFilterbanks.map { frame -> dct(frame) }.toTypedArray()
    }
    
    private fun preEmphasis(signal: FloatArray): FloatArray {
        val result = FloatArray(signal.size)
        result[0] = signal[0]
        for (i in 1 until signal.size) {
            result[i] = signal[i] - preemph * signal[i - 1]
        }
        return result
    }
    
    private fun frameSignal(signal: FloatArray): List<FloatArray> {
        val frameLength = (winLen * sampleRate).toInt()
        val frameStep = (winStep * sampleRate).toInt()
        val signalLength = signal.size
        
        val numFrames = if (signalLength <= frameLength) {
            1
        } else {
            1 + ceil((signalLength - frameLength).toDouble() / frameStep).toInt()
        }
        
        val frames = mutableListOf<FloatArray>()
        for (i in 0 until numFrames) {
            val start = i * frameStep
            val frame = FloatArray(frameLength)
            for (j in 0 until frameLength) {
                val idx = start + j
                frame[j] = if (idx < signalLength) signal[idx] else 0f
            }
            frames.add(frame)
        }
        return frames
    }
    
    private fun applyWindow(frames: List<FloatArray>): List<FloatArray> {
        if (frames.isEmpty()) return emptyList()
        
        val frameLength = frames[0].size
        val window = FloatArray(frameLength) { i ->
            (0.54 - 0.46 * cos(2 * PI * i / (frameLength - 1))).toFloat()
        }
        
        return frames.map { frame ->
            FloatArray(frameLength) { i -> frame[i] * window[i] }
        }
    }
    
    private fun computePowerSpectrum(frames: List<FloatArray>): List<FloatArray> {
        val spectrumSize = nfft / 2 + 1
        
        return frames.map { frame ->
            // Zero-pad to nfft
            val padded = FloatArray(nfft)
            for (i in frame.indices) {
                if (i < nfft) padded[i] = frame[i]
            }
            
            // Compute DFT magnitude squared
            val spectrum = FloatArray(spectrumSize)
            for (k in 0 until spectrumSize) {
                var real = 0.0
                var imag = 0.0
                for (n in 0 until nfft) {
                    val angle = -2 * PI * k * n / nfft
                    real += padded[n] * cos(angle)
                    imag += padded[n] * sin(angle)
                }
                spectrum[k] = ((real * real + imag * imag) / nfft).toFloat()
            }
            spectrum
        }
    }
    
    private fun applyMelFilterbank(powerSpectrum: List<FloatArray>): List<FloatArray> {
        val filterbank = createMelFilterbank()
        
        return powerSpectrum.map { spectrum ->
            FloatArray(nFilt) { i ->
                var sum = 0f
                for (j in spectrum.indices) {
                    sum += spectrum[j] * filterbank[i][j]
                }
                sum
            }
        }
    }
    
    private fun createMelFilterbank(): Array<FloatArray> {
        val spectrumSize = nfft / 2 + 1
        
        // Convert frequencies to Mel scale
        fun hzToMel(hz: Double) = 2595 * log10(1 + hz / 700)
        fun melToHz(mel: Double) = 700 * (10.0.pow(mel / 2595) - 1)
        
        val lowMel = hzToMel(lowFreq.toDouble())
        val highMel = hzToMel(highFreq.toDouble())
        
        // Create mel points
        val melPoints = DoubleArray(nFilt + 2) { i ->
            lowMel + i * (highMel - lowMel) / (nFilt + 1)
        }
        
        // Convert back to Hz and then to FFT bins
        val binPoints = melPoints.map { mel ->
            ((melToHz(mel) * (nfft + 1) / sampleRate).toInt()).coerceIn(0, spectrumSize - 1)
        }
        
        // Create triangular filters
        return Array(nFilt) { i ->
            FloatArray(spectrumSize) { j ->
                val left = binPoints[i]
                val center = binPoints[i + 1]
                val right = binPoints[i + 2]
                
                when {
                    j < left -> 0f
                    j < center -> ((j - left).toFloat() / (center - left).coerceAtLeast(1)).toFloat()
                    j < right -> ((right - j).toFloat() / (right - center).coerceAtLeast(1)).toFloat()
                    else -> 0f
                }
            }
        }
    }
    
    private fun dct(input: FloatArray): FloatArray {
        val n = input.size
        val result = FloatArray(numCep)
        
        for (k in 0 until numCep) {
            var sum = 0.0
            for (i in 0 until n) {
                sum += input[i] * cos(PI * k * (2 * i + 1) / (2 * n))
            }
            result[k] = (sum * sqrt(2.0 / n)).toFloat()
        }
        
        // Apply liftering
        for (i in 0 until numCep) {
            result[i] *= (1 + 22.0 / 2 * sin(PI * i / 22)).toFloat()
        }
        
        return result
    }
}
