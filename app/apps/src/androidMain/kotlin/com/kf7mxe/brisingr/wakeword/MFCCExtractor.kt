package com.kf7mxe.brisingr.wakeword

import kotlin.math.*

/**
 * MFCC (Mel-Frequency Cepstral Coefficients) extractor.
 * Designed to match librosa.feature.mfcc() output as closely as possible.
 *
 * OPTIMIZED VERSION: Uses pre-computed twiddle factors, float arithmetic,
 * and reusable buffers to minimize allocations and CPU usage.
 *
 * Key parameters matching linux-inference-testing-v3.py:
 * - n_fft = 512 (frame_length)
 * - hop_length = 160
 * - n_mfcc = 13
 * - n_mels = 26
 * - fmin = 80
 * - fmax = 8000
 * - pre-emphasis = 0.97
 */
class MFCCExtractor(
    private val sampleRate: Int = 16000,
    private val numCep: Int = 13,
    private val nFilt: Int = 26,
    private val nfft: Int = 512,
    private val lowFreq: Int = 80,
    private val highFreq: Int = 8000,
    private val hopLength: Int = 160,
    private val preemph: Float = 0.97f
) {
    // Pre-compute expensive operations (lazy initialization)
    private val melFilterbank: Array<FloatArray> by lazy { createMelFilterbank() }
    private val hannWindow: FloatArray by lazy { createHannWindow() }
    private val dctMatrix: Array<FloatArray> by lazy { createDCTMatrix() }

    // Pre-computed FFT twiddle factors (cos/sin values) - MAJOR OPTIMIZATION
    private val fftCosTable: FloatArray by lazy { createFFTCosTable() }
    private val fftSinTable: FloatArray by lazy { createFFTSinTable() }

    // Bit-reversal lookup table for FFT
    private val bitRevTable: IntArray by lazy { createBitReversalTable() }

    // Reusable buffers to avoid allocations in hot path
    private val fftReal = FloatArray(nfft)
    private val fftImag = FloatArray(nfft)
    private val powerSpectrumBuffer = FloatArray(nfft / 2 + 1)
    private val melEnergyBuffer = FloatArray(nFilt)
    private val mfccBuffer = FloatArray(numCep)

    // Spectrum size (only need positive frequencies for real input)
    private val spectrumSize = nfft / 2 + 1

    // Debug flag - set to true once to log detailed intermediate values
    private var hasLoggedDebug = false

    // Pre-computed log10 multiplier
    private val log10Multiplier = (10.0 / ln(10.0)).toFloat()

    /**
     * Reset debug flag to allow logging again (for wav file tests).
     */
    fun resetDebugFlag() {
        hasLoggedDebug = false
    }

    /**
     * Create pre-computed cosine table for FFT twiddle factors.
     * This eliminates repeated cos() calls during FFT.
     */
    private fun createFFTCosTable(): FloatArray {
        val table = FloatArray(nfft)
        for (i in 0 until nfft) {
            table[i] = cos(-PI * i / (nfft / 2)).toFloat()
        }
        return table
    }

    /**
     * Create pre-computed sine table for FFT twiddle factors.
     */
    private fun createFFTSinTable(): FloatArray {
        val table = FloatArray(nfft)
        for (i in 0 until nfft) {
            table[i] = sin(-PI * i / (nfft / 2)).toFloat()
        }
        return table
    }

    /**
     * Create bit-reversal lookup table for FFT.
     */
    private fun createBitReversalTable(): IntArray {
        val bits = (ln(nfft.toDouble()) / ln(2.0)).toInt()
        val table = IntArray(nfft)
        for (i in 0 until nfft) {
            var reversed = 0
            var value = i
            for (j in 0 until bits) {
                reversed = (reversed shl 1) or (value and 1)
                value = value shr 1
            }
            table[i] = reversed
        }
        return table
    }

    /**
     * Extract MFCC features from audio signal.
     * OPTIMIZED: Uses pre-computed tables, float arithmetic, and reusable buffers.
     */
    fun extractMFCC(signal: FloatArray): Array<FloatArray> {
        // 1. Pre-emphasis filter (in-place style, but we need original for debug)
        val emphasized = preEmphasis(signal)

        // 2. Calculate number of frames with center padding
        val padSize = nfft / 2
        val paddedLength = signal.size + 2 * padSize
        val numFrames = 1 + (paddedLength - nfft) / hopLength

        if (numFrames <= 0) {
            return arrayOf(FloatArray(numCep))
        }

        // 3. Allocate output array
        val mfccs = Array(numFrames) { FloatArray(numCep) }

        // 4. Process each frame using optimized pipeline
        for (frameIdx in 0 until numFrames) {
            processFrameOptimized(emphasized, frameIdx, padSize, mfccs[frameIdx])
        }

        // Debug logging for frame 0 - only log once to avoid spam
        if (!hasLoggedDebug && numFrames > 0) {
            hasLoggedDebug = true
            android.util.Log.d("MFCCDebug", "=== MFCC Pipeline Debug (Optimized) ===")
            android.util.Log.d("MFCCDebug", "Input signal: length=${signal.size}")
            android.util.Log.d("MFCCDebug", "Num frames: $numFrames")
            android.util.Log.d("MFCCDebug", "MFCC frame 0: ${mfccs[0].map { "%.2f".format(it) }}")
        }

        return mfccs
    }

    /**
     * Process a single frame through the entire MFCC pipeline.
     * OPTIMIZED: Reuses buffers, uses pre-computed tables, avoids allocations.
     */
    private fun processFrameOptimized(emphasized: FloatArray, frameIdx: Int, padSize: Int, output: FloatArray) {
        val frameStart = frameIdx * hopLength - padSize

        // 1. Extract frame with windowing (combined for efficiency)
        for (i in 0 until nfft) {
            val signalIdx = frameStart + i
            val sample = when {
                signalIdx < 0 -> 0f
                signalIdx >= emphasized.size -> 0f
                else -> emphasized[signalIdx]
            }
            // Apply Hann window inline
            fftReal[i] = sample * hannWindow[i]
            fftImag[i] = 0f
        }

        // 2. Compute FFT in-place using pre-computed twiddle factors
        fftInPlaceOptimized()

        // 3. Compute power spectrum directly into buffer
        for (k in 0 until spectrumSize) {
            val re = fftReal[k]
            val im = fftImag[k]
            powerSpectrumBuffer[k] = re * re + im * im
        }

        // 4. Apply mel filterbank directly into buffer
        for (i in 0 until nFilt) {
            var energy = 0f
            val filter = melFilterbank[i]
            for (k in 0 until spectrumSize) {
                energy += powerSpectrumBuffer[k] * filter[k]
            }
            // Apply log (dB scale) inline: 10 * log10(x) = 10/ln(10) * ln(x)
            melEnergyBuffer[i] = log10Multiplier * ln(max(energy, 1e-10f))
        }

        // 5. Apply DCT to get MFCCs
        for (k in 0 until numCep) {
            var sum = 0f
            val dctRow = dctMatrix[k]
            for (n in 0 until nFilt) {
                sum += dctRow[n] * melEnergyBuffer[n]
            }
            output[k] = sum
        }
    }

    /**
     * Optimized in-place FFT using pre-computed twiddle factors.
     * Uses float arithmetic throughout for speed on mobile.
     */
    private fun fftInPlaceOptimized() {
        // Bit-reversal permutation using lookup table
        for (i in 0 until nfft) {
            val j = bitRevTable[i]
            if (i < j) {
                // Swap
                var temp = fftReal[i]
                fftReal[i] = fftReal[j]
                fftReal[j] = temp
                temp = fftImag[i]
                fftImag[i] = fftImag[j]
                fftImag[j] = temp
            }
        }

        // Cooley-Tukey FFT with pre-computed twiddle factors
        var step = 1
        var twiddleStep = nfft / 2
        while (step < nfft) {
            val halfStep = step
            step *= 2

            for (k in 0 until halfStep) {
                val twiddleIdx = k * twiddleStep
                val wr = fftCosTable[twiddleIdx]
                val wi = fftSinTable[twiddleIdx]

                var i = k
                while (i < nfft) {
                    val iPlus = i + halfStep
                    val tr = wr * fftReal[iPlus] - wi * fftImag[iPlus]
                    val ti = wr * fftImag[iPlus] + wi * fftReal[iPlus]

                    fftReal[iPlus] = fftReal[i] - tr
                    fftImag[iPlus] = fftImag[i] - ti
                    fftReal[i] = fftReal[i] + tr
                    fftImag[i] = fftImag[i] + ti

                    i += step
                }
            }
            twiddleStep /= 2
        }
    }

    /**
     * Pre-emphasis filter: y[n] = x[n] - alpha * x[n-1]
     * Matches Python: np.append(signal[0], signal[1:] - 0.97 * signal[:-1])
     */
    private fun preEmphasis(signal: FloatArray): FloatArray {
        if (signal.isEmpty()) return signal

        val result = FloatArray(signal.size)
        result[0] = signal[0]
        for (i in 1 until signal.size) {
            result[i] = signal[i] - preemph * signal[i - 1]
        }
        return result
    }

    /**
     * Create Hann window to match librosa's default.
     * Note: librosa uses periodic Hann window (fftbins=True), which divides by n, not n-1.
     */
    private fun createHannWindow(): FloatArray {
        return FloatArray(nfft) { n ->
            (0.5 - 0.5 * cos(2.0 * PI * n / nfft)).toFloat()
        }
    }

    /**
     * Create Mel filterbank matching librosa's mel filter with 'slaney' normalization.
     * Uses Slaney mel scale formula (librosa default, NOT HTK).
     * Slaney scale: linear below 1000 Hz, logarithmic above.
     *
     * IMPORTANT: Uses frequency interpolation (like librosa) NOT bin index interpolation.
     * This ensures filter weights match librosa exactly.
     */
    private fun createMelFilterbank(): Array<FloatArray> {
        // Slaney mel scale (librosa default when htk=False)
        // Linear below 1000 Hz, logarithmic above
        val fSp = 200.0 / 3.0  // 66.67 Hz per mel below 1000 Hz
        val minLogHz = 1000.0
        val minLogMel = minLogHz / fSp  // 15 mel
        val logStep = ln(6.4) / 27.0

        fun hzToMel(hz: Double): Double {
            return if (hz < minLogHz) {
                hz / fSp
            } else {
                minLogMel + ln(hz / minLogHz) / logStep
            }
        }

        fun melToHz(mel: Double): Double {
            return if (mel < minLogMel) {
                fSp * mel
            } else {
                minLogHz * exp(logStep * (mel - minLogMel))
            }
        }

        // Convert FFT bin to frequency in Hz
        fun binToHz(bin: Int): Double {
            return bin.toDouble() * sampleRate / nfft
        }

        val lowMel = hzToMel(lowFreq.toDouble())
        val highMel = hzToMel(highFreq.toDouble())

        // Create nFilt + 2 equally spaced points in mel scale
        val hzPoints = DoubleArray(nFilt + 2) { i ->
            val mel = lowMel + i * (highMel - lowMel) / (nFilt + 1)
            melToHz(mel)
        }

        // Create triangular filters using FREQUENCY interpolation (matching librosa)
        // This is the key difference from bin-index interpolation
        return Array(nFilt) { i ->
            val filterBank = FloatArray(spectrumSize)

            val leftHz = hzPoints[i]
            val centerHz = hzPoints[i + 1]
            val rightHz = hzPoints[i + 2]

            // Slaney normalization factor: 2 / (right_hz - left_hz)
            val enorm = 2.0 / (rightHz - leftHz)

            // For each FFT bin, calculate weight based on its frequency
            for (k in 0 until spectrumSize) {
                val freq = binToHz(k)

                if (freq >= leftHz && freq < centerHz && centerHz != leftHz) {
                    // Rising edge - interpolate based on frequency
                    filterBank[k] = ((freq - leftHz) / (centerHz - leftHz) * enorm).toFloat()
                } else if (freq >= centerHz && freq < rightHz && rightHz != centerHz) {
                    // Falling edge - interpolate based on frequency
                    filterBank[k] = ((rightHz - freq) / (rightHz - centerHz) * enorm).toFloat()
                }
            }

            filterBank
        }
    }

    /**
     * Apply Mel filterbank to power spectrum.
     */
    private fun applyMelFilterbank(powerSpectrum: List<FloatArray>): List<FloatArray> {
        return powerSpectrum.map { spectrum ->
            FloatArray(nFilt) { i ->
                var energy = 0f
                for (k in 0 until spectrumSize) {
                    energy += spectrum[k] * melFilterbank[i][k]
                }
                energy
            }
        }
    }

    /**
     * Create DCT-II matrix for MFCC computation.
     * Uses 'ortho' normalization to match librosa.
     */
    private fun createDCTMatrix(): Array<FloatArray> {
        // DCT-II with ortho normalization
        val scale0 = sqrt(1.0 / nFilt)
        val scaleK = sqrt(2.0 / nFilt)

        return Array(numCep) { k ->
            FloatArray(nFilt) { n ->
                val cosVal = cos(PI * k * (2 * n + 1) / (2.0 * nFilt))
                val scale = if (k == 0) scale0 else scaleK
                (scale * cosVal).toFloat()
            }
        }
    }

    /**
     * Apply DCT to log mel energies to get MFCCs.
     */
    private fun applyDCT(logMelEnergies: FloatArray): FloatArray {
        return FloatArray(numCep) { k ->
            var sum = 0f
            for (n in 0 until nFilt) {
                sum += dctMatrix[k][n] * logMelEnergies[n]
            }
            sum
        }
    }
}
