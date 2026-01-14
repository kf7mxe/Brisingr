package com.kf7mxe.brisingr.wakeword

import kotlin.math.*

/**
 * MFCC (Mel-Frequency Cepstral Coefficients) extractor.
 * Designed to match librosa.feature.mfcc() output as closely as possible.
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
    // Pre-compute expensive operations
    private val melFilterbank: Array<FloatArray> by lazy { createMelFilterbank() }
    private val hannWindow: FloatArray by lazy { createHannWindow() }
    private val dctMatrix: Array<FloatArray> by lazy { createDCTMatrix() }

    // Spectrum size (only need positive frequencies for real input)
    private val spectrumSize = nfft / 2 + 1

    // Debug flag - set to true once to log detailed intermediate values
    private var hasLoggedDebug = false

    /**
     * Reset debug flag to allow logging again (for wav file tests).
     */
    fun resetDebugFlag() {
        hasLoggedDebug = false
    }

    /**
     * Extract MFCC features from audio signal.
     * Matches librosa.feature.mfcc() behavior.
     */
    fun extractMFCC(signal: FloatArray): Array<FloatArray> {
        // 1. Pre-emphasis filter (matches Python implementation)
        val emphasized = preEmphasis(signal)

        // 2. Frame the signal with center padding (librosa default)
        val frames = frameSignal(emphasized)

        if (frames.isEmpty()) {
            return arrayOf(FloatArray(numCep))
        }

        // 3. Apply Hamming window to each frame
        val windowedFrames = applyWindow(frames)

        // 4. Compute power spectrum via FFT
        val powerSpectrum = computePowerSpectrum(windowedFrames)

        // 5. Apply Mel filterbank
        val melEnergies = applyMelFilterbank(powerSpectrum)

        // 6. Take log of mel energies using dB scale (10*log10) to match librosa
        val logMelEnergies = melEnergies.map { frame ->
            FloatArray(frame.size) { i ->
                (10.0 * log10(max(frame[i], 1e-10f).toDouble())).toFloat()
            }
        }

        // 7. Apply DCT to get MFCCs
        val mfccs = logMelEnergies.map { frame ->
            applyDCT(frame)
        }.toTypedArray()

        // Debug logging for frame 0 - only log once to avoid spam
        if (!hasLoggedDebug && frames.isNotEmpty()) {
            hasLoggedDebug = true
            android.util.Log.d("MFCCDebug", "=== MFCC Pipeline Debug (Frame 0) ===")
            android.util.Log.d("MFCCDebug", "Input signal: length=${signal.size}, first 5: ${signal.take(5).map { "%.6f".format(it) }}")
            android.util.Log.d("MFCCDebug", "After pre-emphasis: first 5: ${emphasized.take(5).map { "%.6f".format(it) }}")
            android.util.Log.d("MFCCDebug", "Frame 0 (first 10): ${frames[0].take(10).map { "%.6f".format(it) }}")
            android.util.Log.d("MFCCDebug", "Windowed frame 0 (first 10): ${windowedFrames[0].take(10).map { "%.6f".format(it) }}")
            android.util.Log.d("MFCCDebug", "Power spectrum frame 0 (bins 0-9): ${powerSpectrum[0].take(10).map { "%.6f".format(it) }}")
            android.util.Log.d("MFCCDebug", "Mel energy frame 0 (filters 0-4): ${melEnergies[0].take(5).map { "%.8f".format(it) }}")
            android.util.Log.d("MFCCDebug", "Log mel frame 0 (filters 0-4): ${logMelEnergies[0].take(5).map { "%.2f".format(it) }}")
            android.util.Log.d("MFCCDebug", "MFCC frame 0: ${mfccs[0].map { "%.2f".format(it) }}")
            android.util.Log.d("MFCCDebug", "=== Expected Python values (positive_16k.wav) ===")
            android.util.Log.d("MFCCDebug", "MFCC frame 0: [-259.49, 4.41, -21.95, -3.47, 1.47, -2.33, -8.70, 1.74, 7.13, 2.11, -4.30, -2.62, 0.07]")
        }

        return mfccs
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
     * Frame the signal with center padding to match librosa's center=True behavior.
     * Uses constant (zero) padding at edges to match librosa's default pad_mode='constant'.
     */
    private fun frameSignal(signal: FloatArray): List<FloatArray> {
        if (signal.isEmpty()) return emptyList()

        // Librosa center=True pads by n_fft//2 on each side
        val padSize = nfft / 2
        val paddedLength = signal.size + 2 * padSize
        val paddedSignal = FloatArray(paddedLength)  // FloatArray is initialized to zeros

        // Copy original signal to center (padding at start and end is already zeros)
        System.arraycopy(signal, 0, paddedSignal, padSize, signal.size)

        // Calculate number of frames
        val numFrames = 1 + (paddedSignal.size - nfft) / hopLength

        val frames = ArrayList<FloatArray>(numFrames)
        for (i in 0 until numFrames) {
            val start = i * hopLength
            if (start + nfft <= paddedSignal.size) {
                val frame = FloatArray(nfft)
                System.arraycopy(paddedSignal, start, frame, 0, nfft)
                frames.add(frame)
            }
        }

        return frames
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
     * Apply Hann window to frames (matches librosa default).
     */
    private fun applyWindow(frames: List<FloatArray>): List<FloatArray> {
        return frames.map { frame ->
            FloatArray(frame.size) { i ->
                frame[i] * hannWindow[i]
            }
        }
    }

    /**
     * Compute power spectrum using real-valued FFT.
     * Returns magnitude squared for each positive frequency bin.
     * Note: Don't divide by nfft - librosa doesn't do this.
     */
    private fun computePowerSpectrum(frames: List<FloatArray>): List<FloatArray> {
        return frames.map { frame ->
            // Compute real FFT
            val (real, imag) = realFFT(frame)

            // Compute power spectrum: |X[k]|^2 (no division by nfft)
            FloatArray(spectrumSize) { k ->
                (real[k] * real[k] + imag[k] * imag[k]).toFloat()
            }
        }
    }

    /**
     * Compute FFT of real-valued input using Cooley-Tukey algorithm.
     * Returns (real, imaginary) arrays for positive frequencies only.
     *
     * O(n log n) complexity - much faster than DFT for large n.
     */
    private fun realFFT(input: FloatArray): Pair<DoubleArray, DoubleArray> {
        // Copy input to complex arrays (imaginary part is 0 for real input)
        val real = DoubleArray(nfft) { if (it < input.size) input[it].toDouble() else 0.0 }
        val imag = DoubleArray(nfft) { 0.0 }

        // In-place Cooley-Tukey FFT
        fftInPlace(real, imag)

        // Return only positive frequencies (0 to N/2)
        return Pair(
            real.sliceArray(0 until spectrumSize),
            imag.sliceArray(0 until spectrumSize)
        )
    }

    /**
     * In-place Cooley-Tukey FFT algorithm.
     * Requires input size to be a power of 2.
     */
    private fun fftInPlace(real: DoubleArray, imag: DoubleArray) {
        val n = real.size

        // Bit-reversal permutation
        var j = 0
        for (i in 0 until n - 1) {
            if (i < j) {
                // Swap real[i] and real[j]
                var temp = real[i]
                real[i] = real[j]
                real[j] = temp
                // Swap imag[i] and imag[j]
                temp = imag[i]
                imag[i] = imag[j]
                imag[j] = temp
            }
            var k = n / 2
            while (k <= j) {
                j -= k
                k /= 2
            }
            j += k
        }

        // Cooley-Tukey iterative FFT
        var step = 1
        while (step < n) {
            val halfStep = step
            step *= 2
            val angleStep = -PI / halfStep

            for (k in 0 until halfStep) {
                val angle = k * angleStep
                val wr = cos(angle)
                val wi = sin(angle)

                var i = k
                while (i < n) {
                    val iPlus = i + halfStep
                    val tr = wr * real[iPlus] - wi * imag[iPlus]
                    val ti = wr * imag[iPlus] + wi * real[iPlus]

                    real[iPlus] = real[i] - tr
                    imag[iPlus] = imag[i] - ti
                    real[i] += tr
                    imag[i] += ti

                    i += step
                }
            }
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
