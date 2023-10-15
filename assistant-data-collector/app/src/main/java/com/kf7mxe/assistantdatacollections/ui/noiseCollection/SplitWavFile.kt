package com.kf7mxe.assistantdatacollections.ui.noiseCollection

import android.content.Context
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.IOException

fun splitWavFile(inputFilePath: String, outputFilePath:String,context:Context) {
    val outputDirectory = File(context.filesDir,outputFilePath) // Change to your desired output directory
    if (!outputDirectory.exists()) {
        outputDirectory.mkdir()
    }
    val inputFile = File(context.filesDir,inputFilePath)
    val fileInputStream = inputFile.inputStream()
    val audioFormat = AudioFormat.ENCODING_PCM_16BIT
//    val sampleRate = 44100 // Adjust to your WAV file's sample rate
//    val seconds = 1 // Split WAV every X seconds
//    val bufferSize = seconds * audioFormat * sampleRate
//
//
//    val buffer = ByteArray(bufferSize)
//

//    val fileSize = inputFile.length()
//    val inputFileSize = fileSize.toInt()
//    val numberOfChunks = inputFileSize / bufferSize
//    val lastChunkSize = inputFileSize % bufferSize
//
//    for (i in 0 until numberOfChunks) {
//        val outputFile = File(outputDirectory, "chunk_$i.wav")
//        val fileOutputStream = FileOutputStream(outputFile)
//        // waveHeader
//        wavHeader(fileOutputStream, bufferSize.toLong(), inputFileSize.toLong(), 1, 2 * sampleRate.toLong(), sampleRate)
//        fileInputStream.read(buffer)
//        fileOutputStream.write(buffer)
//        fileOutputStream.close()
//    }
//    if (lastChunkSize > 0) {
//        val outputFile = File(outputDirectory, "chunk_$numberOfChunks.wav")
//        val fileOutputStream = FileOutputStream(outputFile)
//        fileInputStream.read(buffer, 0, lastChunkSize)
//        fileOutputStream.write(buffer, 0, lastChunkSize)
//        fileOutputStream.close()
//    }
//    fileInputStream.close()
//
//
//







    val bufferSize = AudioRecord.getMinBufferSize(8000, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT) // Adjust this based on your needs
    val bytesPerSample = 2 // 16-bit PCM audio
    val sampleRate = 44100 // Sample rate of your WAV file
    val durationInSeconds = 1 // Duration of each chunk in seconds

    try {
//        val inputFile = FileInputStream(inputFilePath)
        val inputBuffer = ByteArray(bufferSize)

        val headerBuffer = ByteArray(44) // WAV file header size is typically 44 bytes

        // Read the WAV file header
        fileInputStream.read(headerBuffer)

        var chunkNumber = 1

        while (true) {
            val outputFile = File(outputDirectory, "chunk_$chunkNumber.wav")
            val fileOutputStream = FileOutputStream(outputFile)

            // Write the WAV file header to the new chunk
            fileOutputStream.write(headerBuffer)

            var totalBytesRead = 0
            var bytesRead: Int

            while (fileInputStream.read(inputBuffer).also { bytesRead = it } != -1) {
                totalBytesRead += bytesRead

                if (totalBytesRead <= sampleRate * bytesPerSample * durationInSeconds) {
                    // Continue writing to the current chunk
                    fileOutputStream.write(inputBuffer, 0, bytesRead)
                } else {
                    // Start a new chunk
                    break
                }
            }

            fileOutputStream.close()

            if (totalBytesRead == 0) {
                break // Reached the end of the file
            }

            chunkNumber++
        }

        fileInputStream.close()
    } catch (e: IOException) {
        e.printStackTrace()
    }


}

private fun wavHeader(
    fileOutputStream: FileOutputStream,
    totalAudioLen: Long,
    totalDataLen: Long,
    channels: Int,
    byteRate: Long,
    sampleRate: Int,
) {
    val bpp = 16
    try {
        val header = ByteArray(44)
        header[0] = 'R'.code.toByte() // RIFF/WAVE header
        header[1] = 'I'.code.toByte()
        header[2] = 'F'.code.toByte()
        header[3] = 'F'.code.toByte()
        header[4] = (totalDataLen and 0xffL).toByte()
        header[5] = (totalDataLen shr 8 and 0xffL).toByte()
        header[6] = (totalDataLen shr 16 and 0xffL).toByte()
        header[7] = (totalDataLen shr 24 and 0xffL).toByte()
        header[8] = 'W'.code.toByte()
        header[9] = 'A'.code.toByte()
        header[10] = 'V'.code.toByte()
        header[11] = 'E'.code.toByte()
        header[12] = 'f'.code.toByte() // 'fmt ' chunk
        header[13] = 'm'.code.toByte()
        header[14] = 't'.code.toByte()
        header[15] = ' '.code.toByte()
        header[16] = 16 // 4 bytes: size of 'fmt ' chunk
        header[17] = 0
        header[18] = 0
        header[19] = 0
        header[20] = 1 // format = 1
        header[21] = 0
        header[22] = channels.toByte()
        header[23] = 0
        header[24] = (sampleRate.toLong() and 0xffL).toByte()
        header[25] = (sampleRate.toLong() shr 8 and 0xffL).toByte()
        header[26] = (sampleRate.toLong() shr 16 and 0xffL).toByte()
        header[27] = (sampleRate.toLong() shr 24 and 0xffL).toByte()
        header[28] = (byteRate and 0xffL).toByte()
        header[29] = (byteRate shr 8 and 0xffL).toByte()
        header[30] = (byteRate shr 16 and 0xffL).toByte()
        header[31] = (byteRate shr 24 and 0xffL).toByte()
        header[32] = (2 * 16 / 8).toByte() // block align
        header[33] = 0
        header[34] = bpp.toByte() // bits per sample
        header[35] = 0
        header[36] = 'd'.code.toByte()
        header[37] = 'a'.code.toByte()
        header[38] = 't'.code.toByte()
        header[39] = 'a'.code.toByte()
        header[40] = (totalAudioLen and 0xffL).toByte()
        header[41] = (totalAudioLen shr 8 and 0xffL).toByte()
        header[42] = (totalAudioLen shr 16 and 0xffL).toByte()
        header[43] = (totalAudioLen shr 24 and 0xffL).toByte()
        fileOutputStream.write(header, 0, 44)
    } catch (e: Exception) {
        e.printStackTrace()
    }
}