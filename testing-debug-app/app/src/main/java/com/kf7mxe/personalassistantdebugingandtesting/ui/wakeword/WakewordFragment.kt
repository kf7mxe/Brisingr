package com.kf7mxe.personalassistantdebugingandtesting.ui.wakeword

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioDeviceInfo
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.Message
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.core.content.ContextCompat.checkSelfPermission
import androidx.fragment.app.Fragment
import androidx.lifecycle.ViewModelProvider
import com.example.personalassistantdebugandtesting.databinding.FragmentWakewordBinding
import org.merlyn.kotlinspeechfeatures.SpeechFeatures
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import org.pytorch.Tensor
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.nio.ByteBuffer
import kotlin.math.absoluteValue

class WakewordFragment : Fragment() {

    private var _binding: FragmentWakewordBinding ? = null


    private val speechFeatures = SpeechFeatures()


    private val SAMPLE_RATE = 8000
    private val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
    private val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_FLOAT

    // buffer size for 1 second of audio
    private val SECONDS = 1
    private val BUFFER_SIZE = SAMPLE_RATE * SECONDS // 1 seconds buffer
    private val OVERLAP_SIZE = SAMPLE_RATE / 2 // 0.5 seconds overlap

    // Variables
    private var audioThread: Thread? = null
    private var isRecording = false
//    private val circularBuffer = CircularBuffer(BUFFER_SIZE)

    private var mAudioRecord: AudioRecord? = null
    private var mIsRecording = false

    private val PERMISSIONS_REQUEST_RECORD_AUDIO = 1



    private var mRecorder: AudioRecord? = null

    private var mRecording = false

    private var done = false

    private var mThread: Thread? = null

    private var buffer: ShortArray? = null



    var oldFeatures: Array<FloatArray> = Array<FloatArray>(0) { FloatArray(0) }


//    private val BUFFER_SIZE = 4096

    private var bufferForInference: ShortArray? = null
    // This property is only valid between onCreateView and
    // onDestroyView.




    var module:Module? = null;

//    var litModuel:


    private val binding get() = _binding!!

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        try {
            module = LiteModuleLoader.load(assetFilePath("android_lite_model.pt1"))
//            module = Module.load(assetFilePath("android_lite_model.pt1"))
        } catch (e: Exception) {
            e.printStackTrace()
        }

    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        val dashboardViewModel =
            ViewModelProvider(this).get(WakewordViewModel::class.java)

        _binding = FragmentWakewordBinding.inflate(inflater, container, false)
        val root: View = binding.root





        startRecording()

        return root
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }

    val updatePercentagehandler = Handler(Looper.getMainLooper()){
        val pair = it.obj as Pair<Double, Double>
        println("pair is ${pair.first} ${pair.second}")

        // format the pair.second.toSting() to be a percentage

        binding.negativeTextView.text = "${ String.format("%.2f",pair.second)} %"
        binding.positiveTextView.text = "${ String.format("%.2f",pair.first)} %"

        if (pair.second > 80.0) {
            binding.assistantStatusTextView.text = "Activated"
        }
        true
    }

    fun startRecording() {
        if (checkSelfPermission(
                this.requireContext(),
                Manifest.permission.RECORD_AUDIO
            ) != PackageManager.PERMISSION_GRANTED
        ) {

            requestPermissions(
                arrayOf(Manifest.permission.RECORD_AUDIO),
                PERMISSIONS_REQUEST_RECORD_AUDIO
            )
            return
        }

        // get the AudioDeviceInfo for the microphone
        val test_SAMPLING_RATE_IN_HZ = 44100;
        val test_BUFFER_SIZE_FACTOR = 2

        val test_CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
        val test_AUDIO_FORMAT = AudioFormat.ENCODING_PCM_FLOAT
        val test_BUFFER_SIZE = AudioRecord.getMinBufferSize(test_SAMPLING_RATE_IN_HZ,
            test_CHANNEL_CONFIG, test_AUDIO_FORMAT) * test_BUFFER_SIZE_FACTOR;



//        mAudioRecord = AudioRecord(MediaRecorder.AudioSource.DEFAULT, SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT, BUFFER_SIZE)
//        mAudioRecord?.startRecording()


        mAudioRecord = AudioRecord(MediaRecorder.AudioSource.DEFAULT, test_SAMPLING_RATE_IN_HZ, test_CHANNEL_CONFIG, test_AUDIO_FORMAT, test_BUFFER_SIZE)
        mAudioRecord?.startRecording()




        audioThread = Thread {

            var postiveScore:FloatArray = FloatArray(4)
            var negativeScore:FloatArray = FloatArray(4)
            var positiveNegativeScoreIndex = 0
            var wholeBuffer = FloatArray(BUFFER_SIZE)
            while (isRecording) {
                var partBuffer = FloatArray(OVERLAP_SIZE)
                 val resultOfRead = mAudioRecord!!.read (partBuffer, 0, OVERLAP_SIZE, AudioRecord.READ_BLOCKING)

                val topOfWholeBuffer = wholeBuffer.copyOfRange(OVERLAP_SIZE, wholeBuffer.size)
                wholeBuffer = topOfWholeBuffer + partBuffer




                if (resultOfRead < 0){
                    try {
                        Thread.sleep(100,0);
                    } catch (e: InterruptedException) {
                        e.printStackTrace()
                    }
                }


                if (mAudioRecord!!.state != AudioRecord.STATE_INITIALIZED) {
                    println("AudioRecord is not initialized")
                    println("AudioRecord state: ${mAudioRecord!!.state}")
                    break
                }
                if (resultOfRead == AudioRecord.ERROR_INVALID_OPERATION && resultOfRead == AudioRecord.ERROR_BAD_VALUE) {
                    println("bytes read: $resultOfRead")
                    println("AudioRecord error")
                    println(AudioRecord.ERROR_INVALID_OPERATION)
                    println(AudioRecord.ERROR_BAD_VALUE)
                }

                wholeBuffer


//                println("time in seconds: ${System.currentTimeMillis() / 1000}")


                val features = speechFeatures.mfcc(signal = wholeBuffer,
                    sampleRate = SAMPLE_RATE,
                    numCep = 16
                )



                val longArrayOf = longArrayOf(1,1,features.size.toLong(), features[0].size.toLong())
                val toByteBuffer = features.flatMap { it.toList() }.map { it.toFloat() }
                val featureTensor = Tensor.fromBlob(toByteBuffer.toFloatArray(), longArrayOf)

                module?.forward(IValue.from(featureTensor))?.let {
                    val outputTensor = it.toTensor()
                    val scores = outputTensor.dataAsFloatArray
                    val maxScore = scores.maxOrNull()
                    val maxScoreIndex = scores.maxOf { it }
                    println("maxScore: ${maxScore}")
                    println("maxScoreIndex: ${maxScoreIndex}")
                    println("scores size: ${scores.size}")


                    postiveScore[positiveNegativeScoreIndex] = scores[0].absoluteValue
                    negativeScore[positiveNegativeScoreIndex] = scores[1].absoluteValue

                    // average the scores
                    val averagePositiveScore = postiveScore.maxOrNull()
                    val averageNegativeScore = negativeScore.minOrNull()


                    // only show two decimal places

                    if (positiveNegativeScoreIndex === 3){
                        positiveNegativeScoreIndex = 0
                        val msg = updatePercentagehandler.obtainMessage()
                        msg.obj = Pair(averagePositiveScore, averageNegativeScore)

                        updatePercentagehandler.sendMessage(msg)
                    } else {
                        positiveNegativeScoreIndex += 1
                    }

                }



            }

            mAudioRecord?.stop()
            mAudioRecord?.release()
            mAudioRecord = null
        }

        isRecording = true
        audioThread?.start()
    }

    fun stopRecording() {
        isRecording = false
        audioThread?.join()
        audioThread = null
    }


    // on destroy
    override fun onDestroy() {
        super.onDestroy()
        stopRecording()
    }

    override fun onPause() {
        super.onPause()
        stopRecording()
    }


    @Throws(IOException::class)
    fun assetFilePath(assetName: String?): String? {
        if (assetName == null) {
            throw IOException("Asset name cannot be null")
        }
        if (this.context == null) {
            throw IOException("Context cannot be null")
        }
        if (this.requireContext().filesDir == null) {
            throw IOException("Files directory cannot be null")
        }
        val file: File = File(this.requireContext().filesDir, assetName)
        if (file.exists() && file.length() > 0) {
            return file.absolutePath
        }
        this.requireContext().assets.open(assetName).use { `is` ->
            FileOutputStream(file).use { os ->
                val buffer = ByteArray(4 * 1024)
                var read: Int
                while (`is`.read(buffer).also { read = it } != -1) {
                    os.write(buffer, 0, read)
                }
                os.flush()
            }
            return file.absolutePath
        }
    }
}