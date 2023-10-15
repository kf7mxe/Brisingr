package com.kf7mxe.assistantdatacollections.ui.wakeWord

import android.media.MediaRecorder
import android.os.Bundle
import android.view.KeyEvent
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import androidx.lifecycle.ViewModelProvider
import com.kf7mxe.assistantdatacollections.R
import com.kf7mxe.assistantdatacollections.databinding.FragmentWakeWordCollectBinding
import java.io.File


class WakeWordCollectFragment : Fragment() {

    private var _binding: FragmentWakeWordCollectBinding? = null

    private var mediaRecorder: MediaRecorder? = null
    private var wavObj: WaveRecordOneSecond? =null

    // This property is only valid between onCreateView and
    // onDestroyView.
    private val binding get() = _binding!!

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        val homeViewModel =
            ViewModelProvider(this)[HomeViewModel::class.java]

        _binding = FragmentWakeWordCollectBinding.inflate(inflater, container, false)
        val root: View = binding.root


        val subfolderName = "positive"

// Create the subfolder if it doesn't exist
        val subfolder = File(context?.filesDir, subfolderName)

        val allFiles: Array<String> = subfolder.list() ?: emptyArray()

//        val allFiles:Array<String> = context?.fileList() as Array<String>
        // only put the files that end in .wav into a new array from the recordingFiles array
        val recordingFiles = mutableListOf<String>()
        for (i in allFiles.indices) {
            if (allFiles[i].endsWith(".wav")) {
                recordingFiles.add(allFiles[i])
            }
        }

        val recyclerAdapter = WakeWordRecordingRecyclerViewAdapter(requireContext(), recordingFiles)
        binding.wakeWordRecyclerView.adapter = recyclerAdapter
        binding.wakeWordRecyclerView.layoutManager = androidx.recyclerview.widget.LinearLayoutManager(requireContext())


        wavObj = WaveRecordOneSecond(null,requireContext(),binding, recyclerAdapter)

        binding.recordPlayButton.setOnClickListener {
            context?.let { it1 -> wavObj?.startRecording(it1) }
            binding.recordPlayButton.setImageDrawable(resources.getDrawable(R.drawable.baseline_stop_24))
        }




//        val recyclerAdapter = WakeWordRecordingCardViewHolder()

        // run startRecordingForOneSecond() physical volume up button is pressed
        // run stopRecording() physical volume down button is pressed

        return root
    }




    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }

    public fun myOnKeyDown(keyCode: Int) {
        if (keyCode == KeyEvent.KEYCODE_VOLUME_UP) {
            context?.let {
                it1 -> wavObj?.startRecording(it1) }
            binding.recordPlayButton.setImageDrawable(resources.getDrawable(R.drawable.baseline_stop_24))

        }
        if (keyCode == KeyEvent.KEYCODE_VOLUME_DOWN) {
            context?.let {
                    it1 -> wavObj?.startRecording(it1) }
            binding.recordPlayButton.setImageDrawable(resources.getDrawable(R.drawable.baseline_stop_24))
        }
    }

//    fun startRecordingForOneSecond() {
//        mediaRecorder = MediaRecorder().apply {
//            setAudioSource(MediaRecorder.AudioSource.MIC)
//            setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)
//            setAudioEncoder(MediaRecorder.AudioEncoder.AMR_NB)
//            setOutputFile("your_output_file_path.3gp")
//
//            try {
//                prepare()
//            } catch (e: IOException) {
//                e.printStackTrace()
//            }
//
//            start()
//
//            Handler().postDelayed({
//                stopRecording()
//            }, 1000) // Stop recording after 1 second
//        }
//    }
//
//    fun stopRecording() {
//        mediaRecorder?.apply {
//            stop()
//            release()
//        }
//        mediaRecorder = null
//    }
}