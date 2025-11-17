package com.kf7mxe.assistantdatacollections.ui.noiseCollection

import android.os.Bundle
import android.view.KeyEvent
import androidx.fragment.app.Fragment
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import com.kf7mxe.assistantdatacollections.databinding.FragmentUnsplitNoiseCollectionBinding
import com.kf7mxe.assistantdatacollections.R
import java.io.File

// TODO: Rename parameter arguments, choose names that match
// the fragment initialization parameters, e.g. ARG_ITEM_NUMBER
private const val ARG_PARAM1 = "param1"
private const val ARG_PARAM2 = "param2"

/**
 * A simple [Fragment] subclass.
 * Use the [UnsplitNoiseCollection.newInstance] factory method to
 * create an instance of this fragment.
 */
class UnsplitNoiseCollection : Fragment() {
    // TODO: Rename and change types of parameters
    private var param1: String? = null
    private var param2: String? = null

    private var _binding: FragmentUnsplitNoiseCollectionBinding? = null

    private var wavObj: WavClass? =null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        arguments?.let {
            param1 = it.getString(ARG_PARAM1)
            param2 = it.getString(ARG_PARAM2)
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        // Inflate the layout for this fragment

        _binding = FragmentUnsplitNoiseCollectionBinding.inflate(inflater, container, false)
        val recordingFiles = getAllRecordingFiles()
        val recyclerAdapter = UnsplitNoiseRecordingRecyclerViewAdapter(requireContext(), recordingFiles )
        _binding!!.unsplitNoiseCollectionRecyclerView.adapter = recyclerAdapter

        _binding!!.unsplitNoiseCollectionRecyclerView.layoutManager = androidx.recyclerview.widget.LinearLayoutManager(requireContext())


        wavObj = WavClass(null,requireContext(),_binding!!, recyclerAdapter)

        _binding!!.recordPlayButton.setOnClickListener {
            if (wavObj?.isRecording == true){
                wavObj?.stopRecording()
                _binding!!.recordPlayButton.setImageDrawable(resources.getDrawable(R.drawable.record))
            } else {
                context?.let { it1 -> wavObj?.startRecording(it1) }
                _binding!!.recordPlayButton.setImageDrawable(resources.getDrawable(R.drawable.baseline_stop_24))
            }

        }



        val root: View = _binding!!.root
        return root
    }

    public fun getAllRecordingFiles(): MutableList<String>{
        val subfolderName = "unsplit-noise"

        val subFolderSplitName = "split-noise"
        val subFolderSplit = File(context?.filesDir, subFolderSplitName)
        if (!subFolderSplit.exists()) {
            subFolderSplit.mkdir()
        }
        val allSubFolderSplitDirectories = subFolderSplit.list() ?: emptyArray()


        val subfolder = File(context?.filesDir, subfolderName)

        val allFiles: Array<String> = subfolder.list() ?: emptyArray()

        val recordingFiles = mutableListOf<String>()
        for (i in allFiles.indices) {
            if (allFiles[i].endsWith(".wav")) {
                recordingFiles.add(allFiles[i])
            }
        }

        // remove file if there is a directory in the split folder with the same name
        for (i in allSubFolderSplitDirectories.indices) {
            for (j in recordingFiles.indices) {

                if (recordingFiles.size < j) {

                    if (allSubFolderSplitDirectories[i] == recordingFiles[j].replace(".wav", "")) {
                        recordingFiles.removeAt(j)
                    }
                }
            }
        }
        return recordingFiles
    }


    public fun myOnKeyDown(keyCode: Int) {
        if (keyCode == KeyEvent.KEYCODE_VOLUME_UP) {
            context?.let {
                    it1 -> wavObj?.startRecording(it1) }
            _binding?.recordPlayButton?.setImageDrawable(resources.getDrawable(R.drawable.baseline_stop_24))
        }
        if (keyCode == KeyEvent.KEYCODE_VOLUME_DOWN) {
            context?.let {
                    it1 -> wavObj?.stopRecording() }
            _binding?.recordPlayButton?.setImageDrawable(resources.getDrawable(R.drawable.baseline_stop_24))
        }
    }

    companion object {
        /**
         * Use this factory method to create a new instance of
         * this fragment using the provided parameters.
         *
         * @param param1 Parameter 1.
         * @param param2 Parameter 2.
         * @return A new instance of fragment UnsplitNoiseCollection.
         */
        // TODO: Rename and change types and number of parameters
        @JvmStatic
        fun newInstance(param1: String, param2: String) =
            UnsplitNoiseCollection().apply {
                arguments = Bundle().apply {
                    putString(ARG_PARAM1, param1)
                    putString(ARG_PARAM2, param2)
                }
            }
    }
}