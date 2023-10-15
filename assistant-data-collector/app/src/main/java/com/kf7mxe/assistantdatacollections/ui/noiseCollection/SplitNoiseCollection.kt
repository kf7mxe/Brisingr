package com.kf7mxe.assistantdatacollections.ui.noiseCollection

import android.os.Bundle
import androidx.fragment.app.Fragment
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import com.kf7mxe.assistantdatacollections.databinding.FragmentSplitNoiseCollectionBinding
import java.io.File

// TODO: Rename parameter arguments, choose names that match
// the fragment initialization parameters, e.g. ARG_ITEM_NUMBER
private const val ARG_PARAM1 = "param1"
private const val ARG_PARAM2 = "param2"

/**
 * A simple [Fragment] subclass.
 * Use the [SplitNoiseCollection.newInstance] factory method to
 * create an instance of this fragment.
 */
class SplitNoiseCollection : Fragment() {
    // TODO: Rename and change types of parameters
    private var param1: String? = null
    private var param2: String? = null

    public var binding: FragmentSplitNoiseCollectionBinding? = null



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
        binding = FragmentSplitNoiseCollectionBinding.inflate(inflater, container, false)
        val subfolderName = "split-noise"
        val subfolder = File(context?.filesDir, subfolderName)
        val allFolders = subfolder.list() ?: emptyArray()
        val allFiles = mutableListOf<String>()
        for (i in allFolders.indices) {
            val folder = File(subfolder, allFolders[i])
            val files = folder.list() ?: emptyArray()
            for (j in files.indices) {
                allFiles.add(folder.name+"/"+files[j])
            }
        }

        val recyclerViewAdapter = SplitNoiseRecordingRecyclerViewAdapter(requireContext(), allFiles)
        binding!!.splitNoiseCollectionRecyclerView.adapter = recyclerViewAdapter
        binding!!.splitNoiseCollectionRecyclerView.layoutManager = androidx.recyclerview.widget.LinearLayoutManager(requireContext())

        val root: View = binding!!.root
        return root
            }

    companion object {
        /**
         * Use this factory method to create a new instance of
         * this fragment using the provided parameters.
         *
         * @param param1 Parameter 1.
         * @param param2 Parameter 2.
         * @return A new instance of fragment SplitNoiseCollection.
         */
        // TODO: Rename and change types and number of parameters
        @JvmStatic
        fun newInstance(param1: String, param2: String) =
            SplitNoiseCollection().apply {
                arguments = Bundle().apply {
                    putString(ARG_PARAM1, param1)
                    putString(ARG_PARAM2, param2)
                }
            }
    }
}