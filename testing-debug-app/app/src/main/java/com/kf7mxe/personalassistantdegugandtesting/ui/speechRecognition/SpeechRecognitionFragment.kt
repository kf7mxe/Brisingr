package com.kf7mxe.personalassistantdegugandtesting.ui.speechRecognition

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.fragment.app.Fragment
import androidx.lifecycle.ViewModelProvider
import com.example.personalassistantdegugandtesting.databinding.FragmentSpeechRecognitionBinding

class SpeechRecognitionFragment : Fragment() {

    private var _binding: FragmentSpeechRecognitionBinding? = null

    // This property is only valid between onCreateView and
    // onDestroyView.
    private val binding get() = _binding!!

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        val speechRecognitionViewModel =
            ViewModelProvider(this).get(SpeechRecognitionViewModel::class.java)

        _binding = FragmentSpeechRecognitionBinding.inflate(inflater, container, false)
        val root: View = binding.root

        val textView: TextView = binding.textHome
        speechRecognitionViewModel.text.observe(viewLifecycleOwner) {
            textView.text = it
        }
        return root
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}