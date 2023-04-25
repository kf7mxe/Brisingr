package com.kf7mxe.personalassistantdebugingandtesting.ui.textToSpeech

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.fragment.app.Fragment
import androidx.lifecycle.ViewModelProvider
import com.example.personalassistantdebugandtesting.databinding.FragmentTextToSpeechBinding
import com.kf7mxe.personalassistantdebugingandtesting.ui.naturalLanguage.NaturalLanguageViewModel

class TextToSpeechFragment: Fragment() {
    private var _binding: FragmentTextToSpeechBinding? = null

    // This property is only valid between onCreateView and
    // onDestroyView.
    private val binding get() = _binding!!

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        val naturalLanguageViewModel =
            ViewModelProvider(this).get(NaturalLanguageViewModel::class.java)

        _binding = FragmentTextToSpeechBinding.inflate(inflater, container, false)
        val root: View = binding.root

//        val textView: TextView = binding.textNotifications
//        naturalLanguageViewModel.text.observe(viewLifecycleOwner) {
//            textView.text = it
//        }
        return root
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}