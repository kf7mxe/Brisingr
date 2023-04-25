package com.kf7mxe.personalassistantdebugingandtesting.ui.speechRecognition

import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.speech.RecognizerIntent
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.TextView
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.lifecycle.ViewModelProvider
import com.example.personalassistantdebugandtesting.databinding.FragmentSpeechRecognitionBinding
import java.util.Locale
import android.provider.Settings
import android.speech.RecognitionListener
import android.speech.SpeechRecognizer
import android.widget.Toast


class SpeechRecognitionFragment : Fragment() {

    private var _binding: FragmentSpeechRecognitionBinding? = null
    lateinit var outputText: TextView
    lateinit var micIcon: ImageView

    private val REQUEST_CODE_SPEECH_INPUT = 1

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

        outputText = binding.textReecognitionResult
        micIcon = binding.mic
        micIcon.setOnClickListener {
            checkAudioPermission()
            // changing the color of mic icon, which
            // indicates that it is currently listening
//            micIcon.setColorFilter(ContextCompat.getColor(this, R.color.mic_enabled_color)) // #FF0E87E7
            startSpeechToText()
        }



//        val textView: TextView = binding.textHome
//        speechRecognitionViewModel.text.observe(viewLifecycleOwner) {
//            textView.text = it
//        }
        return root
    }



    private fun checkAudioPermission() {
        if(Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {  // M = 23
            if(this.context?.let { ContextCompat.checkSelfPermission(it, "android.permission.RECORD_AUDIO") } != PackageManager.PERMISSION_GRANTED) {
                // this will open settings which asks for permission
                val intent = Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS, Uri.parse("package:com.kf7mxe.personalassistantdegugandtesting"))
                startActivity(intent)
                Toast.makeText(this.context, "Allow Microphone Permission", Toast.LENGTH_SHORT).show()
            }
        }
    }


    private fun startSpeechToText() {
        val speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this.context)
        val speechRecognizerIntent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH)
        speechRecognizerIntent.putExtra(
            RecognizerIntent.EXTRA_LANGUAGE_MODEL,
            RecognizerIntent.LANGUAGE_MODEL_FREE_FORM
        )
        speechRecognizerIntent.putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault())

        speechRecognizer.setRecognitionListener(object : RecognitionListener {
            override fun onReadyForSpeech(bundle: Bundle?) {}
            override fun onBeginningOfSpeech() {}
            override fun onRmsChanged(v: Float) {}
            override fun onBufferReceived(bytes: ByteArray?) {}
            override fun onEndOfSpeech() {}
            override fun onError(i: Int) {}

            override fun onResults(bundle: Bundle) {
                val result = bundle.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                if (result != null) {
                    // result[0] will give the output of speech
                    outputText.text = result[0]
                }
            }
            override fun onPartialResults(bundle: Bundle) {}
            override fun onEvent(i: Int, bundle: Bundle?) {}
        })
        // starts listening ...
        speechRecognizer.startListening(speechRecognizerIntent)
    }
    // use built in voice recognition to get text

// start speech recognition with silero speech to text model
private fun startSileroSpeechToText() {

}


    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}