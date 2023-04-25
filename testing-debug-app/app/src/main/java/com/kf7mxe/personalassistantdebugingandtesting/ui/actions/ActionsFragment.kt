package com.kf7mxe.personalassistantdebugingandtesting.ui.actions

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.fragment.app.Fragment
import androidx.lifecycle.ViewModelProvider
import com.example.personalassistantdebugandtesting.databinding.FragmentActionsBinding
import com.kf7mxe.personalassistantdebugingandtesting.ui.naturalLanguage.NaturalLanguageViewModel

class ActionsFragment: Fragment() {
    private var _binding: FragmentActionsBinding? = null

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

        _binding = FragmentActionsBinding.inflate(inflater, container, false)
        val root: View = binding.root

        val textView: TextView = binding.textNotifications
        naturalLanguageViewModel.text.observe(viewLifecycleOwner) {
            textView.text = it
        }
        return root
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}