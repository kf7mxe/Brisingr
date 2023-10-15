package com.kf7mxe.assistantdatacollections.ui.noiseCollection

import android.os.Bundle
import android.view.KeyEvent
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.fragment.app.Fragment
import androidx.lifecycle.ViewModelProvider
import com.google.android.material.tabs.TabLayoutMediator
import com.kf7mxe.assistantdatacollections.R
import com.kf7mxe.assistantdatacollections.databinding.FragmentNoiseCollectionBinding

class NoiseCollectionFragment : Fragment() {

    private var _binding: FragmentNoiseCollectionBinding? = null


    // This property is only valid between onCreateView and
    // onDestroyView.
    private val binding get() = _binding!!

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        val noiseCollectionViewModel =
            ViewModelProvider(this).get(NoiseCollectionViewModel::class.java)

        _binding = FragmentNoiseCollectionBinding.inflate(inflater, container, false)
        val root: View = binding.root

        val viewPager = binding.viewPager
        val tabLayout = binding.tabLayout

        val adapter = ViewPagerAdapter(childFragmentManager, lifecycle)
        viewPager.adapter = adapter


        TabLayoutMediator(tabLayout, viewPager) { tab, position ->
            when (position) {
                0 -> tab.text = "Unsplit"
                1 -> tab.text = "Split"
                2 -> tab.text = "Split whole"
            }
        }.attach()

        return root
    }




    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}