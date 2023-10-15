package com.kf7mxe.assistantdatacollections.ui.noiseCollection

import androidx.fragment.app.Fragment
import androidx.fragment.app.FragmentManager
import androidx.lifecycle.Lifecycle
import androidx.viewpager2.adapter.FragmentStateAdapter

private val NUM_TABS = 3

public class ViewPagerAdapter(fragmentManager: FragmentManager, lifecycle: Lifecycle) :
    FragmentStateAdapter(fragmentManager, lifecycle) {

    override fun getItemCount(): Int {
        return NUM_TABS
    }

    override fun createFragment(position: Int): Fragment {
        when (position) {
            0 -> return UnsplitNoiseCollection()
            1 -> return SplitNoiseCollection()
        }
        return SplitWholeNoiseCollection()
    }
}