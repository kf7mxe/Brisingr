package com.kf7mxe.personalassistantdegugandtesting.ui.actions

import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel

class ActionsViewModel : ViewModel() {
    private val _text = MutableLiveData<String>().apply {
        value = "This is Action Fragment"
    }
    val text: LiveData<String> = _text
}