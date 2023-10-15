package com.kf7mxe.assistantdatacollections.ui.noiseCollection

import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import com.kf7mxe.assistantdatacollections.R

class SplitNoiseRecordingCardViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
    val fileName: TextView = itemView.findViewById(R.id.recordingName)
    val playButton: ImageView = itemView.findViewById(R.id.playButton)
    val deleteButton: ImageView = itemView.findViewById(R.id.deleteRecordingButton)
    val view = itemView
}