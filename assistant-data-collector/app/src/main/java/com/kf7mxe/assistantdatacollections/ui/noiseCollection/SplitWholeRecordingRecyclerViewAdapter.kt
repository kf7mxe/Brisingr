package com.kf7mxe.assistantdatacollections.ui.noiseCollection

import android.content.Context
import android.media.MediaPlayer
import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.recyclerview.widget.RecyclerView
import com.kf7mxe.assistantdatacollections.R
import java.io.File
import com.kf7mxe.assistantdatacollections.ui.noiseCollection.splitWavFile as SplitWavFile

class SplitWholeRecordingRecyclerViewAdapter(val context:Context, val data: MutableList<String>) : RecyclerView.Adapter<SplitNoiseRecordingCardViewHolder>() {

    val mediaPlayer:MediaPlayer = MediaPlayer()
    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): SplitNoiseRecordingCardViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.wake_word_recording_card, parent, false)
        return SplitNoiseRecordingCardViewHolder(view)
    }


    override fun onBindViewHolder(holder: SplitNoiseRecordingCardViewHolder, position: Int) {
        holder.fileName.text = data[position]
        holder.playButton.setOnClickListener {
            holder.playButton.setImageDrawable(context.getDrawable(R.drawable.baseline_stop_24))

            val subFolderName = "unsplit-noise"
            val fileName = data[position]


            val file = File(context.filesDir, "$subFolderName/$fileName")

            val uri = file.toURI()
            mediaPlayer.reset()
            mediaPlayer.setDataSource(uri.toString())
            mediaPlayer.prepare()
            mediaPlayer.start()
            mediaPlayer.setOnCompletionListener {
                holder.playButton.setImageDrawable(context.getDrawable(R.drawable.baseline_play_circle_24))
                mediaPlayer.reset()
            }

        }

        holder.deleteButton.setOnClickListener {
            val subFolderName = "unsplit-noise"
            val fileName = data[position]
                val file = File(context.filesDir, "$subFolderName/$fileName")
                    file.delete()
            .apply {
                data.removeAt(position)
                notifyDataSetChanged()
            }
        }


    }

    override fun getItemCount(): Int {
        return data.size
    }
}

