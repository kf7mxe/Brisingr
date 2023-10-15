package com.kf7mxe.assistantdatacollections.ui.noiseCollection

import android.content.Context
import android.media.MediaPlayer
import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.recyclerview.widget.RecyclerView
import com.kf7mxe.assistantdatacollections.R
import java.io.File
import com.kf7mxe.assistantdatacollections.ui.noiseCollection.splitWavFile as SplitWavFile

class UnsplitNoiseRecordingRecyclerViewAdapter(val context:Context, val data: MutableList<String>) : RecyclerView.Adapter<UnsplitNoiseRecordingCardViewHolder>() {

    val mediaPlayer:MediaPlayer = MediaPlayer()
    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): UnsplitNoiseRecordingCardViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.unsplit_noise_recording_card, parent, false)
        return UnsplitNoiseRecordingCardViewHolder(view)
    }


    override fun onBindViewHolder(holder: UnsplitNoiseRecordingCardViewHolder, position: Int) {
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

        holder.splitButton.setOnClickListener {
            val unsplitFolderName = "unsplit-noise"
            val unsplitFileName = data[position]
            val splitNoiseSubFolderName = "split-noise"
            val splitNoiseSubfolder = File(context.filesDir, "split-noise")
            if (!splitNoiseSubfolder.exists()) {
                splitNoiseSubfolder.mkdir()
            }
            // create new folder in the subfolder with the same name as the file
            val splitFolderForUnsplitFile = File(splitNoiseSubfolder, data[position].replace(".wav", ""))
            if (!splitFolderForUnsplitFile.exists()) {
                splitFolderForUnsplitFile.mkdir()
            }

            val noiseFile = File(context.filesDir, "$unsplitFolderName/$unsplitFileName")
            SplitWavFile("$unsplitFolderName/$unsplitFileName", splitNoiseSubFolderName+"/"+ data[position].replace(".wav", ""), context)
            data.removeAt(position)
            notifyDataSetChanged()
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

