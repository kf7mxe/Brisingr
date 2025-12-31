package com.kf7mxe.brisingr

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.speech.RecognizerIntent
import android.view.View
import android.view.WindowManager
import com.kf7mxe.brisingr.MainActivity.Companion.dialog
import com.kf7mxe.brisingr.MainActivity.Companion.main
import com.lightningkite.kiteui.*
import com.lightningkite.kiteui.models.Theme
import com.lightningkite.kiteui.navigation.PageNavigator
import com.lightningkite.kiteui.reactive.*
import com.lightningkite.reactive.context.ReactiveContext

class VoiceAssistantActivity : KiteUiActivity() {

    
    companion object {
        val main = PageNavigator { AutoRoutes }
        val dialog = PageNavigator { AutoRoutes }
        const val VOICE_REQUEST_CODE = 1234

    }

    override val theme: ReactiveContext.() -> Theme
        get() = { appTheme() }

    override val mainNavigator: PageNavigator get() = main

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val layoutParams = window.attributes

        // Align to bottom
        layoutParams.gravity = android.view.Gravity.BOTTOM or android.view.Gravity.CENTER_HORIZONTAL

        // Width: Match Parent (Edge to Edge)
        layoutParams.width = WindowManager.LayoutParams.MATCH_PARENT

        // Height: Wrap Content (So it doesn't cover the top)
        // You can also set this to a specific pixel value if WRAP_CONTENT is too small
        layoutParams.height = WindowManager.LayoutParams.WRAP_CONTENT

        window.attributes = layoutParams

        // Optional: Add a slide-up animation if you have one defined in styles,
        // or just ensure the background is transparent.
        window.setBackgroundDrawableResource(R.drawable.bg_assistant_sheet)





        Throwable_report = { ex, ctx ->
            ex.printStackTrace2()
        }

        with(viewWriter) {
            app(main, dialog)
        }
    }

//    override fun onNewIntent(intent: Intent) {
//        super.onNewIntent(intent)
//        handleIntent(intent)
//    }

    private fun handleIntent(intent: Intent) {
        when (intent.action) {
            Intent.ACTION_ASSIST,
            "android.intent.action.VOICE_COMMAND",
            "android.speech.action.VOICE_SEARCH_HANDS_FREE",
            "android.speech.action.WEB_SEARCH" -> {
                // Handle voice assistant activation
                showAssistantInterface()
            }
        }
    }

    private fun showAssistantInterface() {
        // The assistant interface is already shown in onCreate
        // This can be used for any additional setup when assistant is activated
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        
        if (requestCode == VOICE_REQUEST_CODE && resultCode == Activity.RESULT_OK) {
            val results = data?.getStringArrayListExtra(RecognizerIntent.EXTRA_RESULTS)
            if (!results.isNullOrEmpty()) {
                val spokenText = results[0]
                processVoiceInput(spokenText)
            }
        }
    }
    
    private fun processVoiceInput(text: String) {
        // Handle the voice input
        // You can update the UI or process the command here
        println("Voice input: $text")
    }

    override fun finish() {
        super.finish()
        overridePendingTransition(0, 0)
    }
}