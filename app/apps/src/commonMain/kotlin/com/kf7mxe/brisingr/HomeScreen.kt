package com.kf7mxe.brisingr

import com.lightningkite.kiteui.Routable
import com.lightningkite.kiteui.navigation.Page
import com.lightningkite.kiteui.navigation.pageNavigator
import com.lightningkite.kiteui.reactive.*
import com.lightningkite.kiteui.views.*
import com.lightningkite.kiteui.views.direct.*
import com.lightningkite.lightningserver.*
import com.lightningkite.lightningserver.sessions.*
import com.lightningkite.reactive.context.*
import com.lightningkite.reactive.core.*
import com.lightningkite.reactive.extensions.*
import com.lightningkite.reactive.lensing.*
import com.lightningkite.readable.*
import com.lightningkite.services.data.*
import com.lightningkite.services.database.*
import com.lightningkite.services.files.*
import com.kf7mxe.brisingr.sdk.currentSession
import com.kf7mxe.brisingr.sdk.sessionToken
import kotlin.uuid.Uuid
import kotlinx.coroutines.launch
import com.lightningkite.kiteui.Platform
import com.lightningkite.kiteui.current
import com.lightningkite.kiteui.models.rem

@Routable("/")
class HomePage: Page {
    override val title: Reactive<String> get() = Constant("Home")
    override fun ViewWriter.render() {

        reactive {
//            if(currentSession() == null)
//                pageNavigator.reset(LandingPage())
        }

       // Voice assistant trigger button at bottom
        atBottom.sizeConstraints(minHeight = 8.rem).card.col {
            col {
                gap = 0.5.rem
                
                // Assistant activation button
                centered.button {
                    text("🎤 Tap to activate Assistant")
                    onClick {
                        // Show the assistant bottom sheet
                        showAssistantBottomSheet()
                    }
                }
                
                // Quick text input
                sizeConstraints(minWidth = 20.rem).fieldTheme.row {
                    textArea {
                        hint = "Type your command..."
                    }
                }
            }
        }
    }
    
    // Function to show assistant bottom sheet
    private fun showAssistantBottomSheet() {
        // Navigate to a dialog page that shows the assistant
//        pageNavigator.dialog(AssistantBottomSheetPage())
    }
    
    // Platform-specific function to open assistant settings
    private fun openAssistantSettings() {
        // This will be implemented in the Android-specific code
    }
}

// Assistant Bottom Sheet Page
@Routable("/assistant")
class AssistantBottomSheetPage : Page {
    override val title: Reactive<String> get() = Constant("Voice Assistant")
    
    override fun ViewWriter.render() {

    }
}

