package com.kf7mxe.brisingr.wakeword

import com.lightningkite.kiteui.Routable
import com.lightningkite.kiteui.models.*
import com.lightningkite.kiteui.navigation.Page
import com.lightningkite.kiteui.views.*
import com.lightningkite.kiteui.views.direct.*
import com.lightningkite.reactive.core.Signal
import com.lightningkite.reactive.context.reactiveScope
import kotlinx.coroutines.launch
import com.lightningkite.reactive.core.AppScope

@Routable("wake-word")
object WakeWordPage : Page {
    override fun ViewWriter.render() {
        scrolling.col {
            gap = 1.rem
            padding = 1.rem
            
            h1("Wake Word Detection")
            
            subtext { 
                ::content { 
                    if (WakeWordController.isAvailable()) 
                        "Control wake word detection on this device"
                    else 
                        "Wake word detection is not available on this platform"
                }
            }
            
            // Service Status Card
            card.col {
                gap = 0.5.rem
                
                row {
                    expanding.h3("Service Status")
                    text {
                        ::content {
                            if (WakeWordState.serviceRunning()) "🟢 Running" else "⚪ Stopped"
                        }
                    }
                }
                
                // Enable/Disable Toggle
                row {
                    expanding.text("Enable Wake Word")
                    switch {
                        checked bind WakeWordState.enabled
                        reactiveScope {
                            if (WakeWordState.enabled()) {
                                WakeWordController.start()
                            } else {
                                WakeWordController.stop()
                            }
                        }
                    }
                }
            }
            
            // Detection Probability Card
            card.col {
                gap = 0.5.rem
                
                h3("Detection")
                
                // Probability bar
                row {
                    text("Probability:")
                    expanding.frame {
                        // Background bar
                        sizeConstraints(height = 1.5.rem).card.frame { }
                        
                        // Filled portion
                        atStart.sizeConstraints(height = 1.5.rem).frame {
                            reactiveScope {
                                val prob = WakeWordState.probability()
                                // Visual width based on probability
                            }
                        }
                    }
                    text {
                        ::content { "${(WakeWordState.probability() * 100).toInt()}%" }
                    }
                }
                
                // Threshold slider
                row {
                    text("Threshold:")
                    expanding.text {
                        ::content { "${(WakeWordState.threshold() * 100).toInt()}%" }
                    }
                }
                
                // Detection indicator
                important.centered.sizeConstraints(height = 4.rem).frame {
                    text {
                        ::content {
                            if (WakeWordState.detected()) "🎤 WAKE WORD DETECTED!" else "Listening..."
                        }
                    }
                }.shownWhen { WakeWordState.serviceRunning() }
            }
            
            // Error display
            danger.card.col {
                gap = 0.5.rem
                h3("Error")
                text {
                    ::content { WakeWordState.errorMessage() ?: "" }
                }
            }.shownWhen { WakeWordState.errorMessage() != null }
            
            // Settings Card
            card.col {
                gap = 0.5.rem
                h3("Settings")
                
                row {
                    text("Cooldown (seconds):")
                    expanding.text {
                        ::content { WakeWordState.cooldownSeconds().toString() }
                    }
                }
            }
            
            // Test button
            button {
                centered.text("Simulate Detection")
                onClick {
                    WakeWordState.detected.value = true
                    AppScope.launch {
                        kotlinx.coroutines.delay(2000)
                        WakeWordState.detected.value = false
                    }
                }
            }
        }
    }
}
