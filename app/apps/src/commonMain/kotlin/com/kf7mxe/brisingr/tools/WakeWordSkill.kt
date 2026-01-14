package com.kf7mxe.brisingr.tools

import com.kf7mxe.brisingr.wakeword.WakeWordState
import com.kf7mxe.brisingr.wakeword.WakeWordController
import com.lightningkite.kiteui.models.rem
import com.lightningkite.kiteui.views.*
import com.lightningkite.kiteui.views.direct.*
import com.lightningkite.reactive.context.reactiveScope
import com.lightningkite.reactive.core.AppScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

class WakeWordSkill : Skill {
    override val definition = SkillDefinition.forType(SkillType.WakeWord)

    override suspend fun processMessage(message: Message): Message? {
        val command = message.content.lowercase().trim()

        return when {
            command.contains("enable") || command.contains("start") || command.contains("on") -> {
                WakeWordController.start()
                Message(
                    conversationId = message.conversationId,
                    role = MessageRole.Assistant,
                    content = "Wake word detection enabled."
                )
            }
            command.contains("disable") || command.contains("stop") || command.contains("off") -> {
                WakeWordController.stop()
                Message(
                    conversationId = message.conversationId,
                    role = MessageRole.Assistant,
                    content = "Wake word detection disabled."
                )
            }
            command.contains("status") -> {
                val status = if (WakeWordState.serviceRunning.value) "running" else "stopped"
                val threshold = (WakeWordState.threshold.value * 100).toInt()
                Message(
                    conversationId = message.conversationId,
                    role = MessageRole.Assistant,
                    content = "Wake word detection is $status.\nThreshold: $threshold%\nCooldown: ${WakeWordState.cooldownSeconds.value}s"
                )
            }
            command.contains("help") -> {
                Message(
                    conversationId = message.conversationId,
                    role = MessageRole.Assistant,
                    content = """
                        Wake Word Commands:
                        - enable/start/on - Enable wake word detection
                        - disable/stop/off - Disable wake word detection
                        - status - Show current status
                        - help - Show this help

                        Use the toggle above to control detection.
                    """.trimIndent()
                )
            }
            else -> Message(
                conversationId = message.conversationId,
                role = MessageRole.Assistant,
                content = "I can help you control wake word detection. Try 'enable', 'disable', 'status', or 'help'."
            )
        }
    }

    override fun ViewWriter.renderControls() {
        col {
            gap = 0.5.rem
            padding = 0.5.rem

            // Availability check
            subtext {
                ::content {
                    if (WakeWordController.isAvailable())
                        "Control wake word detection"
                    else
                        "Wake word detection is not available on this platform"
                }
            }

            // Service Status Card
            card.col {
                gap = 0.5.rem

                row {
                    expanding.text("Service Status")
                    text {
                        ::content {
                            if (WakeWordState.serviceRunning()) "Running" else "Stopped"
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
            }.shownWhen { WakeWordController.isAvailable() }

            // Detection Card
            card.col {
                gap = 0.5.rem

                // Probability display
                row {
                    text("Detection:")
                    expanding.text {
                        ::content { "${(WakeWordState.probability() * 100).toInt()}%" }
                    }
                }

                // Threshold display
                row {
                    text("Threshold:")
                    expanding.text {
                        ::content { "${(WakeWordState.threshold() * 100).toInt()}%" }
                    }
                }

                // Detection indicator
                important.centered.sizeConstraints(height = 3.rem).frame {
                    text {
                        ::content {
                            if (WakeWordState.detected()) "WAKE WORD DETECTED!" else "Listening..."
                        }
                    }
                }.shownWhen { WakeWordState.serviceRunning() }
            }.shownWhen { WakeWordController.isAvailable() && WakeWordState.serviceRunning() }

            // Error display
            danger.card.col {
                gap = 0.5.rem
                text("Error")
                text {
                    ::content { WakeWordState.errorMessage() ?: "" }
                }
            }.shownWhen { WakeWordState.errorMessage() != null }

            // Test button
            button {
                centered.text("Test Detection")
                onClick {
                    WakeWordState.detected.value = true
                    AppScope.launch {
                        delay(2000)
                        WakeWordState.detected.value = false
                    }
                }
            }.shownWhen { WakeWordController.isAvailable() }
        }
    }
}
