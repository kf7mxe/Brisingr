package com.kf7mxe.brisingr.tools.ui

import com.kf7mxe.brisingr.SelectedBackgroundSetToSpecificColor
import com.kf7mxe.brisingr.attachment
import com.kf7mxe.brisingr.tools.*
import com.lightningkite.kiteui.models.Icon
import com.lightningkite.kiteui.models.rem
import com.lightningkite.kiteui.views.*
import com.lightningkite.kiteui.views.direct.*
import com.lightningkite.kiteui.views.l2.icon
import com.lightningkite.reactive.core.AppScope
import com.lightningkite.reactive.core.remember
import kotlinx.coroutines.launch
import kotlin.time.Clock
import kotlin.uuid.Uuid

fun ViewWriter.skillsInputArea() {
    col {
        card.col {
            gap = 0.5.rem
            padding = 0.5.rem

            dynamicTheme {
                val color = SkillsState.currentColor()
                SelectedBackgroundSetToSpecificColor[color]
            }


            // Row 1: Previous chats button
            row {

            }

            // Row 2: Current skill chip + available skills
            row {
                gap = 0.5.rem

                // Current skill indicator
                currentSkillChip()

                separator()

                // Scrollable skill chips
                scrollingHorizontally.row {
                    gap = 0.25.rem
                    SkillsState.availableSkills.forEach { skill ->
                        skillChip(skill)
                    }
                }
            }

            // Pending attachments preview
            row {
                gap = 0.25.rem
                val attachments = remember { SkillsState.pendingAttachments() }
                forEach(attachments) { attachment ->
                    card.row {
                        gap = 0.25.rem
                        text(attachment.fileName)
                        button {
                            text("x")
                            onClick {
                                SkillsState.removeAttachment(attachment._id)
                            }
                        }
                    }

                }
            }.shownWhen { SkillsState.pendingAttachments().isNotEmpty() }




            // Row 3: Text input with multimedia + send
            fieldTheme.row {
                gap = 0.25.rem
                menuButton {
                    icon(Icon.attachment,"attachments")
                    opensMenu {
                        // Multimedia buttons
                        button {
                            text("Img")
                            onClick {
                                // TODO: Image picker
                            }
                        }
                        button {
                            text("Mic")
                            onClick {
                                // TODO: Audio recorder
                            }
                        }
                        button {
                            text("File")
                            onClick {
                                // TODO: File picker
                            }
                        }
                    }
                }


                // Text area
                expanding.textArea {
                    content bind SkillsState.inputText
                    hint = "Type a message..."
                }

                // Send button
                important.button {
                    text("Send")
                    onClick {
                        sendMessage()
                    }
                }
            }


            // Error display
            shownWhen {
                println("DEBUG SkillsState.errorMessage() ${SkillsState.errorMessage() != null}")
                SkillsState.errorMessage() != null
            }.danger.card.row {
                expanding.text {
                    ::content { SkillsState.errorMessage() ?: "" }
                }
                button {
                    text("Dismiss")
                    onClick { SkillsState.clearError() }
                }
            }
        }
    }
}

private fun sendMessage() {
    val content = SkillsState.inputText.value.trim()
    if (content.isEmpty() && SkillsState.pendingAttachments.value.isEmpty()) return

    AppScope.launch {
        try {
            SkillsState.isLoading.value = true

            // Ensure we have a conversation
            var conversationId = SkillsState.currentConversationId.value
            if (conversationId == null) {
                conversationId = Uuid.random()
                SkillsState.currentConversationId.value = conversationId
            }

            // Create user message
            val userMessage = Message(
                conversationId = conversationId,
                role = MessageRole.User,
                content = content,
                attachments = SkillsState.pendingAttachments.value,
                createdAt = Clock.System.now()
            )

            // Add to messages list
            SkillsState.currentMessages.value = SkillsState.currentMessages.value + userMessage

            // Clear input
            SkillsState.inputText.value = ""
            SkillsState.clearAttachments()

            // Process with skill
            val skill = SkillRegistry.getSkill(SkillsState.currentSkillType.value)
            val response = skill.processMessage(userMessage)

            if (response != null) {
                // Add assistant response
                SkillsState.currentMessages.value = SkillsState.currentMessages.value + response
            }

        } catch (e: Exception) {
            SkillsState.errorMessage.value = e.message ?: "An error occurred"
        } finally {
            SkillsState.isLoading.value = false
        }
    }
}
