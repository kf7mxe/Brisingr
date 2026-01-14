package com.kf7mxe.brisingr.tools.ui

import com.kf7mxe.brisingr.SelectedBackgroundSetToSpecificColor
import com.kf7mxe.brisingr.tools.Message
import com.kf7mxe.brisingr.tools.MessageRole
import com.kf7mxe.brisingr.tools.AttachmentType
import com.lightningkite.kiteui.models.CardSemantic
import com.lightningkite.kiteui.models.Color
import com.lightningkite.kiteui.models.ImageRemote
import com.lightningkite.kiteui.models.rem
import com.lightningkite.kiteui.views.*
import com.lightningkite.kiteui.views.direct.*

fun ViewWriter.messageBubble(message: Message, skillColor: Color) {
    val isUser = message.role == MessageRole.User

    row {
        gap = 0.5.rem

        // Push user messages to rightthemeChoice = if(isUser)   else null
        if (!isUser) {
            expanding.frame { }
        }

        card.col {
            gap = 0.25.rem
            sizeConstraints(maxWidth = 20.rem)

            dynamicTheme {
                if(isUser){
                    SelectedBackgroundSetToSpecificColor[skillColor]
                } else {
                    CardSemantic
                }
            }

            // Message content
            text(message.content)

            // Attachments
            if (message.attachments.isNotEmpty()) {
                col {
                    gap = 0.25.rem
                    message.attachments.forEach { attachment ->
                        when (attachment.type) {
                            AttachmentType.Image -> {
                                sizeConstraints(maxHeight = 8.rem).image {
                                    source = ImageRemote(attachment.url)
                                    description = attachment.fileName
                                }
                            }
                            AttachmentType.Audio -> {
                                row {
                                    gap = 0.25.rem
                                    text("Audio:")
                                    text(attachment.fileName)
                                }
                            }
                            AttachmentType.File -> {
                                row {
                                    gap = 0.25.rem
                                    text("File:")
                                    text(attachment.fileName)
                                }
                            }
                        }
                    }
                }
            }
        }

        // Push assistant messages to left
        if (isUser) {
            expanding.frame { }
        }
    }
}
