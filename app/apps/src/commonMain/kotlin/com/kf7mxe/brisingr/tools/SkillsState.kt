package com.kf7mxe.brisingr.tools

import com.lightningkite.kiteui.models.Color
import com.lightningkite.reactive.core.Signal
import com.lightningkite.reactive.core.Reactive
import com.lightningkite.reactive.core.remember
import kotlin.uuid.Uuid

object SkillsState {
    /** Currently selected skill type */
    val currentSkillType = Signal(SkillType.GeneralAssistant)

    /** Current skill definition (derived) */
    val currentSkill: Reactive<SkillDefinition> = remember {
        SkillDefinition.forType(currentSkillType())
    }

    /** Current skill's theme color */
    val currentColor: Reactive<Color> = remember {
        Color.fromHexString(currentSkill().colorHex)
    }

    /** Current conversation ID for active skill */
    val currentConversationId = Signal<Uuid?>(null)

    /** Messages for current conversation */
    val currentMessages = Signal<List<Message>>(emptyList())

    /** List of conversations for current skill */
    val skillConversations = Signal<List<Conversation>>(emptyList())

    /** Input text for message composition */
    val inputText = Signal("")

    /** Pending attachments for next message */
    val pendingAttachments = Signal<List<MessageAttachment>>(emptyList())

    /** Loading state */
    val isLoading = Signal(false)

    /** Error state */
    val errorMessage = Signal<String?>(null)

    /** Available skills list */
    val availableSkills: List<SkillDefinition> = SkillDefinition.allSkills

    /** Switch to a different skill */
    fun selectSkill(skillType: SkillType) {
        currentSkillType.value = skillType
        currentConversationId.value = null
        currentMessages.value = emptyList()
        skillConversations.value = emptyList()
    }

    /** Add attachment to pending list */
    fun addAttachment(attachment: MessageAttachment) {
        pendingAttachments.value = pendingAttachments.value + attachment
    }

    /** Remove attachment from pending list */
    fun removeAttachment(attachmentId: Uuid) {
        pendingAttachments.value = pendingAttachments.value.filter { it._id != attachmentId }
    }

    /** Clear all pending attachments */
    fun clearAttachments() {
        pendingAttachments.value = emptyList()
    }

    /** Clear error message */
    fun clearError() {
        errorMessage.value = null
    }
}
