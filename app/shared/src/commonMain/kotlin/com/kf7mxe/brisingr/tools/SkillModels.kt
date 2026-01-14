package com.kf7mxe.brisingr.tools

import com.kf7mxe.brisingr.User
import com.lightningkite.services.data.GenerateDataClassPaths
import com.lightningkite.services.data.Index
import com.lightningkite.services.data.References
import com.lightningkite.services.database.HasId
import kotlinx.datetime.Instant
import kotlinx.serialization.Serializable
import kotlin.time.Clock
import kotlin.uuid.Uuid

@Serializable
enum class SkillType {
    GeneralAssistant,
    WakeWord,
    Journal,
    Notes
}

@Serializable
enum class MessageRole {
    User,
    Assistant,
    System
}

@Serializable
enum class AttachmentType {
    Image,
    Audio,
    File
}

@Serializable
data class MessageAttachment(
    val _id: Uuid = Uuid.random(),
    val type: AttachmentType,
    val url: String,
    val mimeType: String,
    val fileName: String,
    val sizeBytes: Long? = null
)

@GenerateDataClassPaths
@Serializable
data class Message(
    override val _id: Uuid = Uuid.random(),
    @Index @References(Conversation::class) val conversationId: Uuid,
    val role: MessageRole,
    val content: String,
    val attachments: List<MessageAttachment> = emptyList(),
    val createdAt: Instant = Clock.System.now(),
    val metadata: Map<String, String> = emptyMap()
) : HasId<Uuid>

@GenerateDataClassPaths
@Serializable
data class Conversation(
    override val _id: Uuid = Uuid.random(),
    @Index @References(User::class) val userId: Uuid,
    val skillType: SkillType,
    val title: String = "New Conversation",
    val createdAt: Instant = Clock.System.now(),
    val updatedAt: Instant = Clock.System.now(),
    val isArchived: Boolean = false
) : HasId<Uuid>

@GenerateDataClassPaths
@Serializable
data class SkillSettings(
    override val _id: Uuid = Uuid.random(),
    @Index @References(User::class) val userId: Uuid,
    val skillType: SkillType,
    val enabled: Boolean = true,
    val settings: Map<String, String> = emptyMap()
) : HasId<Uuid>
