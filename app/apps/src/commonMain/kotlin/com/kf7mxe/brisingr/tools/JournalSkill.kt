package com.kf7mxe.brisingr.tools

import kotlinx.datetime.TimeZone
import kotlinx.datetime.toLocalDateTime
import kotlin.time.Clock.System.now

class JournalSkill : Skill {
    override val definition = SkillDefinition.forType(SkillType.Journal)

    private val prompts = listOf(
        "What are you grateful for today?",
        "How are you feeling right now?",
        "What's one thing you accomplished today?",
        "What's on your mind?",
        "What do you want to remember about today?",
        "What made you smile today?",
        "What challenges did you face?",
        "What did you learn today?"
    )

    override suspend fun processMessage(message: Message): Message? {
        val now = now().toLocalDateTime(TimeZone.currentSystemDefault())
        val dateStr = "${now.month.name.lowercase().replaceFirstChar { it.uppercase() }} ${now.dayOfMonth}, ${now.year}"

        return if (message.content.isBlank()) {
            Message(
                conversationId = message.conversationId,
                role = MessageRole.Assistant,
                content = "Journal Entry - $dateStr\n\n${prompts.random()}"
            )
        } else {
            Message(
                conversationId = message.conversationId,
                role = MessageRole.Assistant,
                content = "Entry saved for $dateStr.\n\nWould you like to continue writing, or shall I offer another prompt?"
            )
        }
    }
}
