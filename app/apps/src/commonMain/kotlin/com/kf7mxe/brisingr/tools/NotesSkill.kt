package com.kf7mxe.brisingr.tools

class NotesSkill : Skill {
    override val definition = SkillDefinition.forType(SkillType.Notes)

    override suspend fun processMessage(message: Message): Message? {
        val content = message.content.trim()

        return when {
            content.startsWith("/list") -> {
                Message(
                    conversationId = message.conversationId,
                    role = MessageRole.Assistant,
                    content = "Your notes are stored in this conversation. Scroll up to see previous notes."
                )
            }
            content.startsWith("/search") -> {
                val query = content.removePrefix("/search").trim()
                Message(
                    conversationId = message.conversationId,
                    role = MessageRole.Assistant,
                    content = "Searching for: '$query'...\n\n(Search functionality coming soon)"
                )
            }
            content.startsWith("/help") -> {
                Message(
                    conversationId = message.conversationId,
                    role = MessageRole.Assistant,
                    content = """
                        Notes Commands:
                        - Just type to save a note
                        - /list - Show all notes in this conversation
                        - /search <query> - Search your notes
                        - /help - Show this help message
                    """.trimIndent()
                )
            }
            else -> {
                Message(
                    conversationId = message.conversationId,
                    role = MessageRole.Assistant,
                    content = "Note saved. Use /list to see all notes or /search <query> to find notes."
                )
            }
        }
    }
}
