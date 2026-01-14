package com.kf7mxe.brisingr.tools

class GeneralAssistantSkill : Skill {
    override val definition = SkillDefinition.forType(SkillType.GeneralAssistant)

    override suspend fun processMessage(message: Message): Message? {
        // TODO: Integrate with AI backend (Gemini, OpenAI, etc.)
        // For now, return an echo response
        return Message(
            conversationId = message.conversationId,
            role = MessageRole.Assistant,
            content = "I received your message: \"${message.content}\"\n\nAI integration coming soon!"
        )
    }
}
