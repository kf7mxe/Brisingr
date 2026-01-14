package com.kf7mxe.brisingr.tools

import kotlinx.serialization.Serializable

@Serializable
data class SkillDefinition(
    val type: SkillType,
    val name: String,
    val icon: String,
    val colorHex: String,
    val description: String
) {
    companion object {
        val allSkills = listOf(
            SkillDefinition(
                type = SkillType.GeneralAssistant,
                name = "Assistant",
                icon = "chat",
                colorHex = "#5d6af0",
                description = "Your general-purpose AI assistant"
            ),
            SkillDefinition(
                type = SkillType.WakeWord,
                name = "Wake Word",
                icon = "mic",
                colorHex = "#FF6B35",
                description = "Voice activation settings"
            ),
            SkillDefinition(
                type = SkillType.Journal,
                name = "Journal",
                icon = "book",
                colorHex = "#4ECDC4",
                description = "Personal journaling assistant"
            ),
            SkillDefinition(
                type = SkillType.Notes,
                name = "Notes",
                icon = "note",
                colorHex = "#FFE66D",
                description = "Quick notes and reminders"
            )
        )

        fun forType(type: SkillType): SkillDefinition =
            allSkills.first { it.type == type }
    }
}
