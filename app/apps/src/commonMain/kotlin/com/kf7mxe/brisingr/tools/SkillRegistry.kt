package com.kf7mxe.brisingr.tools

object SkillRegistry {
    private val skills: Map<SkillType, Skill> by lazy {
        mapOf(
            SkillType.GeneralAssistant to GeneralAssistantSkill(),
            SkillType.WakeWord to WakeWordSkill(),
            SkillType.Journal to JournalSkill(),
            SkillType.Notes to NotesSkill()
        )
    }

    fun getSkill(type: SkillType): Skill =
        skills[type] ?: throw IllegalArgumentException("Unknown skill type: $type")

    fun allSkills(): List<Skill> = skills.values.toList()
}
