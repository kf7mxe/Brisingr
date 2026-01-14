package com.kf7mxe.brisingr.tools

import com.lightningkite.kiteui.views.ViewWriter

/**
 * Interface for skill implementations.
 * Each skill can have its own chat processing logic and custom UI.
 */
interface Skill {
    /** The skill definition with name, icon, color etc. */
    val definition: SkillDefinition

    /** Process incoming user message and return assistant response */
    suspend fun processMessage(message: Message): Message?

    /** Optional: Render custom header content above the message list */
    fun ViewWriter.renderHeader() {}

    /** Optional: Render custom controls specific to this skill */
    fun ViewWriter.renderControls() {}

    /** Called when skill becomes active */
    suspend fun onActivate() {}

    /** Called when skill becomes inactive */
    suspend fun onDeactivate() {}
}
