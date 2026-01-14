package com.kf7mxe.brisingr.tools.ui

import com.kf7mxe.brisingr.tools.SkillsState
import com.kf7mxe.brisingr.tools.SkillRegistry
import com.lightningkite.kiteui.models.rem
import com.lightningkite.kiteui.views.*
import com.lightningkite.kiteui.views.direct.*
import com.lightningkite.reactive.context.reactiveScope

fun ViewWriter.messageList() {
    col {
        gap = 0.5.rem
        padding = 0.5.rem

        // Skill-specific controls at the top
        reactiveScope {
            val skillType = SkillsState.currentSkillType()
            val skill = SkillRegistry.getSkill(skillType)
            with(skill) { renderControls() }
        }

        // Messages
        reactiveScope {
            val messages = SkillsState.currentMessages()
            val color = SkillsState.currentColor()

            if (messages.isEmpty()) {
                centered.col {
                    gap = 0.5.rem
                    subtext { content = "Start a conversation..." }
                    reactiveScope {
                        val skill = SkillsState.currentSkill()
                        subtext { content = skill.description }
                    }
                }
            } else {
                messages.forEach { message ->
                    messageBubble(message, color)
                }
            }
        }
    }
}
