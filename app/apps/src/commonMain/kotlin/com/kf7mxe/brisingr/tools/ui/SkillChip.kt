package com.kf7mxe.brisingr.tools.ui

import com.kf7mxe.brisingr.SelectedBackgroundSetToSpecificColor
import com.kf7mxe.brisingr.tools.SkillDefinition
import com.kf7mxe.brisingr.tools.SkillsState
import com.lightningkite.kiteui.models.Color
import com.lightningkite.kiteui.models.rem
import com.lightningkite.kiteui.views.*
import com.lightningkite.kiteui.views.direct.*
import com.lightningkite.reactive.context.reactiveScope

fun ViewWriter.skillChip(skill: SkillDefinition) {
    val skillColor = Color.fromHexString(skill.colorHex)

    button {
        dynamicTheme {
            val isSelected = SkillsState.currentSkillType() == skill.type

            SelectedBackgroundSetToSpecificColor[skillColor]
        }
//        reactiveScope {
//            val isSelected = SkillsState.currentSkillType() == skill.type
//            themeModifier = { theme ->
//                if (isSelected) {
//                    theme.copy(
//                        background = skillColor,
//                        foreground = Color.white
//                    )
//                } else {
//                    theme.copy(
//                        background = Color.transparent,
//                        outline = skillColor,
//                        outlineWidth = 1.dp
//                    )
//                }
//            }
//        }

        row {
            gap = 0.25.rem
            text(skill.icon)
            text(skill.name)
        }

        onClick {
            SkillsState.selectSkill(skill.type)
        }
    }
}

fun ViewWriter.currentSkillChip() {
    reactiveScope {
        val skill = SkillsState.currentSkill()
        val skillColor = Color.fromHexString(skill.colorHex)

        important.card.row {
            dynamicTheme {
                val isSelected = SkillsState.currentSkillType() == skill.type

                SelectedBackgroundSetToSpecificColor[skillColor]
            }

            gap = 0.25.rem
            text(skill.icon)
            text(skill.name)
        }
    }
}
