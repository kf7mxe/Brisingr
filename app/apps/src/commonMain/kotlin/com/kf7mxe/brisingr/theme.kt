package com.kf7mxe.brisingr

import com.lightningkite.kiteui.models.BarSemantic
import com.lightningkite.kiteui.models.ButtonSemantic
import com.lightningkite.kiteui.models.CardSemantic
import com.lightningkite.kiteui.models.Color
import com.lightningkite.kiteui.models.CornerRadii
import com.lightningkite.kiteui.models.CriticalSemantic
import com.lightningkite.kiteui.models.DangerSemantic
import com.lightningkite.kiteui.models.DialogSemantic
import com.lightningkite.kiteui.models.Edges
import com.lightningkite.kiteui.models.ErrorSemantic
import com.lightningkite.kiteui.models.FieldSemantic
import com.lightningkite.kiteui.models.FontAndStyle
import com.lightningkite.kiteui.models.HeaderSemantic
import com.lightningkite.kiteui.models.ImportantSemantic
import com.lightningkite.kiteui.models.InsetSemantic
import com.lightningkite.kiteui.models.ListSemantic
import com.lightningkite.kiteui.models.MainContentSemantic
import com.lightningkite.kiteui.models.NavSemantic
import com.lightningkite.kiteui.models.OuterSemantic
import com.lightningkite.kiteui.models.Paint
import com.lightningkite.kiteui.models.PopoverSemantic
import com.lightningkite.kiteui.models.SelectedSemantic
import com.lightningkite.kiteui.models.Theme
import com.lightningkite.kiteui.models.UnselectedSemantic
import com.lightningkite.kiteui.models.dp
import com.lightningkite.kiteui.models.lighten
import com.lightningkite.kiteui.models.px
import com.lightningkite.kiteui.models.rem
import kotlin.collections.get

fun Theme.Companion.brisingrLight(primary: Color?): Theme = run {
    val back = Color(red = 229f / 255f, green = 229f / 255f, blue = 234f / 255f, alpha = 1f)
    val defaultColor = Color(red = 0 / 255f, green = 122 / 255f, blue = 255 / 255f, alpha = 1f)
    val highlight = primary ?: defaultColor
    val separator = back.darken(0.1f)
    fun Paint.backInvert() = if (this == Color.white) back else Color.white
    Theme(
        id = "clean-${highlight.toInt()}",
        foreground = Color.black,
        background = Color.white,
        outline = separator,
        elevation = 0.px,
        cornerRadii = CornerRadii.Constant(0.5.rem),
        gap = 0.75.rem,
        padding = Edges(0.75.rem),
        derivations = mapOf(
            MainContentSemantic to {
                it.withoutBack(
                    padding = Edges(top=1.rem,bottom =0.rem,right =0.rem,left=0.rem)
                )
            }
            ,
            CardSemantic to {
                if(it.background != Color.white)
                    it.withBack(background = it.background.backInvert(), foreground = Color.black)
                else
                    it.withBack(outlineWidth = 1.px)
            },
            FieldSemantic to {
                it.withBack(
                    outline = separator,
                    outlineWidth = 1.px,
                    foreground = Color.black,
                    cornerRadii = CornerRadii.ForceConstant(0.5.rem)
                )
            },
            BarSemantic to { it.withBack },
            NavSemantic to { it.withBack },
            OuterSemantic to { it.withBack(cascading = false, gap = 1.px, padding = Edges.ZERO, background = separator) },
            MainContentSemantic to { it.withBack(cascading = false, cornerRadii = CornerRadii.Constant(0.px)) },
            InsetSemantic to { it.withBack(background = it.background.backInvert()) },
            UnselectedSemantic to { it.withBack },
            SelectedSemantic to { it[CardSemantic] },
            DialogSemantic to {
                it.withBack(
                    cascading = false,
                    outline = separator,
                    background = Color.white,
                    foreground = Color.black,
                    elevation = 4.dp
                )
            },
            PopoverSemantic to {
                it.withBack(
                    cascading = false,
                    outline = separator,
                    background = Color.white,
                    foreground = Color.black,
                    elevation = 4.dp
                )
            },
            ImportantSemantic to {
                it.withBack(
                    background = highlight,
                    foreground = if (highlight.perceivedBrightness > 0.4f) Color.white else Color.black,
                )
            },
            ListSemantic to {
                it.copy(id = "lsts", background = back).withBack(
                    cascading = false,
                    cornerRadii = CornerRadii.ForceConstant(0.75.rem),
                    gap = 1.px,
                    padding = Edges(0.px)
                )
            },
        )
    )
}