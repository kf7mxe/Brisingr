package com.kf7mxe.brisingr

import com.lightningkite.kiteui.models.Dimension
import com.lightningkite.kiteui.models.Icon
import com.lightningkite.kiteui.models.rem
import com.lightningkite.lightningserver.*
import com.lightningkite.lightningserver.sessions.*
import com.lightningkite.services.data.*
import com.lightningkite.services.database.*
import com.lightningkite.services.files.*
import kotlin.uuid.Uuid

fun Icon.resize(size: Dimension) = copy(width = size, height = size)

val Icon.Companion.show: Icon
    get() = Icon(
        width = 1.5.rem,
        height = 1.5.rem,
        viewBoxMinX = 0,
        viewBoxMinY = -960,
        viewBoxWidth = 960,
        viewBoxHeight = 960,
        pathDatas = listOf(
            "M480-320q75 0 127.5-52.5T660-500q0-75-52.5-127.5T480-680q-75 0-127.5 52.5T300-500q0 75 52.5 127.5T480-320Zm0-72q-45 0-76.5-31.5T372-500q0-45 31.5-76.5T480-608q45 0 76.5 31.5T588-500q0 45-31.5 76.5T480-392Zm0 192q-146 0-266-81.5T40-500q54-137 174-218.5T480-800q146 0 266 81.5T920-500q-54 137-174 218.5T480-200Zm0-300Zm0 220q113 0 207.5-59.5T832-500q-50-101-144.5-160.5T480-720q-113 0-207.5 59.5T128-500q50 101 144.5 160.5T480-280Z"
        )
    )

val Icon.Companion.bug: Icon
    get() = Icon(
        width = 1.5.rem,
        height = 1.5.rem,
        viewBoxMinX = 0,
        viewBoxMinY = -960,
        viewBoxWidth = 960,
        viewBoxHeight = 960,
        pathDatas = listOf(
            "M480-200q66 0 113-47t47-113v-160q0-66-47-113t-113-47q-66 0-113 47t-47 113v160q0 66 47 113t113 47Zm-80-120h160v-80H400v80Zm0-160h160v-80H400v80Zm80 40Zm0 320q-65 0-120.5-32T272-240H160v-80h84q-3-20-3.5-40t-.5-40h-80v-80h80q0-20 .5-40t3.5-40h-84v-80h112q14-23 31.5-43t40.5-35l-64-66 56-56 86 86q28-9 57-9t57 9l88-86 56 56-66 66q23 15 41.5 34.5T688-640h112v80h-84q3 20 3.5 40t.5 40h80v80h-80q0 20-.5 40t-3.5 40h84v80H688q-32 56-87.5 88T480-120Z"
        )
    )

val Icon.Companion.folderCopy: Icon
    get() = Icon(
        width = 1.5.rem,
        height = 1.5.rem,
        viewBoxMinX = 0,
        viewBoxMinY = -960,
        viewBoxWidth = 960,
        viewBoxHeight = 960,
        pathDatas = listOf(
            "M120-120q-33 0-56.5-23.5T40-200v-520h80v520h680v80H120Zm160-160q-33 0-56.5-23.5T200-360v-440q0-33 23.5-56.5T280-880h200l80 80h280q33 0 56.5 23.5T920-720v360q0 33-23.5 56.5T840-280H280Zm0-80h560v-360H527l-80-80H280v440Zm0 0v-440 440Z"
        )
    )

val Icon.Companion.work: Icon
    get() = Icon(
        width = 1.5.rem,
        height = 1.5.rem,
        viewBoxMinX = 0,
        viewBoxMinY = -960,
        viewBoxWidth = 960,
        viewBoxHeight = 960,
        pathDatas = listOf(
            "M160-120q-33 0-56.5-23.5T80-200v-440q0-33 23.5-56.5T160-720h160v-80q0-33 23.5-56.5T400-880h160q33 0 56.5 23.5T640-800v80h160q33 0 56.5 23.5T880-640v440q0 33-23.5 56.5T800-120H160Zm0-80h640v-440H160v440Zm240-520h160v-80H400v80ZM160-200v-440 440Z"
        )
    )

val Icon.Companion.changeOrder: Icon
    get() = Icon(
        width = 1.5.rem,
        height = 1.5.rem,
        viewBoxMinX = 0,
        viewBoxMinY = -960,
        viewBoxWidth = 960,
        viewBoxHeight = 960,
        pathDatas = listOf(
            "M280-160 80-360l200-200 56 57-103 103h287v80H233l103 103-56 57Zm400-240-56-57 103-103H440v-80h287L624-743l56-57 200 200-200 200Z"
        )
    )

val Icon.Companion.feature: Icon
    get() = Icon(
        width = 1.5.rem,
        height = 1.5.rem,
        viewBoxMinX = 0,
        viewBoxMinY = -960,
        viewBoxWidth = 960,
        viewBoxHeight = 960,
        pathDatas = listOf(
            "m320-240 160-122 160 122-60-198 160-114H544l-64-208-64 208H220l160 114-60 198ZM480-80q-83 0-156-31.5T197-197q-54-54-85.5-127T80-480q0-83 31.5-156T197-763q54-54 127-85.5T480-880q83 0 156 31.5T763-763q54 54 85.5 127T880-480q0 83-31.5 156T763-197q-54 54-127 85.5T480-80Zm0-80q134 0 227-93t93-227q0-134-93-227t-227-93q-134 0-227 93t-93 227q0 134 93 227t227 93Zm0-320Z"
        )
    )

val Icon.Companion.maintenance: Icon
    get() = Icon(
        width = 1.5.rem,
        height = 1.5.rem,
        viewBoxMinX = 0,
        viewBoxMinY = -960,
        viewBoxWidth = 960,
        viewBoxHeight = 960,
        pathDatas = listOf(
            "M686-132 444-376q-20 8-40.5 12t-43.5 4q-100 0-170-70t-70-170q0-36 10-68.5t28-61.5l146 146 72-72-146-146q29-18 61.5-28t68.5-10q100 0 170 70t70 170q0 23-4 43.5T584-516l244 242q12 12 12 29t-12 29l-84 84q-12 12-29 12t-29-12Zm29-85 27-27-256-256q18-20 26-46.5t8-53.5q0-60-38.5-104.5T386-758l74 74q12 12 12 28t-12 28L332-500q-12 12-28 12t-28-12l-74-74q9 57 53.5 95.5T360-440q26 0 52-8t47-25l256 256ZM472-488Z"
        )
    )
val Icon.Companion.paperwork: Icon
    get() = Icon(
        width = 1.5.rem,
        height = 1.5.rem,
        viewBoxMinX = 0,
        viewBoxMinY = -960,
        viewBoxWidth = 960,
        viewBoxHeight = 960,
        pathDatas = listOf(
            "M200-120q-33 0-56.5-23.5T120-200v-560q0-33 23.5-56.5T200-840h168q13-36 43.5-58t68.5-22q38 0 68.5 22t43.5 58h168q33 0 56.5 23.5T840-760v560q0 33-23.5 56.5T760-120H200Zm0-80h560v-560H200v560Zm80-80h280v-80H280v80Zm0-160h400v-80H280v80Zm0-160h400v-80H280v80Zm200-190q13 0 21.5-8.5T510-820q0-13-8.5-21.5T480-850q-13 0-21.5 8.5T450-820q0 13 8.5 21.5T480-790ZM200-200v-560 560Z"
        )
    )

val Icon.Companion.customerSupport: Icon
    get() = Icon(
        width = 1.5.rem,
        height = 1.5.rem,
        viewBoxMinX = 0,
        viewBoxMinY = -960,
        viewBoxWidth = 960,
        viewBoxHeight = 960,
        pathDatas = listOf(
            "M440-120v-80h320v-284q0-117-81.5-198.5T480-764q-117 0-198.5 81.5T200-484v244h-40q-33 0-56.5-23.5T80-320v-80q0-21 10.5-39.5T120-469l3-53q8-68 39.5-126t79-101q47.5-43 109-67T480-840q68 0 129 24t109 66.5Q766-707 797-649t40 126l3 52q19 9 29.5 27t10.5 38v92q0 20-10.5 38T840-249v49q0 33-23.5 56.5T760-120H440Zm-80-280q-17 0-28.5-11.5T320-440q0-17 11.5-28.5T360-480q17 0 28.5 11.5T400-440q0 17-11.5 28.5T360-400Zm240 0q-17 0-28.5-11.5T560-440q0-17 11.5-28.5T600-480q17 0 28.5 11.5T640-440q0 17-11.5 28.5T600-400Zm-359-62q-7-106 64-182t177-76q89 0 156.5 56.5T720-519q-91-1-167.5-49T435-698q-16 80-67.5 142.5T241-462Z"
        )
    )

val Icon.Companion.hide: Icon
    get() = Icon(
        width = 1.5.rem,
        height = 1.5.rem,
        viewBoxMinX = 0,
        viewBoxMinY = -960,
        viewBoxWidth = 960,
        viewBoxHeight = 960,
        pathDatas = listOf(
            "m644-428-58-58q9-47-27-88t-93-32l-58-58q17-8 34.5-12t37.5-4q75 0 127.5 52.5T660-500q0 20-4 37.5T644-428Zm128 126-58-56q38-29 67.5-63.5T832-500q-50-101-143.5-160.5T480-720q-29 0-57 4t-55 12l-62-62q41-17 84-25.5t90-8.5q151 0 269 83.5T920-500q-23 59-60.5 109.5T772-302Zm20 246L624-222q-35 11-70.5 16.5T480-200q-151 0-269-83.5T40-500q21-53 53-98.5t73-81.5L56-792l56-56 736 736-56 56ZM222-624q-29 26-53 57t-41 67q50 101 143.5 160.5T480-280q20 0 39-2.5t39-5.5l-36-38q-11 3-21 4.5t-21 1.5q-75 0-127.5-52.5T300-500q0-11 1.5-21t4.5-21l-84-82Zm319 93Zm-151 75Z"
        )
    )

val Icon.Companion.pin: Icon
    get() = Icon(
        width = 1.5.rem,
        height = 1.5.rem,
        viewBoxMinX = 0,
        viewBoxMinY = -960,
        viewBoxWidth = 960,
        viewBoxHeight = 960,
        pathDatas = listOf(
            "m640-480 80 80v80H520v240l-40 40-40-40v-240H240v-80l80-80v-280h-40v-80h400v80h-40v280Zm-286 80h252l-46-46v-314H400v314l-46 46Zm126 0Z"
        )
    )

val Icon.Companion.chevronUp: Icon
    get() = Icon(
        width = 1.5.rem,
        height = 1.5.rem,
        viewBoxMinX = 0,
        viewBoxMinY = -960,
        viewBoxWidth = 960,
        viewBoxHeight = 960,
        pathDatas = listOf(
            "M480-528 296-344l-56-56 240-240 240 240-56 56-184-184Z"
        )
    )

val Icon.Companion.chevronDown: Icon
    get() = Icon(
        width = 1.5.rem,
        height = 1.5.rem,
        viewBoxMinX = 0,
        viewBoxMinY = -960,
        viewBoxWidth = 960,
        viewBoxHeight = 960,
        pathDatas = listOf(
            "M480-344 240-584l56-56 184 184 184-184 56 56-240 240Z"
        )
    )

val Icon.Companion.pause: Icon
    get() = Icon(
        width = 1.5.rem,
        height = 1.5.rem,
        viewBoxMinX = 0,
        viewBoxMinY = -960,
        viewBoxWidth = 960,
        viewBoxHeight = 960,
        pathDatas = listOf(
            "M520-200v-560h240v560H520Zm-320 0v-560h240v560H200Zm400-80h80v-400h-80v400Zm-320 0h80v-400h-80v400Zm0-400v400-400Zm320 0v400-400Z"
        )
    )

val Icon.Companion.play: Icon
    get() = Icon(
        width = 1.5.rem,
        height = 1.5.rem,
        viewBoxMinX = 0,
        viewBoxMinY = -960,
        viewBoxWidth = 960,
        viewBoxHeight = 960,
        pathDatas = listOf(
            "M320-200v-560l440 280-440 280Zm80-280Zm0 134 210-134-210-134v268Z"
        )
    )

val Icon.Companion.save: Icon
    get() = Icon(
        width = 1.5.rem,
        height = 1.5.rem,
        viewBoxMinX = 0,
        viewBoxMinY = -960,
        viewBoxWidth = 960,
        viewBoxHeight = 960,
        pathDatas = listOf(
            "M840-680v480q0 33-23.5 56.5T760-120H200q-33 0-56.5-23.5T120-200v-560q0-33 23.5-56.5T200-840h480l160 160Zm-80 34L646-760H200v560h560v-446ZM480-240q50 0 85-35t35-85q0-50-35-85t-85-35q-50 0-85 35t-35 85q0 50 35 85t85 35ZM240-560h360v-160H240v160Zm-40-86v446-560 114Z"
        )
    )

val Icon.Companion.clock: Icon
    get() = Icon(
        width = 1.5.rem,
        height = 1.5.rem,
        viewBoxMinX = 0,
        viewBoxMinY = -960,
        viewBoxWidth = 960,
        viewBoxHeight = 960,
        pathDatas = listOf(
            "m612-292 56-56-148-148v-184h-80v216l172 172ZM480-80q-83 0-156-31.5T197-197q-54-54-85.5-127T80-480q0-83 31.5-156T197-763q54-54 127-85.5T480-880q83 0 156 31.5T763-763q54 54 85.5 127T880-480q0 83-31.5 156T763-197q-54 54-127 85.5T480-80Zm0-400Zm0 320q133 0 226.5-93.5T800-480q0-133-93.5-226.5T480-800q-133 0-226.5 93.5T160-480q0 133 93.5 226.5T480-160Z"
        )
    )

val Icon.Companion.disabledClock: Icon
    get() = Icon(
        width = 1.5.rem,
        height = 1.5.rem,
        viewBoxMinX = 0,
        viewBoxMinY = -960,
        viewBoxWidth = 960,
        viewBoxHeight = 960,
        pathDatas = listOf(
            "M612-292 440-464v-216h80v184l148 148-56 56Zm-498-25q-13-29-21-60t-11-63h81q3 21 8.5 42t13.5 41l-71 40ZM82-520q3-32 11-63.5t22-60.5l70 40q-8 20-13.5 41t-8.5 43H82Zm165 366q-27-20-50-43.5T154-248l70-40q14 18 29.5 33.5T287-225l-40 71Zm-22-519-71-40q20-27 43-50t50-43l40 71q-17 14-32.5 29.5T225-673ZM440-82q-32-3-63.5-11T316-115l40-70q20 8 41 13.5t43 8.5v81Zm-84-693-40-70q29-14 60.5-22t63.5-11v81q-22 3-43 8.5T356-775ZM520-82v-81q22-3 43-8.5t41-13.5l40 70q-29 14-60.5 22T520-82Zm84-693q-20-8-41-13.5t-43-8.5v-81q32 3 63.5 11t60.5 22l-40 70Zm109 621-40-71q17-14 32.5-29.5T735-287l71 40q-20 27-43 50.5T713-154Zm22-519q-14-17-29.5-32.5T673-735l40-71q27 19 50 42t42 50l-70 41Zm62 153q-3-22-8.5-43T775-604l70-41q13 30 21.5 61.5T878-520h-81Zm48 204-70-40q8-20 13.5-41t8.5-43h81q-3 32-11 63.5T845-316Z"
        )
    )

val Icon.Companion.checklist: Icon
    get() = Icon(
        width = 1.5.rem,
        height = 1.5.rem,
        viewBoxMinX = 0,
        viewBoxMinY = -960,
        viewBoxWidth = 960,
        viewBoxHeight = 960,
        pathDatas = listOf(
            "M222-200 80-342l56-56 85 85 170-170 56 57-225 226Zm0-320L80-662l56-56 85 85 170-170 56 57-225 226Zm298 240v-80h360v80H520Zm0-320v-80h360v80H520Z"
        )
    )

val Icon.Companion.switchDirection: Icon
    get() = Icon(
        width = 1.5.rem,
        height = 1.5.rem,
        viewBoxMinX = 0,
        viewBoxMinY = -960,
        viewBoxWidth = 960,
        viewBoxHeight = 960,
        pathDatas = listOf(
            "M320-440v-287L217-624l-57-56 200-200 200 200-57 56-103-103v287h-80ZM600-80 400-280l57-56 103 103v-287h80v287l103-103 57 56L600-80Z"
        )
    )

val Icon.Companion.downArrow: Icon
    get() = Icon(
        width = 1.5.rem,
        height = 1.5.rem,
        viewBoxMinX = 0,
        viewBoxMinY = -960,
        viewBoxWidth = 960,
        viewBoxHeight = 960,
        pathDatas = listOf(
            "M440-800v487L216-537l-56 57 320 320 320-320-56-57-224 224v-487h-80Z"
        )
    )

val Icon.Companion.upArrow: Icon
    get() = Icon(
        width = 1.5.rem,
        height = 1.5.rem,
        viewBoxMinX = 0,
        viewBoxMinY = -960,
        viewBoxWidth = 960,
        viewBoxHeight = 960,
        pathDatas = listOf(
            "M440-160v-487L216-423l-56-57 320-320 320 320-56 57-224-224v487h-80Z"
        )
    )

val Icon.Companion.forwardArrow: Icon
    get() = Icon(
        width = 1.5.rem,
        height = 1.5.rem,
        viewBoxMinX = 0,
        viewBoxMinY = -960,
        viewBoxWidth = 960,
        viewBoxHeight = 960,
        pathDatas = listOf(
            "M647-440H160v-80h487L423-744l57-56 320 320-320 320-57-56 224-224Z"
        )
    )

val Icon.Companion.edit: Icon
    get() = Icon(
        width = 1.5.rem,
        height = 1.5.rem,
        viewBoxMinX = 0,
        viewBoxMinY = -960,
        viewBoxWidth = 960,
        viewBoxHeight = 960,
        pathDatas = listOf(
            "M200-200h57l391-391-57-57-391 391v57Zm-80 80v-170l528-527q12-11 26.5-17t30.5-6q16 0 31 6t26 18l55 56q12 11 17.5 26t5.5 30q0 16-5.5 30.5T817-647L290-120H120Zm640-584-56-56 56 56Zm-141 85-28-29 57 57-29-28Z"
        )
    )

val Icon.Companion.target: Icon
    get() = Icon(
        width = 1.5.rem,
        height = 1.5.rem,
        viewBoxMinX = 0,
        viewBoxMinY = -960,
        viewBoxWidth = 960,
        viewBoxHeight = 960,
        pathDatas = listOf(
            "M480-80q-83 0-156-31.5T197-197q-54-54-85.5-127T80-480q0-83 31.5-156T197-763q54-54 127-85.5T480-880q83 0 156 31.5T763-763q54 54 85.5 127T880-480q0 83-31.5 156T763-197q-54 54-127 85.5T480-80Zm0-80q134 0 227-93t93-227q0-134-93-227t-227-93q-134 0-227 93t-93 227q0 134 93 227t227 93Zm0-80q-100 0-170-70t-70-170q0-100 70-170t170-70q100 0 170 70t70 170q0 100-70 170t-170 70Zm0-80q66 0 113-47t47-113q0-66-47-113t-113-47q-66 0-113 47t-47 113q0 66 47 113t113 47Zm0-80q-33 0-56.5-23.5T400-480q0-33 23.5-56.5T480-560q33 0 56.5 23.5T560-480q0 33-23.5 56.5T480-400Z"
        )
    )

val Icon.Companion.tune: Icon
    get() = Icon(
        width = 1.5.rem,
        height = 1.5.rem,
        viewBoxMinX = 0,
        viewBoxMinY = -960,
        viewBoxWidth = 960,
        viewBoxHeight = 960,
        pathDatas = listOf(
            "M520-600v-80h120v-160h80v160h120v80H520Zm120 480v-400h80v400h-80Zm-400 0v-160H120v-80h320v80H320v160h-80Zm0-320v-400h80v400h-80Z"
        )
    )

val Icon.Companion.updateTimer: Icon
    get() = Icon(
        width = 1.5.rem,
        height = 1.5.rem,
        viewBoxMinX = 0,
        viewBoxMinY = -960,
        viewBoxWidth = 960,
        viewBoxHeight = 960,
        pathDatas = listOf(
            "M480-120q-75 0-140.5-28.5t-114-77q-48.5-48.5-77-114T120-480q0-75 28.5-140.5t77-114q48.5-48.5 114-77T480-840q82 0 155.5 35T760-706v-94h80v240H600v-80h110q-41-56-101-88t-129-32q-117 0-198.5 81.5T200-480q0 117 81.5 198.5T480-200q105 0 183.5-68T756-440h82q-15 137-117.5 228.5T480-120Zm112-192L440-464v-216h80v184l128 128-56 56Z"
        )
    )

val Icon.Companion.history: Icon
    get() = Icon(
        width = 1.5.rem,
        height = 1.5.rem,
        viewBoxMinX = 0,
        viewBoxMinY = -960,
        viewBoxWidth = 960,
        viewBoxHeight = 960,
        pathDatas = listOf(
            "M480-120q-138 0-240.5-91.5T122-440h82q14 104 92.5 172T480-200q117 0 198.5-81.5T760-480q0-117-81.5-198.5T480-760q-69 0-129 32t-101 88h110v80H120v-240h80v94q51-64 124.5-99T480-840q75 0 140.5 28.5t114 77q48.5 48.5 77 114T840-480q0 75-28.5 140.5t-77 114q-48.5 48.5-114 77T480-120Zm112-192L440-464v-216h80v184l128 128-56 56Z"
        )
    )

val Icon.Companion.addClock: Icon
    get() = Icon(
        width = 1.5.rem,
        height = 1.5.rem,
        viewBoxMinX = 0,
        viewBoxMinY = -960,
        viewBoxWidth = 960,
        viewBoxHeight = 960,
        pathDatas = listOf(
            "M440-120q-75 0-140.5-28T185-225q-49-49-77-114.5T80-480q0-75 28-140.5T185-735q49-49 114.5-77T440-840q21 0 40.5 2.5T520-830v82q-20-6-39.5-9t-40.5-3q-118 0-199 81t-81 199q0 118 81 199t199 81q118 0 199-81t81-199q0-11-1-20t-3-20h82q2 11 2 20v20q0 75-28 140.5T695-225q-49 49-114.5 77T440-120Zm112-192L400-464v-216h80v184l128 128-56 56Zm168-288v-120H600v-80h120v-120h80v120h120v80H800v120h-80Z"
        )
    )

val Icon.Companion.timeEntry: Icon
    get() = Icon(
        width = 1.5.rem,
        height = 1.5.rem,
        viewBoxMinX = 0,
        viewBoxMinY = -960,
        viewBoxWidth = 960,
        viewBoxHeight = 960,
        pathDatas = listOf(
            "m787-145 28-28-75-75v-112h-40v128l87 87Zm-587 25q-33 0-56.5-23.5T120-200v-560q0-33 23.5-56.5T200-840h560q33 0 56.5 23.5T840-760v268q-19-9-39-15.5t-41-9.5v-243H200v560h242q3 22 9.5 42t15.5 38H200Zm0-120v40-560 243-3 280Zm80-40h163q3-21 9.5-41t14.5-39H280v80Zm0-160h244q32-30 71.5-50t84.5-27v-3H280v80Zm0-160h400v-80H280v80ZM720-40q-83 0-141.5-58.5T520-240q0-83 58.5-141.5T720-440q83 0 141.5 58.5T920-240q0 83-58.5 141.5T720-40Z"
        )
    )

val Icon.Companion.expandAll: Icon
    get() = Icon(
        width = 1.5.rem,
        height = 1.5.rem,
        viewBoxMinX = 0,
        viewBoxMinY = -960,
        viewBoxWidth = 960,
        viewBoxHeight = 960,
        pathDatas = listOf(
            "M480-80 240-320l57-57 183 183 183-183 57 57L480-80ZM298-584l-58-56 240-240 240 240-58 56-182-182-182 182Z"
        )
    )

val Icon.Companion.collapseAll: Icon
    get() = Icon(
        width = 1.5.rem,
        height = 1.5.rem,
        viewBoxMinX = 0,
        viewBoxMinY = -960,
        viewBoxWidth = 960,
        viewBoxHeight = 960,
        pathDatas = listOf(
            "m296-80-56-56 240-240 240 240-56 56-184-184L296-80Zm184-504L240-824l56-56 184 184 184-184 56 56-240 240Z"
        )
    )

val Icon.Companion.shipped: Icon
    get() = Icon(
        width = 1.5.rem,
        height = 1.5.rem,
        viewBoxMinX = 0,
        viewBoxMinY = -960,
        viewBoxWidth = 960,
        viewBoxHeight = 960,
        pathDatas = listOf(
            "M280-160q-50 0-85-35t-35-85H60l18-80h113q17-19 40-29.5t49-10.5q26 0 49 10.5t40 29.5h167l84-360H182l4-17q6-28 27.5-45.5T264-800h456l-37 160h117l120 160-40 200h-80q0 50-35 85t-85 35q-50 0-85-35t-35-85H400q0 50-35 85t-85 35Zm357-280h193l4-21-74-99h-95l-28 120Zm-19-273 2-7-84 360 2-7 34-146 46-200ZM20-427l20-80h220l-20 80H20Zm80-146 20-80h260l-20 80H100Zm180 333q17 0 28.5-11.5T320-280q0-17-11.5-28.5T280-320q-17 0-28.5 11.5T240-280q0 17 11.5 28.5T280-240Zm400 0q17 0 28.5-11.5T720-280q0-17-11.5-28.5T680-320q-17 0-28.5 11.5T640-280q0 17 11.5 28.5T680-240Z"
        )
    )

val Icon.Companion.test: Icon
    get() = Icon(
        width = 1.5.rem,
        height = 1.5.rem,
        viewBoxMinX = 0,
        viewBoxMinY = -960,
        viewBoxWidth = 960,
        viewBoxHeight = 960,
        pathDatas = listOf(
            "M480-80q-83 0-141.5-58.5T280-280v-360q-33 0-56.5-23.5T200-720v-80q0-33 23.5-56.5T280-880h400q33 0 56.5 23.5T760-800v80q0 33-23.5 56.5T680-640v360q0 83-58.5 141.5T480-80ZM280-720h400v-80H280v80Zm200 560q50 0 85-35t35-85H480v-80h120v-80H480v-80h120v-120H360v360q0 50 35 85t85 35ZM280-720v-80 80Z"
        )
    )

val Icon.Companion.inProgress: Icon
    get() = Icon(
        width = 1.5.rem,
        height = 1.5.rem,
        viewBoxMinX = 0,
        viewBoxMinY = -960,
        viewBoxWidth = 960,
        viewBoxHeight = 960,
        pathDatas = listOf(
            "M647-440H160v-80h487L423-744l57-56 320 320-320 320-57-56 224-224Z"
        )
    )

val Icon.Companion.visible: Icon
    get() = Icon(
        width = 1.5.rem,
        height = 1.5.rem,
        viewBoxMinX = 0,
        viewBoxMinY = -960,
        viewBoxWidth = 960,
        viewBoxHeight = 960,
        pathDatas = listOf(
            "M480-320q75 0 127.5-52.5T660-500q0-75-52.5-127.5T480-680q-75 0-127.5 52.5T300-500q0 75 52.5 127.5T480-320Zm0-72q-45 0-76.5-31.5T372-500q0-45 31.5-76.5T480-608q45 0 76.5 31.5T588-500q0 45-31.5 76.5T480-392Zm0 192q-146 0-266-81.5T40-500q54-137 174-218.5T480-800q146 0 266 81.5T920-500q-54 137-174 218.5T480-200Zm0-300Zm0 220q113 0 207.5-59.5T832-500q-50-101-144.5-160.5T480-720q-113 0-207.5 59.5T128-500q50 101 144.5 160.5T480-280Z"
        )
    )

val Icon.Companion.active: Icon
    get() = Icon(
        width = 1.5.rem,
        height = 1.5.rem,
        viewBoxMinX = 0,
        viewBoxMinY = -960,
        viewBoxWidth = 960,
        viewBoxHeight = 960,
        pathDatas = listOf(
            "M756-120 537-339l84-84 219 219-84 84Zm-552 0-84-84 276-276-68-68-28 28-51-51v82l-28 28-121-121 28-28h82l-50-50 142-142q20-20 43-29t47-9q24 0 47 9t43 29l-92 92 50 50-28 28 68 68 90-90q-4-11-6.5-23t-2.5-24q0-59 40.5-99.5T701-841q15 0 28.5 3t27.5 9l-99 99 72 72 99-99q7 14 9.5 27.5T841-701q0 59-40.5 99.5T701-561q-12 0-24-2t-23-7L204-120Z"
        )
    )

val Icon.Companion.merge: Icon
    get() = Icon(
        width = 1.5.rem,
        height = 1.5.rem,
        viewBoxMinX = 0,
        viewBoxMinY = -960,
        viewBoxWidth = 960,
        viewBoxHeight = 960,
        pathDatas = listOf(
            "m256-120-56-56 193-194q23-23 35-52t12-61v-204l-64 63-56-56 160-160 160 160-56 56-64-63v204q0 32 12 61t35 52l193 194-56 56-224-224-224 224Z"
        )
    )

val Icon.Companion.pullRequest: Icon
    get() = Icon(
        width = 1.5.rem,
        height = 1.5.rem,
        viewBoxMinX = 0,
        viewBoxMinY = 0,
        viewBoxWidth = 512,
        viewBoxHeight = 512,
        pathDatas = listOf(
            "M192,96a64,64,0,1,0-96,55.39V360.61a64,64,0,1,0,64,0V151.39A64,64,0,0,0,192,96ZM128,64A32,32,0,1,1,96,96,32,32,0,0,1,128,64Zm0,384a32,32,0,1,1,32-32A32,32,0,0,1,128,448Z",
            "M416,360.61V156a92.1,92.1,0,0,0-92-92H304V32a16,16,0,0,0-27.31-11.31l-64,64a16,16,0,0,0,0,22.62l64,64A16,16,0,0,0,304,160V128h20a28,28,0,0,1,28,28V360.61a64,64,0,1,0,64,0ZM384,448a32,32,0,1,1,32-32A32,32,0,0,1,384,448Z"
        )
    )

val Icon.Companion.attachment: Icon
    get() = Icon(
        width = 1.5.rem,
        height = 1.5.rem,
        viewBoxMinX = 0,
        viewBoxMinY = -960,
        viewBoxWidth = 960,
        viewBoxHeight = 960,
        pathDatas = listOf(
            "M720-330q0 104-73 177T470-80q-104 0-177-73t-73-177v-370q0-75 52.5-127.5T400-880q75 0 127.5 52.5T580-700v350q0 46-32 78t-78 32q-46 0-78-32t-32-78v-370h80v370q0 13 8.5 21.5T470-320q13 0 21.5-8.5T500-350v-350q-1-42-29.5-71T400-800q-42 0-71 29t-29 71v370q-1 71 49 120.5T470-160q70 0 119-49.5T640-330v-390h80v390Z"
        )
    )

val Icon.Companion.header1: Icon
    get() = Icon(
        width = 1.5.rem,
        height = 1.5.rem,
        viewBoxMinX = 0,
        viewBoxMinY = -960,
        viewBoxWidth = 960,
        viewBoxHeight = 960,
        pathDatas = listOf("M200-280v-400h80v160h160v-160h80v400h-80v-160H280v160h-80Zm480 0v-320h-80v-80h160v400h-80Z")
    )

val Icon.Companion.header2: Icon
    get() = Icon(
        width = 1.5.rem,
        height = 1.5.rem,
        viewBoxMinX = 0,
        viewBoxMinY = -960,
        viewBoxWidth = 960,
        viewBoxHeight = 960,
        pathDatas = listOf("M120-280v-400h80v160h160v-160h80v400h-80v-160H200v160h-80Zm400 0v-160q0-33 23.5-56.5T600-520h160v-80H520v-80h240q33 0 56.5 23.5T840-600v80q0 33-23.5 56.5T760-440H600v80h240v80H520Z")
    )

val Icon.Companion.header3: Icon
    get() = Icon(
        width = 1.5.rem,
        height = 1.5.rem,
        viewBoxMinX = 0,
        viewBoxMinY = -960,
        viewBoxWidth = 960,
        viewBoxHeight = 960,
        pathDatas = listOf("M120-280v-400h80v160h160v-160h80v400h-80v-160H200v160h-80Zm400 0v-80h240v-80H600v-80h160v-80H520v-80h240q33 0 56.5 23.5T840-600v240q0 33-23.5 56.5T760-280H520Z")
    )

val Icon.Companion.italic: Icon
    get() = Icon(
        width = 1.5.rem,
        height = 1.5.rem,
        viewBoxMinX = 0,
        viewBoxMinY = -960,
        viewBoxWidth = 960,
        viewBoxHeight = 960,
        pathDatas = listOf("M200-200v-100h160l120-360H320v-100h400v100H580L460-300h140v100H200Z")
    )
val Icon.Companion.unorderedList: Icon
    get() = Icon(
        width = 1.5.rem,
        height = 1.5.rem,
        viewBoxMinX = 0,
        viewBoxMinY = -960,
        viewBoxWidth = 960,
        viewBoxHeight = 960,
        pathDatas = listOf("M360-200v-80h480v80H360Zm0-240v-80h480v80H360Zm0-240v-80h480v80H360ZM200-160q-33 0-56.5-23.5T120-240q0-33 23.5-56.5T200-320q33 0 56.5 23.5T280-240q0 33-23.5 56.5T200-160Zm0-240q-33 0-56.5-23.5T120-480q0-33 23.5-56.5T200-560q33 0 56.5 23.5T280-480q0 33-23.5 56.5T200-400Zm0-240q-33 0-56.5-23.5T120-720q0-33 23.5-56.5T200-800q33 0 56.5 23.5T280-720q0 33-23.5 56.5T200-640Z")
    )
val Icon.Companion.orderedList: Icon
    get() = Icon(
        width = 1.5.rem,
        height = 1.5.rem,
        viewBoxMinX = 0,
        viewBoxMinY = -960,
        viewBoxWidth = 960,
        viewBoxHeight = 960,
        pathDatas = listOf("M120-80v-60h100v-30h-60v-60h60v-30H120v-60h120q17 0 28.5 11.5T280-280v40q0 17-11.5 28.5T240-200q17 0 28.5 11.5T280-160v40q0 17-11.5 28.5T240-80H120Zm0-280v-110q0-17 11.5-28.5T160-510h60v-30H120v-60h120q17 0 28.5 11.5T280-560v70q0 17-11.5 28.5T240-450h-60v30h100v60H120Zm60-280v-180h-60v-60h120v240h-60Zm180 440v-80h480v80H360Zm0-240v-80h480v80H360Zm0-240v-80h480v80H360Z")
    )
val Icon.Companion.qoute: Icon
    get() = Icon(
        width = 1.5.rem,
        height = 1.5.rem,
        viewBoxMinX = 0,
        viewBoxMinY = -960,
        viewBoxWidth = 960,
        viewBoxHeight = 960,
        pathDatas = listOf("m228-240 92-160q-66 0-113-47t-47-113q0-66 47-113t113-47q66 0 113 47t47 113q0 23-5.5 42.5T458-480L320-240h-92Zm360 0 92-160q-66 0-113-47t-47-113q0-66 47-113t113-47q66 0 113 47t47 113q0 23-5.5 42.5T818-480L680-240h-92ZM320-500q25 0 42.5-17.5T380-560q0-25-17.5-42.5T320-620q-25 0-42.5 17.5T260-560q0 25 17.5 42.5T320-500Zm360 0q25 0 42.5-17.5T740-560q0-25-17.5-42.5T680-620q-25 0-42.5 17.5T620-560q0 25 17.5 42.5T680-500Zm0-60Zm-360 0Z")
    )
val Icon.Companion.code: Icon
    get() = Icon(
        width = 1.5.rem,
        height = 1.5.rem,
        viewBoxMinX = 0,
        viewBoxMinY = -960,
        viewBoxWidth = 960,
        viewBoxHeight = 960,
        pathDatas = listOf("M320-240 80-480l240-240 57 57-184 184 183 183-56 56Zm320 0-57-57 184-184-183-183 56-56 240 240-240 240Z")
    )
val Icon.Companion.codeBlock: Icon
    get() = Icon(
        width = 1.5.rem,
        height = 1.5.rem,
        viewBoxMinX = 0,
        viewBoxMinY = -960,
        viewBoxWidth = 960,
        viewBoxHeight = 960,
        pathDatas = listOf("m384-336 56-57-87-87 87-87-56-57-144 144 144 144Zm192 0 144-144-144-144-56 57 87 87-87 87 56 57ZM200-120q-33 0-56.5-23.5T120-200v-560q0-33 23.5-56.5T200-840h560q33 0 56.5 23.5T840-760v560q0 33-23.5 56.5T760-120H200Zm0-80h560v-560H200v560Zm0-560v560-560Z")
    )

val Icon.Companion.bold: Icon
    get() = Icon(
        width = 1.5.rem,
        height = 1.5.rem,
        viewBoxMinX = 0,
        viewBoxMinY = -960,
        viewBoxWidth = 960,
        viewBoxHeight = 960,
        pathDatas = listOf("M272-200v-560h221q65 0 120 40t55 111q0 51-23 78.5T602-491q25 11 55.5 41t30.5 90q0 89-65 124.5T501-200H272Zm121-112h104q48 0 58.5-24.5T566-372q0-11-10.5-35.5T494-432H393v120Zm0-228h93q33 0 48-17t15-38q0-24-17-39t-44-15h-95v109Z")
    )
val Icon.Companion.review: Icon
    get() = Icon(
        width = 1.5.rem,
        height = 1.5.rem,
        viewBoxMinX = 0,
        viewBoxMinY = -960,
        viewBoxWidth = 960,
        viewBoxHeight = 960,
        pathDatas = listOf("M824-120 636-308q-41 32-90.5 50T440-240q-90 0-162.5-44T163-400h98q34 37 79.5 58.5T440-320q100 0 170-70t70-170q0-100-70-170t-170-70q-94 0-162.5 63.5T201-580h-80q8-127 99.5-213.5T440-880q134 0 227 93t93 227q0 56-18 105.5T692-364l188 188-56 56ZM397-400l-63-208-52 148H80v-60h160l66-190h60l61 204 43-134h60l60 120h30v60h-67l-47-94-50 154h-59Z")
    )
val Icon.Companion.packaging: Icon
    get() = Icon(
        width = 1.5.rem,
        height = 1.5.rem,
        viewBoxMinX = 0,
        viewBoxMinY = -960,
        viewBoxWidth = 960,
        viewBoxHeight = 960,
        pathDatas = listOf("M440-183v-274L200-596v274l240 139Zm80 0 240-139v-274L520-457v274Zm-80 92L160-252q-19-11-29.5-29T120-321v-318q0-22 10.5-40t29.5-29l280-161q19-11 40-11t40 11l280 161q19 11 29.5 29t10.5 40v318q0 22-10.5 40T800-252L520-91q-19 11-40 11t-40-11Zm200-528 77-44-237-137-78 45 238 136Zm-160 93 78-45-237-137-78 45 237 137Z")
    )
val Icon.Companion.hold: Icon
    get() = Icon(
        width = 1.5.rem,
        height = 1.5.rem,
        viewBoxMinX = 0,
        viewBoxMinY = -960,
        viewBoxWidth = 960,
        viewBoxHeight = 960,
        pathDatas = listOf("M512-40q-82 0-154-37.5T240-182L48-464l19-19q20-21 49.5-24t53.5 14l110 76v-383q0-17 11.5-28.5T320-840q17 0 28.5 11.5T360-800v537L212-366l95 138q35 51 89 79.5T512-120q103 0 175.5-72.5T760-368v-392q0-17 11.5-28.5T800-800q17 0 28.5 11.5T840-760v392q0 137-95.5 232.5T512-40Zm-72-440v-400q0-17 11.5-28.5T480-920q17 0 28.5 11.5T520-880v400h-80Zm160 0v-360q0-17 11.5-28.5T640-880q17 0 28.5 11.5T680-840v360h-80ZM486-300Z")
    )
val Icon.Companion.ready: Icon
    get() = Icon(
        width = 1.5.rem,
        height = 1.5.rem,
        viewBoxMinX = 0,
        viewBoxMinY = -960,
        viewBoxWidth = 960,
        viewBoxHeight = 960,
        pathDatas = listOf("M80-240v-480h80v480H80Zm560 0-57-56 144-144H240v-80h487L584-664l56-56 240 240-240 240Z")
    )
val Icon.Companion.blocked: Icon
    get() = Icon(
        width = 1.5.rem,
        height = 1.5.rem,
        viewBoxMinX = 0,
        viewBoxMinY = -960,
        viewBoxWidth = 960,
        viewBoxHeight = 960,
        pathDatas = listOf("M480-80q-83 0-156-31.5T197-197q-54-54-85.5-127T80-480q0-83 31.5-156T197-763q54-54 127-85.5T480-880q83 0 156 31.5T763-763q54 54 85.5 127T880-480q0 83-31.5 156T763-197q-54 54-127 85.5T480-80Zm0-80q54 0 104-17.5t92-50.5L228-676q-33 42-50.5 92T160-480q0 134 93 227t227 93Zm252-124q33-42 50.5-92T800-480q0-134-93-227t-227-93q-54 0-104 17.5T284-732l448 448Z")
    )

val Icon.Companion.report: Icon
    get() = Icon(
        width = 1.5.rem,
        height = 1.5.rem,
        viewBoxMinX = 0,
        viewBoxMinY = -960,
        viewBoxWidth = 960,
        viewBoxHeight = 960,
        pathDatas = listOf("M280-280h280v-80H280v80Zm0-160h400v-80H280v80Zm0-160h400v-80H280v80Zm-80 480q-33 0-56.5-23.5T120-200v-560q0-33 23.5-56.5T200-840h560q33 0 56.5 23.5T840-760v560q0 33-23.5 56.5T760-120H200Zm0-80h560v-560H200v560Zm0-560v560-560Z")
    )

val Icon.Companion.csv: Icon
    get() = Icon(
        width = 1.5.rem,
        height = 1.5.rem,
        viewBoxMinX = 0,
        viewBoxMinY = -960,
        viewBoxWidth = 960,
        viewBoxHeight = 960,
        pathDatas = listOf("M230-360h120v-60H250v-120h100v-60H230q-17 0-28.5 11.5T190-560v160q0 17 11.5 28.5T230-360Zm156 0h120q17 0 28.5-11.5T546-400v-60q0-17-11.5-31.5T506-506h-60v-34h100v-60H426q-17 0-28.5 11.5T386-560v60q0 17 11.5 30.5T426-456h60v36H386v60Zm264 0h60l70-240h-60l-40 138-40-138h-60l70 240ZM160-160q-33 0-56.5-23.5T80-240v-480q0-33 23.5-56.5T160-800h640q33 0 56.5 23.5T880-720v480q0 33-23.5 56.5T800-160H160Zm0-80h640v-480H160v480Zm0 0v-480 480Z")
    )

val Icon.Companion.tableView: Icon
    get() = Icon(
        width = 1.5.rem,
        height = 1.5.rem,
        viewBoxMinX = 0,
        viewBoxMinY = -960,
        viewBoxWidth = 960,
        viewBoxHeight = 960,
        pathDatas = listOf("M280-600v-80h400v80H280Zm0 160v-80h400v80H280Zm0 160v-80h400v80H280ZM160-120q-33 0-56.5-23.5T80-200v-560q0-33 23.5-56.5T160-840h640q33 0 56.5 23.5T880-760v560q0 33-23.5 56.5T800-120H160Zm0-80h640v-560H160v560Zm0-560v560-560Z")
    )

val Icon.Companion.chartView: Icon
    get() = Icon(
        width = 1.5.rem,
        height = 1.5.rem,
        viewBoxMinX = 0,
        viewBoxMinY = -960,
        viewBoxWidth = 960,
        viewBoxHeight = 960,
        pathDatas = listOf("M120-200v-80l200-200 160 160 280-280 56 57-336 335-160-160-144 144-56-56Zm0-320v-80l200-200 160 160 280-280 56 57-336 335-160-160-144 144-56-56Z")
    )