package com.kf7mxe.brisingr

import com.lightningkite.kiteui.Build
import com.lightningkite.kiteui.Platform
import com.lightningkite.kiteui.current
import com.lightningkite.kiteui.exceptions.ExceptionToMessages
import com.lightningkite.kiteui.exceptions.installLsError
import com.lightningkite.kiteui.models.*
import com.lightningkite.kiteui.navigation.PageNavigator
import com.lightningkite.kiteui.reactive.*
import com.lightningkite.kiteui.navigation.dialogPageNavigator
import com.lightningkite.kiteui.views.ViewWriter
import com.lightningkite.kiteui.views.direct.confirmDanger
import com.lightningkite.kiteui.views.l2.appNav
import com.kf7mxe.brisingr.extensions.toAppPlatform
import com.kf7mxe.brisingr.sdk.currentSession
import com.kf7mxe.brisingr.sdk.installLoggedOutErrors
import com.kf7mxe.brisingr.sdk.selectedApi
import com.kf7mxe.brisingr.utils.fcmSetup
import com.kf7mxe.brisingr.utils.notificationPermissions
import com.kf7mxe.brisingr.utils.requestNotificationPermissions
import com.kf7mxe.brisingr.wakeword.WakeWordController
import com.kf7mxe.brisingr.wakeword.WakeWordPage
import com.kf7mxe.brisingr.wakeword.WakeWordState
import com.lightningkite.kiteui.mimeType
import com.lightningkite.kiteui.navigation.mainPageNavigator
import com.lightningkite.kiteui.navigation.pageNavigator
import com.lightningkite.kiteui.views.atBottom
import com.lightningkite.kiteui.views.atStart
import com.lightningkite.kiteui.views.card
import com.lightningkite.kiteui.views.centered
import com.lightningkite.kiteui.views.danger
import com.lightningkite.kiteui.views.direct.button
import com.lightningkite.kiteui.views.direct.col
import com.lightningkite.kiteui.views.direct.coordinatorFrame
import com.lightningkite.kiteui.views.direct.frame
import com.lightningkite.kiteui.views.direct.h1
import com.lightningkite.kiteui.views.direct.h3
import com.lightningkite.kiteui.views.direct.image
import com.lightningkite.kiteui.views.direct.onClick
import com.lightningkite.kiteui.views.direct.padded
import com.lightningkite.kiteui.views.direct.rawImage
import com.lightningkite.kiteui.views.direct.row
import com.lightningkite.kiteui.views.direct.scrolling
import com.lightningkite.kiteui.views.direct.shownWhen
import com.lightningkite.kiteui.views.direct.sizeConstraints
import com.lightningkite.kiteui.views.direct.subtext
import com.lightningkite.kiteui.views.direct.switch
import com.lightningkite.kiteui.views.direct.text
import com.lightningkite.kiteui.views.direct.textArea
import com.lightningkite.kiteui.views.direct.textInput
import com.lightningkite.kiteui.views.expanding
import com.lightningkite.kiteui.views.important
import com.lightningkite.kiteui.views.l2.appBase
import com.lightningkite.kiteui.views.l2.applySafeInsets
import com.lightningkite.kiteui.views.l2.coordinatorFrame
import com.lightningkite.kiteui.views.l2.navigatorViewDialog
import com.lightningkite.reactive.context.await
import com.lightningkite.reactive.context.invoke
import com.lightningkite.reactive.context.reactiveScope
import com.lightningkite.reactive.context.reactiveSuspending
import com.lightningkite.reactive.core.AppScope
import com.lightningkite.reactive.core.Signal
import com.lightningkite.reactive.core.remember
import com.lightningkite.reactive.core.rememberSuspending
import com.lightningkite.reactive.extensions.value
import com.lightningkite.services.database.Query
import com.lightningkite.services.database.condition
import com.lightningkite.services.database.eq
import kotlinx.coroutines.launch

//val defaultTheme = brandBasedExperimental("bsa", normalBack = Color.white)
val defaultTheme = Theme.flat2("default", Angle(0.55f))// brandBasedExperimental("bsa", normalBack = Color.white)
val appTheme = Signal<Theme>(Theme.brisingrLight(Color.fromHexString("#5d6af0")))
val animation = rememberSuspending {
    println("DEBUG in start get blob")
    println("DEBUG blob ${Resources.gearFlame().mimeType()}")
    val blob = Resources.gearFlame()
    ImageRaw(blob)
}
// Notification Items
val fcmToken: Signal<String?> = Signal(null)
val setFcmToken =
    { token: String -> fcmToken.value = token } //This is for iOS. It is used in the iOS app. Do not remove.

var appUpdateChecked = false

fun ViewWriter.app(navigator: PageNavigator, dialog: PageNavigator) {
    ExceptionToMessages.root.installLsError()
    ExceptionToMessages.root.installLoggedOutErrors()

//    AppScope.reactiveSuspending {
//        if (currentSession() == null) return@reactiveSuspending
//        val permission = notificationPermissions()
//        when (permission) {
//            false -> {}
//
//            true -> {
//                fcmSetup()
//            }
//
//            null -> {
//                confirmDanger(
//                    "Send notifications?",
//                    "LS KiteUI Starter would like to send you notifications.",
//                    "Allow"
//                ) {
//                    requestNotificationPermissions()
//                }
//            }
//        }
//    }


//    if (Platform.current != Platform.Web && !appUpdateChecked) {
//        appUpdateChecked = true
//        AppScope.launch {
//            val currentBuild = Build.version
//            val releases = try {
//                selectedApi.await().api.appRelease.query(
//                    Query(
//                        condition { it.platform.eq(Platform.current.toAppPlatform()) }
//                    ))
//            } catch (_: Exception) {
//                return@launch
//            }
//
//            val currentRelease = releases.find { it.version == currentBuild } ?: return@launch
//            val latestRelease = releases.maxByOrNull { it.releaseDate } ?: return@launch
//            if (latestRelease._id != currentRelease._id) {
//                dialogPageNavigator.navigate(
//                    UpdateDialog(
//                        newVersion = latestRelease.version,
//                        forceUpdate = releases.any { it.requiredUpdate && it.releaseDate > currentRelease.releaseDate }
//                    )
//                )
//            }
//        }
//    }

//    atBottom.frame {
//        sizeConstraints(width = 100.rem, height = 50.rem).text {
//            content = "Test"
//        }
//        coordinatorFrame?.bottomSheet (){
//            sizeConstraints(width = 50.rem, height = 100.rem).card.text{
//                content= "Testing"
//            }
//        }
//    }
//    navigator.navigate(HomePage())
//    return appBase(navigator, dialog) {



    frame{
        gap = 0.rem
////            sessionToken.addListener {
////                if (sessionToken.value != null) AppScope.launch {
////                    session().me.invoke()?.muteBackupInfo?.let { muteBackupInfo.value = it }
////                    if (!hasSubscription()) mainPageNavigator.navigate(
////                        SubscriptionPage()
////                    )
////                    else mainPageNavigator.navigate(LandingPage())
////                }
////            }
//
        beforeNextElementSetup {
            applySafeInsets(bottom = true)
        }
        padding = 0.0.rem
        gap = 0.0.rem
//            navigator = mainPageNavigator
//            main.bindToPlatform(context)
//            mainPageNavigator = main
//            pageNavigator = main
//            overlayFrame = this
//            dialogPageNavigator = dialog
        expanding.rawImage(Resources.background, "background",ImageScaleType.Stretch)
        scrolling.col {
            gap = 0.5.rem


            sizeConstraints(height = 20.rem).image {
                ::source {
                    animation()
                }
                description = ""
            }










            scrolling.col {
                gap = 1.rem
                padding = 1.rem

                h1("Wake Word Detection")

                subtext {
                    ::content {
                        if (WakeWordController.isAvailable())
                            "Control wake word detection on this device"
                        else
                            "Wake word detection is not available on this platform"
                    }
                }

                // Service Status Card
                card.col {
                    gap = 0.5.rem

                    row {
                        expanding.h3("Service Status")
                        text {
                            ::content {
                                if (WakeWordState.serviceRunning()) "🟢 Running" else "⚪ Stopped"
                            }
                        }
                    }

                    // Enable/Disable Toggle
                    row {
                        expanding.text("Enable Wake Word")
                        switch {
                            checked bind WakeWordState.enabled
                            reactiveScope {
                                if (WakeWordState.enabled()) {
                                    WakeWordController.start()
                                } else {
                                    WakeWordController.stop()
                                }
                            }
                        }
                    }
                }

                // Detection Probability Card
                card.col {
                    gap = 0.5.rem

                    h3("Detection")

                    // Probability bar
                    row {
                        text("Probability:")
                        expanding.frame {
                            // Background bar
                            sizeConstraints(height = 1.5.rem).card.frame { }

                            // Filled portion
                            atStart.sizeConstraints(height = 1.5.rem).frame {
                                reactiveScope {
                                    val prob = WakeWordState.probability()
                                    // Visual width based on probability
                                }
                            }
                        }
                        text {
                            ::content { "${(WakeWordState.probability() * 100).toInt()}%" }
                        }
                    }

                    // Threshold slider
                    row {
                        text("Threshold:")
                        expanding.text {
                            ::content { "${(WakeWordState.threshold() * 100).toInt()}%" }
                        }
                    }

                    // Detection indicator
                    important.centered.sizeConstraints(height = 4.rem).frame {
                        text {
                            ::content {
                                if (WakeWordState.detected()) "🎤 WAKE WORD DETECTED!" else "Listening..."
                            }
                        }
                    }.shownWhen { WakeWordState.serviceRunning() }
                }

                // Error display
                danger.card.col {
                    gap = 0.5.rem
                    h3("Error")
                    text {
                        ::content { WakeWordState.errorMessage() ?: "" }
                    }
                }.shownWhen { WakeWordState.errorMessage() != null }

                // Settings Card
                card.col {
                    gap = 0.5.rem
                    h3("Settings")

                    row {
                        text("Cooldown (seconds):")
                        expanding.text {
                            ::content { WakeWordState.cooldownSeconds().toString() }
                        }
                    }
                }

                // Test button
                button {
                    centered.text("Simulate Detection")
                    onClick {
                        WakeWordState.detected.value = true
                        AppScope.launch {
                            kotlinx.coroutines.delay(2000)
                            WakeWordState.detected.value = false
                        }
                    }
                }
            }

























            atBottom.card.col {
                row {

                }
                row {
                    textArea {

                    }
                }
            }

        }

//            navigatorViewDialog()
//

        }
//    }
}

interface UseFullPage


