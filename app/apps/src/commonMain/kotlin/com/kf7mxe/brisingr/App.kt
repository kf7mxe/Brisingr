package com.kf7mxe.brisingr

import com.lightningkite.kiteui.exceptions.ExceptionToMessages
import com.lightningkite.kiteui.exceptions.installLsError
import com.lightningkite.kiteui.models.*
import com.lightningkite.kiteui.navigation.PageNavigator
import com.lightningkite.kiteui.views.ViewWriter
import com.kf7mxe.brisingr.sdk.installLoggedOutErrors
import com.kf7mxe.brisingr.tools.ui.messageList
import com.kf7mxe.brisingr.tools.ui.skillsInputArea
import com.lightningkite.kiteui.mimeType
import com.lightningkite.kiteui.views.atBottom
import com.lightningkite.kiteui.views.card
import com.lightningkite.kiteui.views.direct.col
import com.lightningkite.kiteui.views.direct.frame
import com.lightningkite.kiteui.views.direct.image
import com.lightningkite.kiteui.views.direct.padded
import com.lightningkite.kiteui.views.direct.rawImage
import com.lightningkite.kiteui.views.direct.scrolling
import com.lightningkite.kiteui.views.direct.sizeConstraints
import com.lightningkite.kiteui.views.expanding
import com.lightningkite.kiteui.views.l2.applySafeInsets
import com.lightningkite.reactive.core.Signal
import com.lightningkite.reactive.core.rememberSuspending

//val defaultTheme = brandBasedExperimental("bsa", normalBack = Color.white)
val defaultTheme = Theme.flat2("default", Angle(0.55f))// brandBasedExperimental("bsa", normalBack = Color.white)
val appTheme = Signal {
    Theme.brisingrLight()
}
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
//        gap = 0.rem
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
//        padding = 0.0.rem
//        gap = 0.0.rem
//            navigator = mainPageNavigator
//            main.bindToPlatform(context)
//            mainPageNavigator = main
//            pageNavigator = main
//            overlayFrame = this
//            dialogPageNavigator = dialog
        expanding.rawImage(Resources.background, "background",ImageScaleType.Crop )
        scrolling.col {
            gap = 0.5.rem
            padding = 1.rem


            sizeConstraints(height = 20.rem).image {
                ::source {
                    animation()
                }
                description = ""
            }

            // Message list - displays current skill's chat messages
            messageList()





            // Bottom input area with skills

        }

      padded.atBottom.skillsInputArea()

//            navigatorViewDialog()
//

        }
//    }
}

interface UseFullPage


