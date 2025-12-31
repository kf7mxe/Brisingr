package com.kf7mxe.brisingr.extensions

import com.lightningkite.kiteui.Platform
import com.kf7mxe.brisingr.AppPlatform


fun Platform.toAppPlatform(): AppPlatform = when (this) {
    Platform.iOS -> AppPlatform.iOS
    Platform.Android -> AppPlatform.Android
    Platform.Web -> AppPlatform.Web
    Platform.Desktop -> AppPlatform.Desktop
}

