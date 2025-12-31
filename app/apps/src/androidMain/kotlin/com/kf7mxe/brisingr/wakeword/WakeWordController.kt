package com.kf7mxe.brisingr.wakeword

import android.content.Context
import android.content.Intent
import com.lightningkite.kiteui.KiteUiActivity

/**
 * Android implementation of WakeWordController.
 * Controls the WakeWordService foreground service.
 */
actual object WakeWordController {
    private var context: Context? = null
    
    fun init(context: Context) {
        this.context = context
    }
    
    actual fun start() {
        val ctx = context ?: return
        val intent = Intent(ctx, WakeWordService::class.java)
        ctx.startForegroundService(intent)
    }
    
    actual fun stop() {
        val ctx = context ?: return
        val intent = Intent(ctx, WakeWordService::class.java)
        ctx.stopService(intent)
    }
    
    actual fun isAvailable(): Boolean = context != null
}
