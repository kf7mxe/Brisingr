package com.kf7mxe.brisingr.wakeword

import com.lightningkite.reactive.core.Signal

/**
 * Shared state for wake word detection across all platforms.
 * Android updates these signals from the WakeWordService.
 */
object WakeWordState {
    /** Whether wake word detection is enabled by the user */
    val enabled = Signal(false)
    
    /** Current detection probability (0.0 - 1.0) */
    val probability = Signal(0f)
    
    /** True when wake word was just detected (resets after cooldown) */
    val detected = Signal(false)
    
    /** True when the background service is actively running */
    val serviceRunning = Signal(false)
    
    /** Error message if something went wrong, null if no error */
    val errorMessage = Signal<String?>(null)
    
    /** Detection threshold (default 0.65) */
    val threshold = Signal(0.65f)
    
    /** Cooldown between detections in seconds */
    val cooldownSeconds = Signal(2.0f)
}

/**
 * Platform-specific wake word controller interface.
 * Implemented by each platform (Android, iOS, Web).
 */
expect object WakeWordController {
    fun start()
    fun stop()
    fun isAvailable(): Boolean
}
