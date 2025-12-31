package com.kf7mxe.brisingr.wakeword

/**
 * JavaScript/Web implementation of WakeWordController.
 * Currently not supported - placeholder for future Web Audio API implementation.
 */
actual object WakeWordController {
    actual fun start() {
        WakeWordState.errorMessage.value = "Wake word detection is not yet supported on web"
    }
    
    actual fun stop() {
        // No-op
    }
    
    actual fun isAvailable(): Boolean = false
}
