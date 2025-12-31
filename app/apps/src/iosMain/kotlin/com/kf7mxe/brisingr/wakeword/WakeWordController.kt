package com.kf7mxe.brisingr.wakeword

/**
 * iOS implementation of WakeWordController.
 * Currently placeholder - future implementation will use AVAudioEngine and CoreML.
 */
actual object WakeWordController {
    actual fun start() {
        WakeWordState.errorMessage.value = "Wake word detection is not yet supported on iOS"
    }
    
    actual fun stop() {
        // No-op
    }
    
    actual fun isAvailable(): Boolean = false
}
