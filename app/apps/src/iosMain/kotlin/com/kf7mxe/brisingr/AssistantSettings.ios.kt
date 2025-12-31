package com.kf7mxe.brisingr

// Platform-specific implementation for non-Android platforms
actual fun openAssistantSettings() {
    // No-op for iOS and Web platforms
    println("Assistant settings only available on Android")
}