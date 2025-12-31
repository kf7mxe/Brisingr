package com.kf7mxe.brisingr.wakeword

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import androidx.core.app.NotificationCompat
import com.kf7mxe.brisingr.MainActivity
import com.kf7mxe.brisingr.R
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch

/**
 * Android Foreground Service for continuous wake word detection.
 * Runs in the background with a persistent notification.
 */
class WakeWordService : Service() {
    
    companion object {
        private const val CHANNEL_ID = "wake_word_channel"
        private const val NOTIFICATION_ID = 1001
    }
    
    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private var detector: WakeWordDetector? = null
    
    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
        WakeWordController.init(this)
    }
    
    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        startForeground()
        startDetection()
        return START_STICKY
    }
    
    private fun startForeground() {
        val notification = createNotification()
        
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            startForeground(
                NOTIFICATION_ID, 
                notification,
                ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE
            )
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }
    }
    
    private fun createNotificationChannel() {
        val channel = NotificationChannel(
            CHANNEL_ID,
            "Wake Word Detection",
            NotificationManager.IMPORTANCE_LOW
        ).apply {
            description = "Listening for wake word"
            setShowBadge(false)
        }
        
        val manager = getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(channel)
    }
    
    private fun createNotification(): Notification {
        val intent = Intent(this, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_SINGLE_TOP
        }
        
        val pendingIntent = PendingIntent.getActivity(
            this,
            0,
            intent,
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )
        
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Wake Word Active")
            .setContentText("Listening for wake word...")
            .setSmallIcon(R.mipmap.ic_launcher)
            .setContentIntent(pendingIntent)
            .setOngoing(true)
            .setSilent(true)
            .build()
    }
    
    private fun startDetection() {
        serviceScope.launch {
            try {
                WakeWordState.serviceRunning.value = true
                WakeWordState.errorMessage.value = null
                
                detector = WakeWordDetector(this@WakeWordService)
                detector?.startDetection()
                
            } catch (e: Exception) {
                WakeWordState.errorMessage.value = "Detection error: ${e.message}"
                e.printStackTrace()
            }
        }
    }
    
    private fun stopDetection() {
        detector?.stopDetection()
        detector = null
        WakeWordState.serviceRunning.value = false
    }
    
    override fun onDestroy() {
        super.onDestroy()
        stopDetection()
        serviceScope.cancel()
    }
    
    override fun onBind(intent: Intent?): IBinder? = null
}
