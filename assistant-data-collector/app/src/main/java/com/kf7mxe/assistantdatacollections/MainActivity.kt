package com.kf7mxe.assistantdatacollections

import android.content.Intent
import android.os.Bundle
import android.view.KeyEvent
import com.google.android.material.bottomnavigation.BottomNavigationView
import androidx.appcompat.app.AppCompatActivity
import androidx.navigation.findNavController
import androidx.navigation.fragment.NavHostFragment
import androidx.navigation.ui.AppBarConfiguration
import androidx.navigation.ui.setupActionBarWithNavController
import androidx.navigation.ui.setupWithNavController
import com.kf7mxe.assistantdatacollections.databinding.ActivityMainBinding
import com.kf7mxe.assistantdatacollections.ui.wakeWord.WakeWordCollectFragment

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val navView: BottomNavigationView = binding.navView

        val navController = findNavController(R.id.nav_host_fragment_activity_main)
        // Passing each menu ID as a set of Ids because each
        // menu should be considered as top level destinations.
        val appBarConfiguration = AppBarConfiguration(
            setOf(
                R.id.navigation_home, R.id.navigation_dashboard, R.id.navigation_notifications
            )
        )
        setupActionBarWithNavController(navController, appBarConfiguration)
        navView.setupWithNavController(navController)
    }

    override fun onKeyDown(keyCode: Int, event: KeyEvent?): Boolean {
        if (keyCode == KeyEvent.KEYCODE_VOLUME_UP) {
            // call the myOnKeyDown in the current fragment
            val navHostFragment = supportFragmentManager.findFragmentById(R.id.nav_host_fragment_activity_main) as NavHostFragment
            val fragment = navHostFragment.childFragmentManager.fragments[0] as WakeWordCollectFragment
            if (fragment is WakeWordCollectFragment) {
                fragment.myOnKeyDown(keyCode)
            }

            return true
        }
        if (keyCode == KeyEvent.KEYCODE_VOLUME_DOWN) {
            val navHostFragment = supportFragmentManager.findFragmentById(R.id.nav_host_fragment_activity_main) as NavHostFragment
            val fragment = navHostFragment.childFragmentManager.fragments[0] as WakeWordCollectFragment
            if (fragment is WakeWordCollectFragment) {
                fragment.myOnKeyDown(keyCode)
            }
            return true
        }
        return super.onKeyDown(keyCode, event)
    }
}