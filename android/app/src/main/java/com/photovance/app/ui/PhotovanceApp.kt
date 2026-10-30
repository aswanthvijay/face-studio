package com.photovance.app.ui

import android.content.Context
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.photovance.app.ui.screens.AuthScreen
import com.photovance.app.ui.screens.HomeScreen
import com.photovance.app.ui.screens.SettingsSheet
import com.photovance.app.ui.theme.AppAppearance
import com.photovance.app.ui.theme.PhotovanceTheme

@Composable
fun PhotovanceApp(studio: StudioViewModel) {
    val context = LocalContext.current
    val preferences = remember { context.getSharedPreferences("studio_design", Context.MODE_PRIVATE) }
    var appearanceName by rememberSaveable {
        mutableStateOf(preferences.getString("appearance", AppAppearance.System.name) ?: AppAppearance.System.name)
    }
    val appearance = AppAppearance.entries.firstOrNull { it.name == appearanceName } ?: AppAppearance.System
    var guest by rememberSaveable { mutableStateOf(false) }
    var settings by rememberSaveable { mutableStateOf(false) }
    val snackbar = remember { SnackbarHostState() }
    LaunchedEffect(studio.message) {
        studio.message?.let { message ->
            snackbar.showSnackbar(message)
            if (studio.message == message) studio.message = null
        }
    }
    val currentUser by studio.currentUser.collectAsState()
    val canGoBack = (guest || currentUser != null) && !settings
    BackHandler(canGoBack) {
        if (!studio.isGenerating) {
            if (studio.step > 0) {
                studio.goToStep(studio.step - 1)
            } else if (guest) {
                guest = false
            }
        }
    }
    PhotovanceTheme(appearance) {
        Surface(Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
            Box(Modifier.fillMaxSize().safeDrawingPadding(), contentAlignment = Alignment.TopCenter) {
                Box(Modifier.widthIn(max = 600.dp).fillMaxSize()) {
                    if (guest || currentUser != null) {
                        HomeScreen(studio, onSettings = { settings = true })
                    } else {
                        AuthScreen(
                            authRepository = studio.authRepository,
                            onGuest = { studio.authRepository.signInAsGuest(); guest = true },
                            onSettings = { settings = true }
                        )
                    }
                    SnackbarHost(snackbar, Modifier.align(Alignment.BottomCenter).padding(16.dp))
                }
            }
            if (settings) SettingsSheet(studio, appearance, onAppearance = {
                appearanceName = it.name
                preferences.edit().putString("appearance", it.name).apply()
            }, onDismiss = { settings = false })
        }
    }
}
