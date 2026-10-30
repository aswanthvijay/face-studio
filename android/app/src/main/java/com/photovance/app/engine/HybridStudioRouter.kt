package com.photovance.app.engine

import android.content.Context
import com.photovance.app.hardware.DeviceCapabilityDetector
import com.photovance.app.hardware.DeviceSpec
import com.photovance.app.hardware.DeviceTier
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

enum class StudioEngineMode(val title: String, val subtitle: String, val badge: String) {
    LOCAL_OFFLINE(
        title = "⚡ Offline Local Studio",
        subtitle = "1–5s On-Device NPU · Zero Server Latency · 100% Private",
        badge = "Snapdragon NPU"
    ),
    CLOUD_8K(
        title = "☁️ Cloud 8K Studio",
        subtitle = "Photovance Ultra-Portrait™ · Studio Lighting · 8K Master",
        badge = "8K Master"
    )
}

class HybridStudioRouter(context: Context) {
    val deviceSpec: DeviceSpec = DeviceCapabilityDetector.detect(context)

    private val _engineMode = MutableStateFlow(
        if (deviceSpec.supportsLocalNpuStudio) StudioEngineMode.LOCAL_OFFLINE
        else StudioEngineMode.CLOUD_8K
    )
    val engineMode: StateFlow<StudioEngineMode> = _engineMode.asStateFlow()

    fun switchMode(newMode: StudioEngineMode): Boolean {
        if (newMode == StudioEngineMode.LOCAL_OFFLINE && !deviceSpec.supportsLocalNpuStudio) {
            // Device does not support local offline execution
            return false
        }
        _engineMode.value = newMode
        return true
    }

    fun toggleMode() {
        if (!deviceSpec.supportsLocalNpuStudio) return
        _engineMode.value = if (_engineMode.value == StudioEngineMode.LOCAL_OFFLINE) {
            StudioEngineMode.CLOUD_8K
        } else {
            StudioEngineMode.LOCAL_OFFLINE
        }
    }
}
