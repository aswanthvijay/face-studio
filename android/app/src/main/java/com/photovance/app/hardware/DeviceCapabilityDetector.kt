package com.photovance.app.hardware

import android.app.ActivityManager
import android.content.Context
import android.os.Build

enum class DeviceTier {
    HIGH_PERFORMANCE_NPU, // S24 Ultra, Snapdragon 8 Gen 3, >= 11GB RAM
    STANDARD_CLOUD        // Other devices (routes to cloud 8K GPU)
}

data class DeviceSpec(
    val tier: DeviceTier,
    val socName: String,
    val totalRamGb: Double,
    val deviceModel: String,
    val isS24Ultra: Boolean,
    val supportsLocalNpuStudio: Boolean
)

object DeviceCapabilityDetector {

    fun detect(context: Context): DeviceSpec {
        val actManager = context.getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager
        val memInfo = ActivityManager.MemoryInfo()
        actManager?.getMemoryInfo(memInfo)

        val totalRamGb = memInfo.totalMem / (1024.0 * 1024.0 * 1024.0)

        val socModel = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) Build.SOC_MODEL else null
        val soc = (socModel ?: Build.HARDWARE ?: "").lowercase()
        val model = (Build.MODEL ?: "").lowercase()
        val board = (Build.BOARD ?: "").lowercase()

        // S24 Ultra model codes: SM-S928B, SM-S928U, SM-S928W, SM-S9280, etc.
        // Snapdragon 8 Gen 3 codename: pineapple, sm8650
        val isS24Ultra = model.contains("sm-s928") || model.contains("s24 ultra")
        val isSnapdragon8Gen3 = soc.contains("sm8650") || soc.contains("pineapple") || board.contains("pineapple")
        val hasFlagshipRam = totalRamGb >= 10.5 // 12GB device reports ~11.1 GB usable

        val isEligible = isS24Ultra || (isSnapdragon8Gen3 && hasFlagshipRam) || (totalRamGb >= 11.0)

        val tier = if (isEligible) DeviceTier.HIGH_PERFORMANCE_NPU else DeviceTier.STANDARD_CLOUD

        val displaySoc = when {
            isS24Ultra -> "Snapdragon 8 Gen 3 for Galaxy"
            isSnapdragon8Gen3 -> "Snapdragon 8 Gen 3"
            soc.isNotBlank() -> socModel ?: Build.HARDWARE ?: "ARM64 Processor"
            else -> "ARM64 Processor"
        }

        return DeviceSpec(
            tier = tier,
            socName = displaySoc,
            totalRamGb = Math.round(totalRamGb * 10.0) / 10.0,
            deviceModel = Build.MODEL ?: "Android Device",
            isS24Ultra = isS24Ultra,
            supportsLocalNpuStudio = isEligible
        )
    }
}
