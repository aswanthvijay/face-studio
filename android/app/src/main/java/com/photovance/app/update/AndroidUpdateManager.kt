package com.photovance.app.update

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.IntentSenderRequest
import com.google.android.play.core.appupdate.AppUpdateInfo
import com.google.android.play.core.appupdate.AppUpdateManager
import com.google.android.play.core.appupdate.AppUpdateManagerFactory
import com.google.android.play.core.appupdate.AppUpdateOptions
import com.google.android.play.core.install.model.AppUpdateType
import com.google.android.play.core.install.model.UpdateAvailability
import okhttp3.OkHttpClient
import okhttp3.Request
import org.json.JSONObject
import java.io.IOException
import java.util.concurrent.TimeUnit

sealed class UpdateOffer {
    data class PlayStore(val info: AppUpdateInfo) : UpdateOffer()
    data class Sideloaded(val version: String, val url: String) : UpdateOffer()
}

sealed class UpdateCheckResult {
    data class Available(val offer: UpdateOffer) : UpdateCheckResult()
    object UpToDate : UpdateCheckResult()
    data class Failed(val message: String) : UpdateCheckResult()
}

class AndroidUpdateManager(context: Context) {
    private val appContext = context.applicationContext
    private val playManager: AppUpdateManager = AppUpdateManagerFactory.create(appContext)
    private val client = OkHttpClient.Builder()
        .callTimeout(15, TimeUnit.SECONDS)
        .build()

    fun checkForUpdate(onResult: (UpdateCheckResult) -> Unit) {
        if (isInstalledFromPlay()) {
            playManager.appUpdateInfo
                .addOnSuccessListener { info ->
                    val availability = info.updateAvailability()
                    val available = availability == UpdateAvailability.UPDATE_AVAILABLE ||
                        availability == UpdateAvailability.DEVELOPER_TRIGGERED_UPDATE_IN_PROGRESS
                    onResult(if (available) UpdateCheckResult.Available(UpdateOffer.PlayStore(info))
                    else UpdateCheckResult.UpToDate)
                }
                .addOnFailureListener { error ->
                    onResult(UpdateCheckResult.Failed(error.message ?: "Could not check Google Play for updates."))
                }
            return
        }

        val request = Request.Builder()
            .url(RELEASES_API)
            .header("Accept", "application/vnd.github+json")
            .header("User-Agent", "Photovance-Android")
            .build()
        client.newCall(request).enqueue(object : okhttp3.Callback {
            override fun onFailure(call: okhttp3.Call, error: IOException) {
                onResult(UpdateCheckResult.Failed(error.message ?: "Could not check for APK updates."))
            }

            override fun onResponse(call: okhttp3.Call, response: okhttp3.Response) {
                response.use {
                    if (!it.isSuccessful) {
                        onResult(UpdateCheckResult.Failed("Release check failed with HTTP ${it.code}."))
                        return
                    }
                    try {
                        val releases = org.json.JSONArray(it.body?.string().orEmpty())
                        val releasesByVersion = (0 until releases.length())
                            .mapNotNull { index ->
                                val release = releases.optJSONObject(index) ?: return@mapNotNull null
                                val tag = release.optString("tag_name")
                                val version = ANDROID_TAG_REGEX.matchEntire(tag)
                                    ?.groupValues?.get(1) ?: return@mapNotNull null
                                version to release
                            }
                            .filter { (version, _) -> isNewer(version, currentVersionName()) }
                            .sortedWith { left, right -> compareVersions(left.first, right.first) }
                        val availableRelease = releasesByVersion.lastOrNull()
                        if (availableRelease == null) {
                            onResult(UpdateCheckResult.UpToDate)
                            return
                        }
                        val (version, release) = availableRelease
                        val tag = release.optString("tag_name")
                        val expectedAssetName = "photovance-android-$version.apk"
                        val asset = release.optJSONArray("assets")
                            ?.let { assets ->
                                (0 until assets.length())
                                    .map { index -> assets.getJSONObject(index) }
                                    .firstOrNull { item ->
                                        item.optString("name") == expectedAssetName
                                    }
                            }
                        if (asset == null) {
                            onResult(UpdateCheckResult.Failed("The Android release is missing its APK asset."))
                        } else {
                            val url = asset.optString("browser_download_url")
                            val expectedUrl =
                                "https://github.com/$REPOSITORY/releases/download/$tag/$expectedAssetName"
                            if (url != expectedUrl) {
                                onResult(UpdateCheckResult.Failed("The APK release URL is not from the official repository."))
                            } else {
                                onResult(UpdateCheckResult.Available(UpdateOffer.Sideloaded(version, url)))
                            }
                        }
                    } catch (error: Exception) {
                        onResult(UpdateCheckResult.Failed(error.message ?: "The release response was invalid."))
                    }
                }
            }
        })
    }

    fun startPlayUpdate(
        activity: Activity,
        offer: UpdateOffer.PlayStore,
        launcher: ActivityResultLauncher<IntentSenderRequest>
    ): Boolean = playManager.startUpdateFlowForResult(
        offer.info,
        launcher,
        AppUpdateOptions.newBuilder(AppUpdateType.IMMEDIATE).build()
    )

    fun openApkDownload(context: Context, offer: UpdateOffer.Sideloaded) {
        context.startActivity(
            Intent(Intent.ACTION_VIEW, Uri.parse(offer.url))
                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        )
    }

    private fun isInstalledFromPlay(): Boolean {
        val installer = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            appContext.packageManager.getInstallSourceInfo(appContext.packageName).installingPackageName
        } else {
            @Suppress("DEPRECATION")
            appContext.packageManager.getInstallerPackageName(appContext.packageName)
        }
        return installer == GOOGLE_PLAY_PACKAGE
    }

    private fun currentVersionName(): String {
        val info = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            appContext.packageManager.getPackageInfo(
                appContext.packageName,
                PackageManager.PackageInfoFlags.of(0)
            )
        } else {
            @Suppress("DEPRECATION")
            appContext.packageManager.getPackageInfo(appContext.packageName, 0)
        }
        return info.versionName.orEmpty()
    }

    private fun isNewer(candidate: String, current: String): Boolean {
        val candidateParts = candidate.split('.').map { it.toIntOrNull() ?: return false }
        val currentParts = current.split('.').map { it.toIntOrNull() ?: return false }
        return compareVersionParts(candidateParts, currentParts) > 0
    }

    private fun compareVersions(left: String, right: String): Int {
        val leftParts = left.split('.').map { it.toInt() }
        val rightParts = right.split('.').map { it.toInt() }
        return compareVersionParts(leftParts, rightParts)
    }

    private fun compareVersionParts(left: List<Int>, right: List<Int>): Int {
        val size = maxOf(left.size, right.size)
        for (index in 0 until size) {
            val comparison = (left.getOrElse(index) { 0 }).compareTo(right.getOrElse(index) { 0 })
            if (comparison != 0) return comparison
        }
        return 0
    }

    companion object {
        private const val GOOGLE_PLAY_PACKAGE = "com.android.vending"
        private const val REPOSITORY = "aswanthvijay/face-studio"
        private const val RELEASES_API = "https://api.github.com/repos/$REPOSITORY/releases?per_page=30"
        private val ANDROID_TAG_REGEX = Regex("^android-v(\\d+\\.\\d+\\.\\d+)$")
    }
}
