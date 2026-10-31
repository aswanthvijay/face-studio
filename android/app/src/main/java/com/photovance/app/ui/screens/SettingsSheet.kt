package com.photovance.app.ui.screens

import android.app.Activity
import android.app.Activity.RESULT_OK
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.IntentSenderRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CheckCircleOutline
import androidx.compose.material.icons.filled.CloudOff
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import com.photovance.app.network.ApiClient
import com.photovance.app.ui.StudioViewModel
import com.photovance.app.update.AndroidUpdateManager
import com.photovance.app.update.UpdateCheckResult
import com.photovance.app.update.UpdateOffer
import com.photovance.app.ui.components.StudioButton
import com.photovance.app.ui.theme.AppAppearance
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsSheet(studio: StudioViewModel, appearance: AppAppearance,
    onAppearance: (AppAppearance) -> Unit, onDismiss: () -> Unit, onLinkAccount: () -> Unit) {
    var address by remember { mutableStateOf(ApiClient.baseUrl) }
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val activity = context as? Activity
    val updateManager = remember(context) { AndroidUpdateManager(context) }
    var updateOffer by remember { mutableStateOf<UpdateOffer?>(null) }
    var updateMessage by remember { mutableStateOf<String?>(null) }
    var checkingUpdates by remember { mutableStateOf(false) }
    var confirmAccountDeletion by remember { mutableStateOf(false) }
    var accountDeletionError by remember { mutableStateOf<String?>(null) }
    val updateLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.StartIntentSenderForResult()
    ) { result ->
        if (result.resultCode != RESULT_OK) updateMessage = "The update was canceled."
    }
    val sheet = rememberModalBottomSheetState(skipPartiallyExpanded = true)
    ModalBottomSheet(onDismissRequest = onDismiss, sheetState = sheet,
        containerColor = MaterialTheme.colorScheme.surface) {
        Column(Modifier.fillMaxWidth().verticalScroll(rememberScrollState()).imePadding()
            .padding(horizontal = 24.dp).padding(bottom = 32.dp), verticalArrangement = Arrangement.spacedBy(18.dp)) {
            Text("Make yourself at home.", style = MaterialTheme.typography.headlineMedium)
            Text("Appearance", style = MaterialTheme.typography.titleSmall)
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                AppAppearance.entries.forEach { option ->
                    FilterChip(selected = appearance == option, onClick = { onAppearance(option) },
                        label = { Text(option.title) }, modifier = Modifier.weight(1f))
                }
            }
            HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
            Text("Your studio connection", style = MaterialTheme.typography.titleLarge)
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                Icon(if (studio.online) Icons.Default.CheckCircleOutline else Icons.Default.CloudOff,
                    null, tint = MaterialTheme.colorScheme.primary)
                Text(if (studio.checkingConnection) "Checking connection…" else if (studio.online) "Connected and ready"
                    else "Not connected", style = MaterialTheme.typography.bodyMedium)
            }
            Text("Connect to the server that creates your photos. You can explore styles while offline.",
                color = MaterialTheme.colorScheme.onSurfaceVariant, style = MaterialTheme.typography.bodyMedium)
            OutlinedTextField(value = address, onValueChange = { address = it }, label = { Text("Studio address") },
                placeholder = { Text("https://your-studio.example/") }, singleLine = true,
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Uri),
                modifier = Modifier.fillMaxWidth(), shape = MaterialTheme.shapes.small,
                enabled = !studio.isGenerating && !studio.checkingConnection)
            StudioButton("Save and connect", onClick = { studio.checkConnection(address) },
                enabled = !studio.isGenerating, loading = studio.checkingConnection)
            HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
            val currentUser by studio.currentUser.collectAsState()
            Text("Account & Security", style = MaterialTheme.typography.titleLarge)
            Surface(shape = MaterialTheme.shapes.medium, color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f),
                border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant)) {
                Column(Modifier.fillMaxWidth().padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text(if (currentUser?.isGuest == true) "Guest Creator" else (currentUser?.displayIdentifier ?: "Signed In"),
                        style = MaterialTheme.typography.titleMedium)
                    Text(if (currentUser?.isGuest == true) "Exploring studio without account" else (currentUser?.email ?: currentUser?.phoneNumber ?: "Authenticated session"),
                        style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    if (currentUser?.isGuest == true) {
                        Text("Link an account to keep access to this guest profile on other devices.",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant)
                        OutlinedButton(onClick = onLinkAccount, modifier = Modifier.fillMaxWidth()) {
                            Text("Create or link an account")
                        }
                    }
                    if (currentUser != null && !currentUser.isGuest) {
                        OutlinedButton(onClick = { studio.authRepository.signOut(); onDismiss() },
                            modifier = Modifier.fillMaxWidth().padding(top = 4.dp)) {
                            Text("Sign out of Photovance")
                        }
                    }
                    if (currentUser != null) {
                        TextButton(onClick = { confirmAccountDeletion = true },
                            modifier = Modifier.fillMaxWidth()) {
                            Text("Delete account", color = MaterialTheme.colorScheme.error)
                        }
                    }
                }
            }
            accountDeletionError?.let {
                Text(it, color = MaterialTheme.colorScheme.error,
                    style = MaterialTheme.typography.bodySmall)
            }

            HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
            val engineMode by studio.engineMode.collectAsState()
            val deviceSpec = studio.hybridRouter.deviceSpec
            Text("Hardware & Engine", style = MaterialTheme.typography.titleLarge)
            Surface(shape = MaterialTheme.shapes.medium,
                color = if (deviceSpec.supportsLocalNpuStudio) MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.3f) else MaterialTheme.colorScheme.surface,
                border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant)) {
                Column(Modifier.fillMaxWidth().padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text(if (deviceSpec.supportsLocalNpuStudio) "⚡ High-Performance NPU Ready" else "☁️ Standard Cloud Processing",
                            style = MaterialTheme.typography.titleSmall, color = MaterialTheme.colorScheme.primary)
                        Spacer(Modifier.weight(1f))
                        Text("${deviceSpec.totalRamGb} GB RAM", style = MaterialTheme.typography.labelSmall)
                    }
                    Text("${deviceSpec.deviceModel} · ${deviceSpec.socName}",
                        style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    if (deviceSpec.supportsLocalNpuStudio) {
                        Row(Modifier.fillMaxWidth().padding(top = 6.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            FilterChip(
                                selected = engineMode == com.photovance.app.engine.StudioEngineMode.LOCAL_OFFLINE,
                                onClick = { studio.hybridRouter.switchMode(com.photovance.app.engine.StudioEngineMode.LOCAL_OFFLINE) },
                                label = { Text("⚡ Local Studio (Offline)") },
                                modifier = Modifier.weight(1f)
                            )
                            FilterChip(
                                selected = engineMode == com.photovance.app.engine.StudioEngineMode.CLOUD_8K,
                                onClick = { studio.hybridRouter.switchMode(com.photovance.app.engine.StudioEngineMode.CLOUD_8K) },
                                label = { Text("☁️ Cloud 8K Master") },
                                modifier = Modifier.weight(1f)
                            )
                        }
                    }
                }
            }

            HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
            Text("Creation options", style = MaterialTheme.typography.titleLarge)
            Text("Rendering quality", style = MaterialTheme.typography.titleSmall)
            listOf(
                Triple("cyberrealistic_xl", "CyberRealistic XL Master™", "DSLR micro-pores · Natural epidermal depth · Zero plastic blur"),
                Triple("epicrealism_xl", "epiCRealism XL Editorial™", "Cinematic contrast · Studio illumination · Authentic fabric drape"),
                Triple("realvis_xl", "RealVisXL V5.0 Benchmark™", "Razor-sharp portrait detail · High-speed analog realism"),
                Triple("flux_klein", "Flux-2-Klein 4B Turbo™", "Next-gen DiT · Unrestricted prompt adherence · Studio lighting")
            ).forEach { (key, title, detail) ->
                Surface(shape = MaterialTheme.shapes.medium,
                    color = if (studio.modelKey == key) MaterialTheme.colorScheme.primaryContainer else MaterialTheme.colorScheme.surface,
                    border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant)) {
                    Row(Modifier.fillMaxWidth().clip(MaterialTheme.shapes.medium)
                        .selectable(selected = studio.modelKey == key, enabled = !studio.isGenerating, role = Role.RadioButton,
                            onClick = { studio.modelKey = key }).padding(12.dp),
                        verticalAlignment = Alignment.CenterVertically) {
                        RadioButton(studio.modelKey == key, onClick = null, enabled = !studio.isGenerating)
                        Spacer(Modifier.width(14.dp))
                        Column(Modifier.weight(1f)) {
                            Text(title, style = MaterialTheme.typography.titleSmall)
                            Text(detail, style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant)
                        }
                    }
                }
            }
            OutlinedTextField(value = studio.negativePrompt, onValueChange = { studio.negativePrompt = it },
                label = { Text("Things to avoid (optional)") }, minLines = 2, maxLines = 5,
                modifier = Modifier.fillMaxWidth(), shape = MaterialTheme.shapes.medium, enabled = !studio.isGenerating)
            Text("Photovance · Pro Photo Studio", style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant)
            HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
            Text("App updates", style = MaterialTheme.typography.titleLarge)
            Text("Updates use Google Play for Play Store installs and the official signed APK for sideloaded installs.",
                color = MaterialTheme.colorScheme.onSurfaceVariant, style = MaterialTheme.typography.bodyMedium)
            StudioButton(
                text = when (updateOffer) {
                    is UpdateOffer.PlayStore -> "Update now"
                    is UpdateOffer.Sideloaded -> "Download update"
                    null -> "Check for updates"
                },
                onClick = {
                    when (val offer = updateOffer) {
                        is UpdateOffer.PlayStore -> {
                            if (activity == null) updateMessage = "Open Photovance on your Android device to install updates."
                            else {
                                val started = updateManager.startPlayUpdate(activity, offer, updateLauncher)
                                if (!started) updateMessage = "Google Play could not start the update. Try again later."
                            }
                        }
                        is UpdateOffer.Sideloaded -> {
                            updateManager.openApkDownload(context, offer)
                            updateMessage = "The signed APK download was opened in your browser."
                        }
                        null -> {
                            checkingUpdates = true
                            updateManager.checkForUpdate { result ->
                                scope.launch {
                                    checkingUpdates = false
                                    when (result) {
                                        is UpdateCheckResult.Available -> {
                                            updateOffer = result.offer
                                            updateMessage = when (val available = result.offer) {
                                                is UpdateOffer.PlayStore -> "An update is available in Google Play."
                                                is UpdateOffer.Sideloaded -> "Version ${available.version} is available."
                                            }
                                        }
                                        UpdateCheckResult.UpToDate -> {
                                            updateOffer = null
                                            updateMessage = "Photovance is up to date."
                                        }
                                        is UpdateCheckResult.Failed -> updateMessage = result.message
                                    }
                                }
                            }
                        }
                    }
                },
                enabled = !checkingUpdates,
                loading = checkingUpdates
            )
            updateMessage?.let {
                Text(it, color = MaterialTheme.colorScheme.onSurfaceVariant,
                    style = MaterialTheme.typography.bodySmall)
            }
            TextButton(onClick = onDismiss, modifier = Modifier.fillMaxWidth()) { Text("Done") }
        }
    }
    if (confirmAccountDeletion) {
        AlertDialog(
            onDismissRequest = { confirmAccountDeletion = false },
            title = { Text("Delete your account?") },
            text = {
                Text("This permanently removes your Photovance account and its subscription verification records. Photos already sent to a studio server follow that server's retention policy.")
            },
            confirmButton = {
                TextButton(onClick = {
                    confirmAccountDeletion = false
                    studio.authRepository.deleteAccount { success, result ->
                        if (success) {
                            studio.message = result
                            onDismiss()
                        } else {
                            accountDeletionError = result ?: "Could not delete the account."
                        }
                    }
                }) {
                    Text("Delete account", color = MaterialTheme.colorScheme.error)
                }
            },
            dismissButton = {
                TextButton(onClick = { confirmAccountDeletion = false }) { Text("Cancel") }
            }
        )
    }
}
