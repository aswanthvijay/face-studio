package com.photovance.app.ui

import android.app.Application
import android.app.Activity
import android.net.Uri
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.photovance.app.R
import com.photovance.app.media.savePortrait
import com.photovance.app.media.sharePortrait
import com.photovance.app.media.uploadPortrait
import com.photovance.app.network.ApiClient
import com.photovance.app.network.ChangeDressRequest
import com.photovance.app.network.LoraInfo
import com.photovance.app.ui.components.VISUAL_STYLE_PRESETS
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import okhttp3.HttpUrl.Companion.toHttpUrlOrNull

import com.photovance.app.auth.AuthRepository
import com.photovance.app.auth.PhotovanceAuthRepository
import com.photovance.app.billing.BillingManager
import com.photovance.app.billing.VipPlan
import com.photovance.app.engine.HybridStudioRouter
import com.photovance.app.engine.StudioEngineMode

data class Creation(
    val beforeUrl: String,
    val imageUrl: String,
    val prompt: String,
    val elapsed: Double?,
    val sourceImageId: String? = null,
    val studioUrl: String = "",
    val variationUrls: List<String> = emptyList(),
    val model: String? = null,
    val mode: String? = null
)

class StudioViewModel(
    application: Application,
    authRepositoryOverride: AuthRepository? = null
) : AndroidViewModel(application) {
    val authRepository: AuthRepository = authRepositoryOverride ?: PhotovanceAuthRepository(application)
    val hybridRouter: HybridStudioRouter = HybridStudioRouter(application)
    val billingManager: BillingManager = BillingManager(application)
    val currentUser = authRepository.currentUser
    val engineMode = hybridRouter.engineMode
    val isVip = billingManager.isVip
    val remainingCredits = billingManager.remainingCredits

    var showPaywall by mutableStateOf(false)
    var variationsCount by mutableIntStateOf(1)
    var garmentUri by mutableStateOf<Uri?>(null)
    private var uploadedGarmentId: String? = null
    var activeVariationIndex by mutableIntStateOf(0)

    var step by mutableIntStateOf(0)
        private set
    var photoUri by mutableStateOf<Uri?>(null)
        private set
    var isSample by mutableStateOf(false)
        private set
    var originalUrl by mutableStateOf<String?>(null)
        private set
    private var uploadedId: String? = null
    val hasPhoto get() = photoUri != null || isSample || (originalUrl != null && uploadedId != null)
    val photoModel: Any? get() = photoUri ?: originalUrl ?: if (isSample) R.drawable.cc0_studio else null

    var mode by mutableStateOf("in_place")
    var modelKey by mutableStateOf("cyberrealistic_xl")
    var aspectRatio by mutableStateOf("portrait")
    var target by mutableStateOf("dress")
    var category by mutableStateOf("All")
    var selectedPresetId by mutableStateOf<String?>("monaco_yacht")
        private set
    var prompt by mutableStateOf(VISUAL_STYLE_PRESETS.first().prompt)
        private set
    var negativePrompt by mutableStateOf("")

    var lorasMap by mutableStateOf<Map<String, LoraInfo>>(emptyMap())
    var selectedLoraId by mutableStateOf<String?>(null)
    var loraWeight by mutableFloatStateOf(0.75f)
    var isGenerating by mutableStateOf(false)
        private set
    var isExporting by mutableStateOf(false)
        private set
    var elapsedSeconds by mutableIntStateOf(0)
        private set
    var progressLabel by mutableStateOf("Preparing your photo")
        private set
    var online by mutableStateOf(false)
        private set
    var checkingConnection by mutableStateOf(false)
        private set
    var error by mutableStateOf<String?>(null)
        private set
    var message by mutableStateOf<String?>(null)
    var result by mutableStateOf<Creation?>(null)
        private set
    val recent = mutableStateListOf<Creation>()

    init { checkConnection() }

    fun selectPhoto(uri: Uri) {
        if (isGenerating) return
        photoUri = uri
        isSample = false
        originalUrl = null
        uploadedId = null
        result = null
        error = null
    }

    fun selectSample() {
        if (isGenerating) return
        photoUri = null
        isSample = true
        originalUrl = null
        uploadedId = null
        result = null
        error = null
    }

    fun goToStep(destination: Int) {
        if (isGenerating) return
        if (destination == 0 || (destination == 1 && hasPhoto) || (destination == 2 && result != null)) {
            step = destination
            error = null
        }
    }

    fun selectPreset(id: String) {
        if (isGenerating) return
        val preset = VISUAL_STYLE_PRESETS.first { it.id == id }
        selectedPresetId = id
        prompt = preset.prompt
        // A style changes the look, while the user's chosen editing mode stays intact.
    }

    fun editPrompt(value: String) {
        prompt = value
        selectedPresetId = null
    }

    fun selectLora(id: String?) {
        selectedLoraId = id
        if (id != null) {
            val lora = lorasMap[id]
            loraWeight = lora?.defaultWeight ?: 0.75f
        }
    }

    fun setLoraWeightValue(weight: Float) {
        loraWeight = weight
    }

    fun injectLoraTrigger() {
        val lora = selectedLoraId?.let { lorasMap[it] } ?: return
        val trigger = lora.trigger?.trim() ?: return
        if (trigger.isNotEmpty() && !prompt.contains(trigger)) {
            prompt = if (prompt.isBlank()) trigger else "$prompt, $trigger"
        }
    }

    fun checkConnection(url: String? = null) {
        if (checkingConnection || isGenerating) return
        if (url == null && ApiClient.baseUrl.isBlank()) {
            online = false
            return
        }
        if (url != null) {
            val parsed = url.trim().toHttpUrlOrNull()
            if (parsed == null || parsed.username.isNotEmpty() || parsed.password.isNotEmpty() ||
                parsed.query != null || parsed.fragment != null) {
                message = "Enter a valid HTTPS studio address."
                return
            }
            try {
                ApiClient.saveBaseUrl(getApplication(), parsed.toString())
            } catch (failure: IllegalArgumentException) {
                message = failure.message ?: "Studio addresses must use HTTPS."
                return
            }
            uploadedId = null
            originalUrl = null
        }
        checkingConnection = true
        viewModelScope.launch {
            try {
                val status = ApiClient.api.getStatus()
                online = status.ready
                if (negativePrompt.isBlank()) negativePrompt = status.defaultNegativePrompt.orEmpty()
                if (status.loras != null && status.loras.isNotEmpty()) {
                    lorasMap = status.loras
                } else {
                    try {
                        val loraRes = ApiClient.api.getLoras()
                        if (loraRes.loras != null) {
                            lorasMap = loraRes.loras
                        }
                    } catch (_: Exception) {
                        // Optional fallback
                    }
                }
                if (url != null) message = if (online) "Studio connected" else "Connected. The studio is warming up."
            } catch (cancelled: CancellationException) { throw cancelled
            } catch (_: Exception) {
                online = false
                if (url != null) message = "Couldn't connect. Check the address and your network."
            } finally { checkingConnection = false }
        }
    }

    fun setGarment(uri: Uri?) {
        garmentUri = uri
        uploadedGarmentId = null
    }

    fun selectVariation(index: Int) {
        val current = result ?: return
        if (index in current.variationUrls.indices) {
            activeVariationIndex = index
            result = current.copy(imageUrl = current.variationUrls[index])
        }
    }

    fun openPaywall() {
        showPaywall = true
    }

    fun dismissPaywall() {
        showPaywall = false
    }

    fun subscribeVip(activity: Activity, plan: VipPlan) {
        billingManager.subscribe(activity, plan) { success, result ->
            if (success) showPaywall = false
            message = result
        }
    }

    fun restorePurchases() {
        billingManager.restorePurchases { restored, result ->
            if (restored) showPaywall = false
            message = result
        }
    }

    fun generate() {
        if (isGenerating || !hasPhoto) return
        if (!billingManager.canGenerate()) {
            showPaywall = true
            return
        }
        if (prompt.isBlank()) { error = "Choose a style or describe the look you'd like."; return }
        val requestedPrompt = prompt.trim()
        val requestedMode = mode
        val requestedModel = modelKey
        val requestedRatio = aspectRatio
        val requestedTarget = target
        val requestedNegative = negativePrompt
        val requestedVariations = variationsCount
        val requestedLoraId = selectedLoraId
        val requestedLoraWeight = loraWeight
        isGenerating = true
        error = null
        elapsedSeconds = 0
        progressLabel = "Preparing your photo"
        viewModelScope.launch {
            val timer = launch { while (true) { delay(1000); elapsedSeconds++ } }
            try {
                if (uploadedId == null) {
                    val inputUri = if (isSample) Uri.parse("android.resource://${getApplication<Application>().packageName}/${R.drawable.cc0_studio}")
                        else checkNotNull(photoUri)
                    val upload = uploadPortrait(getApplication(), inputUri)
                    val nextId = upload.imageId ?: throw IllegalStateException(upload.error ?: "Couldn't upload your photo.")
                    val nextOriginal = upload.imageUrl?.let(ApiClient::getFullUrl)
                        ?: throw IllegalStateException("The studio didn't return your photo. Try again.")
                    uploadedId = nextId
                    originalUrl = nextOriginal
                }

                // Upload reference garment if present
                if (garmentUri != null && uploadedGarmentId == null) {
                    progressLabel = "Analyzing reference garment"
                    val garmentUpload = uploadPortrait(getApplication(), checkNotNull(garmentUri))
                    uploadedGarmentId = garmentUpload.imageId
                }

                progressLabel = if (requestedVariations > 1) "Generating $requestedVariations studio variations" else "Creating your new look"
                val response = ApiClient.api.changeDress(ChangeDressRequest(
                    imageId = checkNotNull(uploadedId), prompt = requestedPrompt,
                    negativePrompt = requestedNegative, modelKey = requestedModel,
                    mode = requestedMode, aspectRatio = requestedRatio, target = requestedTarget, keepFace = true,
                    variations = requestedVariations, garmentId = uploadedGarmentId,
                    loraId = requestedLoraId, loraWeight = requestedLoraWeight
                ))
                if (!response.success || response.resultUrl == null) {
                    throw IllegalStateException(response.error ?: "Couldn't create this look. Please try again.")
                }
                val variationList = if (!response.results.isNullOrEmpty()) {
                    response.results.map { ApiClient.getFullUrl(it) }
                } else {
                    listOf(ApiClient.getFullUrl(response.resultUrl))
                }
                val creation = Creation(
                    checkNotNull(originalUrl),
                    variationList.first(),
                    requestedPrompt,
                    response.elapsed,
                    uploadedId,
                    ApiClient.baseUrl,
                    variationList,
                    requestedModel,
                    requestedMode
                )
                activeVariationIndex = 0
                result = creation
                recent.add(0, creation)
                if (recent.size > 20) recent.removeAt(recent.lastIndex)
                billingManager.consumeCreditIfFree()
                online = true
                step = 2
            } catch (cancelled: CancellationException) { throw cancelled
            } catch (failure: Exception) {
                error = if (failure is IllegalStateException || failure is IllegalArgumentException) {
                    failure.message ?: "Couldn't create this look. Try again."
                } else "Couldn't reach the studio. Check your connection in Settings, then try again."
            } finally {
                timer.cancel()
                isGenerating = false
            }
        }
    }

    fun openCreation(creation: Creation) {
        if (isGenerating) return
        result = creation
        photoUri = null
        isSample = false
        uploadedId = if (creation.studioUrl == ApiClient.baseUrl) creation.sourceImageId else null
        originalUrl = if (uploadedId != null) creation.beforeUrl else null
        prompt = creation.prompt
        selectedPresetId = null
        error = null
        step = 2
    }

    fun save() = export(share = false)
    fun share() = export(share = true)
    fun shareStory() = export(share = true, asStory = true)

    private fun export(share: Boolean, asStory: Boolean = false) {
        val photo = result ?: return
        if (isExporting) return
        isExporting = true
        viewModelScope.launch {
            try {
                val vip = isVip.value
                if (share) {
                    sharePortrait(getApplication(), photo.imageUrl, isVip = vip, asStory = asStory)
                } else {
                    savePortrait(getApplication(), photo.imageUrl, isVip = vip)
                    message = "Photo saved to your gallery"
                }
            } catch (cancelled: CancellationException) { throw cancelled
            } catch (_: Exception) {
                message = if (share) "Couldn't share this photo. Try again." else "Couldn't save this photo. Try again."
            } finally { isExporting = false }
        }
    }

    override fun onCleared() {
        authRepository.close()
        billingManager.close()
        super.onCleared()
    }
}
