package com.photovance.app.network

import com.google.gson.annotations.SerializedName

data class StatusResponse(
    @SerializedName("ready") val ready: Boolean,
    @SerializedName("models") val models: Map<String, ModelInfo>?,
    @SerializedName("models_list") val modelsList: List<ModelInfo>? = null,
    @SerializedName("loras") val loras: Map<String, LoraInfo>? = null,
    @SerializedName("default_negative_prompt") val defaultNegativePrompt: String?
)

data class ModelInfo(
    @SerializedName("id") val id: String,
    @SerializedName("name") val name: String,
    @SerializedName("desc") val desc: String?,
    @SerializedName("type") val type: String?,
    @SerializedName("badge") val badge: String? = null
)

data class LoraInfo(
    @SerializedName("id") val id: String,
    @SerializedName("name") val name: String,
    @SerializedName("filename") val filename: String? = null,
    @SerializedName("trigger") val trigger: String? = null,
    @SerializedName("default_weight") val defaultWeight: Float? = null,
    @SerializedName("badge") val badge: String? = null,
    @SerializedName("desc") val desc: String? = null,
    @SerializedName("steps") val steps: Int? = null,
    @SerializedName("cfg") val cfg: Float? = null,
    @SerializedName("sampler") val sampler: String? = null,
    @SerializedName("scheduler") val scheduler: String? = null
)

data class LorasResponse(
    @SerializedName("loras") val loras: Map<String, LoraInfo>?,
    @SerializedName("default_lora") val defaultLora: String? = null
)

data class UploadRequest(
    @SerializedName("data") val data: String
)

data class UploadResponse(
    @SerializedName("image_id") val imageId: String?,
    @SerializedName("image_url") val imageUrl: String?,
    @SerializedName("width") val width: Int?,
    @SerializedName("height") val height: Int?,
    @SerializedName("error") val error: String?
)

data class ChangeDressRequest(
    @SerializedName("image_id") val imageId: String,
    @SerializedName("prompt") val prompt: String,
    @SerializedName("negative_prompt") val negativePrompt: String,
    @SerializedName("model_key") val modelKey: String,
    @SerializedName("mode") val mode: String,
    @SerializedName("aspect_ratio") val aspectRatio: String = "portrait",
    @SerializedName("target") val target: String = "dress",
    @SerializedName("expand") val expand: Int = 5,
    @SerializedName("keep_face") val keepFace: Boolean = true,
    @SerializedName("variations") val variations: Int = 1,
    @SerializedName("garment_id") val garmentId: String? = null,
    @SerializedName("lora_id") val loraId: String? = null,
    @SerializedName("lora_weight") val loraWeight: Float? = null
)

data class ChangeDressResponse(
    @SerializedName("success") val success: Boolean,
    @SerializedName("result_url") val resultUrl: String?,
    @SerializedName("preview_url") val previewUrl: String?,
    @SerializedName("elapsed") val elapsed: Double?,
    @SerializedName("model_name") val modelName: String?,
    @SerializedName("mode") val mode: String?,
    @SerializedName("coverage") val coverage: Double?,
    @SerializedName("results") val results: List<String>? = null,
    @SerializedName("error") val error: String?
)

data class HistoryItem(
    @SerializedName("id") val id: String,
    @SerializedName("url") val url: String,
    @SerializedName("prompt") val prompt: String,
    @SerializedName("model") val model: String?,
    @SerializedName("mode") val mode: String?,
    @SerializedName("elapsed") val elapsed: Double?
)

data class SampleResponse(
    @SerializedName("image_id") val imageId: String,
    @SerializedName("image_url") val imageUrl: String,
    @SerializedName("width") val width: Int?,
    @SerializedName("height") val height: Int?
)

