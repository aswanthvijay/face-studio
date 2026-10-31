package com.photovance.app.network

import android.content.Context
import okhttp3.OkHttpClient
import okhttp3.HttpUrl.Companion.toHttpUrl
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.POST
import java.util.concurrent.TimeUnit

interface ApiService {
    @GET("api/status")
    suspend fun getStatus(): StatusResponse

    @GET("api/outfit-progress")
    suspend fun getOutfitProgress(@retrofit2.http.Query("id") requestId: String): OutfitProgressResponse

    @POST("api/upload")
    suspend fun uploadImage(@Body request: UploadRequest): UploadResponse

    @POST("api/change-dress")
    suspend fun changeDress(@Body request: ChangeDressRequest): ChangeDressResponse

    @GET("api/history")
    suspend fun getHistory(): List<HistoryItem>

    @GET("api/sample")
    suspend fun getSample(): SampleResponse

    @GET("api/loras")
    suspend fun getLoras(): LorasResponse
}

object ApiClient {
    private const val PREFS_NAME = "photovance_prefs"
    private const val KEY_BASE_URL = "base_url"
    const val DEFAULT_URL = "http://192.168.1.230:7870/"

    var baseUrl: String = DEFAULT_URL
        set(value) {
            field = if (value.endsWith("/")) value else "$value/"
            retrofitInstance = null
            apiServiceInstance = null
        }

    fun init(context: Context) {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val saved = prefs.getString(KEY_BASE_URL, null)
        // Migrate the old emulator default; preserve custom server addresses.
        if (saved?.trimEnd('/') == "http://10.0.2.2:7870") {
            saveBaseUrl(context, DEFAULT_URL)
        } else if (!saved.isNullOrBlank()) {
            baseUrl = saved
        }
    }

    fun saveBaseUrl(context: Context, newUrl: String) {
        val formatted = if (newUrl.endsWith("/")) newUrl else "$newUrl/"
        baseUrl = formatted
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        prefs.edit().putString(KEY_BASE_URL, formatted).apply()
    }

    private var retrofitInstance: Retrofit? = null
    private var apiServiceInstance: ApiService? = null

    private fun getOkHttpClient(): OkHttpClient {
        val logging = HttpLoggingInterceptor().apply {
            level = HttpLoggingInterceptor.Level.BASIC
        }
        return OkHttpClient.Builder()
            .addInterceptor(logging)
            .connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(180, TimeUnit.SECONDS)
            .writeTimeout(180, TimeUnit.SECONDS)
            .build()
    }

    val api: ApiService
        get() {
            if (apiServiceInstance == null) {
                val retrofit = Retrofit.Builder()
                    .baseUrl(baseUrl)
                    .client(getOkHttpClient())
                    .addConverterFactory(GsonConverterFactory.create())
                    .build()
                retrofitInstance = retrofit
                apiServiceInstance = retrofit.create(ApiService::class.java)
            }
            return apiServiceInstance!!
        }

    fun getFullUrl(relativeUrl: String): String {
        return baseUrl.toHttpUrl().resolve(relativeUrl)?.toString()
            ?: throw IllegalArgumentException("The studio returned an invalid photo address.")
    }
}

