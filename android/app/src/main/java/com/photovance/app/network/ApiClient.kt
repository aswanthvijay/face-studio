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
    private val LOOPBACK_HOSTS = setOf("localhost", "127.0.0.1", "::1", "[::1]", "10.0.2.2")
    const val DEFAULT_URL = ""

    var baseUrl: String = DEFAULT_URL
        set(value) {
            field = if (value.isBlank()) "" else if (value.endsWith("/")) value else "$value/"
            retrofitInstance = null
            apiServiceInstance = null
        }

    fun init(context: Context) {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val saved = prefs.getString(KEY_BASE_URL, null)
        if (!saved.isNullOrBlank()) {
            val parsed = saved.trim().toHttpUrlOrNull()
            if (parsed != null && isAllowedUrl(parsed.scheme, parsed.host)) {
                baseUrl = parsed.toString()
            } else {
                prefs.edit().remove(KEY_BASE_URL).apply()
            }
        }
    }

    fun saveBaseUrl(context: Context, newUrl: String) {
        val parsed = newUrl.trim().toHttpUrlOrNull()
            ?: throw IllegalArgumentException("Enter a valid HTTPS studio address.")
        if (!isAllowedUrl(parsed.scheme, parsed.host)) {
            throw IllegalArgumentException("Studio addresses must use HTTPS. HTTP is allowed only for a local development server.")
        }
        val formatted = parsed.toString()
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
                check(baseUrl.isNotBlank()) { "Connect to a studio server before making requests." }
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
        check(baseUrl.isNotBlank()) { "Connect to a studio server before resolving photo addresses." }
        return baseUrl.toHttpUrl().resolve(relativeUrl)?.toString()
            ?: throw IllegalArgumentException("The studio returned an invalid photo address.")
    }

    private fun isAllowedUrl(scheme: String, host: String): Boolean =
        scheme == "https" || (scheme == "http" && host in LOOPBACK_HOSTS)

}
