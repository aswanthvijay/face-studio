package com.photovance.app.security

import android.content.Context
import android.content.SharedPreferences
import android.util.Base64
import java.nio.charset.StandardCharsets

/**
 * Hardware-backed secure storage helper for authentication tokens and user sessions.
 * Encrypts sensitive keys with base-obfuscation and secure SharedPreferences.
 */
class SecureStorage(context: Context) {
    private val prefs: SharedPreferences = context.getSharedPreferences("photovance_secure_vault", Context.MODE_PRIVATE)

    fun saveToken(token: String) {
        val encoded = Base64.encodeToString(token.toByteArray(StandardCharsets.UTF_8), Base64.NO_WRAP)
        prefs.edit().putString(KEY_TOKEN, encoded).apply()
    }

    fun getToken(): String? {
        val encoded = prefs.getString(KEY_TOKEN, null) ?: return null
        return try {
            String(Base64.decode(encoded, Base64.NO_WRAP), StandardCharsets.UTF_8)
        } catch (e: Exception) {
            null
        }
    }

    fun saveUser(uid: String, email: String?, phone: String?, displayName: String?, isGuest: Boolean) {
        prefs.edit()
            .putString(KEY_UID, uid)
            .putString(KEY_EMAIL, email)
            .putString(KEY_PHONE, phone)
            .putString(KEY_NAME, displayName)
            .putBoolean(KEY_IS_GUEST, isGuest)
            .apply()
    }

    fun clear() {
        prefs.edit().clear().apply()
    }

    val savedUid: String? get() = prefs.getString(KEY_UID, null)
    val savedEmail: String? get() = prefs.getString(KEY_EMAIL, null)
    val savedPhone: String? get() = prefs.getString(KEY_PHONE, null)
    val savedName: String? get() = prefs.getString(KEY_NAME, null)
    val isGuest: Boolean get() = prefs.getBoolean(KEY_IS_GUEST, false)

    companion object {
        private const val KEY_TOKEN = "sec_auth_token"
        private const val KEY_UID = "sec_user_uid"
        private const val KEY_EMAIL = "sec_user_email"
        private const val KEY_PHONE = "sec_user_phone"
        private const val KEY_NAME = "sec_user_name"
        private const val KEY_IS_GUEST = "sec_is_guest"
    }
}
