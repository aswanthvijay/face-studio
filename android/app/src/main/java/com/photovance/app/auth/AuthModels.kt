package com.photovance.app.auth

data class AuthUser(
    val uid: String,
    val displayName: String? = null,
    val email: String? = null,
    val phoneNumber: String? = null,
    val photoUrl: String? = null,
    val isGuest: Boolean = false,
    val isVip: Boolean = false
) {
    val displayIdentifier: String
        get() = displayName ?: email ?: phoneNumber ?: if (isGuest) "Guest Creator" else "Photovance Creator"
}

sealed class AuthState {
    object Idle : AuthState()
    object Loading : AuthState()
    data class CodeSent(val verificationId: String, val phoneNumber: String) : AuthState()
    data class Authenticated(val user: AuthUser) : AuthState()
    data class Error(val message: String) : AuthState()
}

enum class AuthProviderType {
    GOOGLE,
    APPLE,
    EMAIL_MAGIC_LINK,
    PHONE_SMS,
    GUEST
}
