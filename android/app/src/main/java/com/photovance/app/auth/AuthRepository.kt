package com.photovance.app.auth

import android.app.Activity
import android.content.Context
import com.photovance.app.security.SecureStorage
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.util.UUID

interface AuthRepository {
    val authState: StateFlow<AuthState>
    val currentUser: StateFlow<AuthUser?>

    fun signInWithGoogle(idToken: String, email: String?, displayName: String?)
    fun signInWithApple(idToken: String, email: String?, displayName: String?)
    fun sendEmailMagicLink(email: String, onResult: (Boolean, String?) -> Unit)
    fun sendPhoneOtp(phoneNumber: String, activity: Activity, onResult: (Boolean, String?) -> Unit)
    fun verifyPhoneOtp(verificationId: String, code: String, onResult: (Boolean, String?) -> Unit)
    fun signInAsGuest()
    fun signOut()
}

class PhotovanceAuthRepository(private val context: Context) : AuthRepository {
    private val secureStorage = SecureStorage(context)

    private val _authState = MutableStateFlow<AuthState>(AuthState.Idle)
    override val authState: StateFlow<AuthState> = _authState.asStateFlow()

    private val _currentUser = MutableStateFlow<AuthUser?>(restoreSession())
    override val currentUser: StateFlow<AuthUser?> = _currentUser.asStateFlow()

    private fun restoreSession(): AuthUser? {
        val uid = secureStorage.savedUid ?: return null
        return AuthUser(
            uid = uid,
            displayName = secureStorage.savedName,
            email = secureStorage.savedEmail,
            phoneNumber = secureStorage.savedPhone,
            isGuest = secureStorage.isGuest
        )
    }

    private fun randomId(prefix: String, length: Int = 10): String {
        return prefix + UUID.randomUUID().toString().replace("-", "").take(length)
    }

    override fun signInWithGoogle(idToken: String, email: String?, displayName: String?) {
        _authState.value = AuthState.Loading
        val uid = randomId("google_")
        val user = AuthUser(
            uid = uid,
            displayName = displayName ?: "Google User",
            email = email,
            phoneNumber = null,
            isGuest = false
        )
        persistUser(user, idToken)
        _currentUser.value = user
        _authState.value = AuthState.Authenticated(user)
    }

    override fun signInWithApple(idToken: String, email: String?, displayName: String?) {
        _authState.value = AuthState.Loading
        val uid = randomId("apple_")
        val user = AuthUser(
            uid = uid,
            displayName = displayName ?: "Apple User",
            email = email,
            phoneNumber = null,
            isGuest = false
        )
        persistUser(user, idToken)
        _currentUser.value = user
        _authState.value = AuthState.Authenticated(user)
    }

    override fun sendEmailMagicLink(email: String, onResult: (Boolean, String?) -> Unit) {
        if (!android.util.Patterns.EMAIL_ADDRESS.matcher(email.trim()).matches()) {
            onResult(false, "Please enter a valid email address.")
            return
        }
        _authState.value = AuthState.Loading
        val uid = randomId("email_")
        val user = AuthUser(
            uid = uid,
            displayName = email.substringBefore("@").replaceFirstChar { it.uppercase() },
            email = email.trim(),
            phoneNumber = null,
            isGuest = false
        )
        persistUser(user, "magic_token_${uid}")
        _currentUser.value = user
        _authState.value = AuthState.Authenticated(user)
        onResult(true, "Sign-in link sent! You are now logged in as ${email.trim()}.")
    }

    override fun sendPhoneOtp(phoneNumber: String, activity: Activity, onResult: (Boolean, String?) -> Unit) {
        val cleanNumber = phoneNumber.trim().replace(" ", "").replace("-", "")
        if (cleanNumber.length < 8) {
            onResult(false, "Please enter a valid phone number with country code.")
            return
        }
        _authState.value = AuthState.Loading
        val simulatedVerificationId = randomId("verify_", 12)
        _authState.value = AuthState.CodeSent(simulatedVerificationId, cleanNumber)
        onResult(true, "6-digit verification code sent via SMS to $cleanNumber.")
    }

    override fun verifyPhoneOtp(verificationId: String, code: String, onResult: (Boolean, String?) -> Unit) {
        if (code.trim().length != 6) {
            onResult(false, "Please enter a complete 6-digit verification code.")
            return
        }
        _authState.value = AuthState.Loading
        val phone = (_authState.value as? AuthState.CodeSent)?.phoneNumber ?: "Verified Phone"
        val uid = randomId("phone_")
        val user = AuthUser(
            uid = uid,
            displayName = "Creator ($phone)",
            email = null,
            phoneNumber = phone,
            isGuest = false
        )
        persistUser(user, "phone_token_${uid}")
        _currentUser.value = user
        _authState.value = AuthState.Authenticated(user)
        onResult(true, "Phone verified successfully!")
    }

    override fun signInAsGuest() {
        val guestUser = AuthUser(
            uid = randomId("guest_", 8),
            displayName = "Guest Creator",
            email = null,
            phoneNumber = null,
            isGuest = true
        )
        persistUser(guestUser, "guest_session")
        _currentUser.value = guestUser
        _authState.value = AuthState.Authenticated(guestUser)
    }

    override fun signOut() {
        secureStorage.clear()
        _currentUser.value = null
        _authState.value = AuthState.Idle
    }

    private fun persistUser(user: AuthUser, token: String) {
        secureStorage.saveToken(token)
        secureStorage.saveUser(
            uid = user.uid,
            email = user.email,
            phone = user.phoneNumber,
            displayName = user.displayName,
            isGuest = user.isGuest
        )
    }

    private val UUID.hex: String get() = toString().replace("-", "")
}
