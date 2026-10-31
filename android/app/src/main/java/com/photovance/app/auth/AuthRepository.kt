package com.photovance.app.auth

import android.app.Activity
import android.content.Context
import androidx.credentials.CredentialManager
import androidx.credentials.GetCredentialRequest
import com.google.android.libraries.identity.googleid.GetGoogleIdOption
import com.google.android.libraries.identity.googleid.GoogleIdTokenCredential
import com.google.firebase.FirebaseApp
import com.google.firebase.auth.FirebaseAuth
import com.google.firebase.auth.FirebaseUser
import com.google.firebase.auth.GoogleAuthProvider
import com.google.firebase.auth.PhoneAuthOptions
import com.google.firebase.auth.PhoneAuthProvider
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.tasks.await
import java.util.concurrent.TimeUnit

interface AuthRepository {
    val authState: StateFlow<AuthState>
    val currentUser: StateFlow<AuthUser?>

    suspend fun signInWithGoogle(activity: Activity): AuthUser
    fun signInWithEmail(email: String, password: String, onResult: (Boolean, String?) -> Unit)
    fun createEmailAccount(name: String, email: String, password: String, onResult: (Boolean, String?) -> Unit)
    fun sendPasswordReset(email: String, onResult: (Boolean, String?) -> Unit)
    fun sendPhoneOtp(phoneNumber: String, activity: Activity, onResult: (Boolean, String?) -> Unit)
    fun verifyPhoneOtp(verificationId: String, code: String, onResult: (Boolean, String?) -> Unit)
    fun signInAsGuest(onResult: (Boolean, String?) -> Unit)
    fun signOut()
    fun deleteAccount(onResult: (Boolean, String?) -> Unit)
    fun close()
}

class PhotovanceAuthRepository(context: Context) : AuthRepository {
    private val app = FirebaseApp.initializeApp(context.applicationContext)
    private val auth = app?.let { FirebaseAuth.getInstance(it) }

    private val _authState = MutableStateFlow<AuthState>(
        if (auth == null) AuthState.Error("Firebase is not configured. Add android/app/google-services.json to enable sign-in.")
        else AuthState.Idle
    )
    override val authState: StateFlow<AuthState> = _authState.asStateFlow()

    private val _currentUser = MutableStateFlow(auth?.currentUser?.toAuthUser())
    override val currentUser: StateFlow<AuthUser?> = _currentUser.asStateFlow()

    private val authListener = FirebaseAuth.AuthStateListener { firebaseAuth ->
        val user = firebaseAuth.currentUser?.toAuthUser()
        _currentUser.value = user
        _authState.value = if (user == null) AuthState.Idle else AuthState.Authenticated(user)
    }

    init {
        auth?.addAuthStateListener(authListener)
    }

    override suspend fun signInWithGoogle(activity: Activity): AuthUser {
        val firebaseAuth = requireAuth()
        _authState.value = AuthState.Loading
        return try {
            val resourceId = activity.resources.getIdentifier(
                "default_web_client_id",
                "string",
                activity.packageName
            )
            if (resourceId == 0) {
                throw IllegalStateException("Firebase Google sign-in is not configured for this app.")
            }
            val serverClientId = activity.getString(resourceId)
            val googleIdOption = GetGoogleIdOption.Builder()
                .setServerClientId(serverClientId)
                .setFilterByAuthorizedAccounts(false)
                .setAutoSelectEnabled(false)
                .build()
            val request = GetCredentialRequest.Builder()
                .addCredentialOption(googleIdOption)
                .build()
            val credential = CredentialManager.create(activity).getCredential(activity, request).credential
            if (credential !is androidx.credentials.CustomCredential ||
                credential.type != GoogleIdTokenCredential.TYPE_GOOGLE_ID_TOKEN_CREDENTIAL
            ) {
                throw IllegalStateException("Google sign-in returned an unsupported credential.")
            }
            val googleToken = GoogleIdTokenCredential.createFrom(credential.data).idToken
            val firebaseCredential = GoogleAuthProvider.getCredential(googleToken, null)
            val current = firebaseAuth.currentUser
            val result = if (current?.isAnonymous == true) {
                current.linkWithCredential(firebaseCredential).await()
            } else {
                firebaseAuth.signInWithCredential(firebaseCredential).await()
            }
            result.user?.toAuthUser() ?: throw IllegalStateException("Google sign-in did not return a user.")
        } catch (error: Exception) {
            _authState.value = AuthState.Error(error.message ?: "Google sign-in failed.")
            throw error
        }
    }

    override fun signInWithEmail(email: String, password: String, onResult: (Boolean, String?) -> Unit) {
        val firebaseAuth = authOrReport(onResult) ?: return
        _authState.value = AuthState.Loading
        firebaseAuth.signInWithEmailAndPassword(email.trim(), password).addOnCompleteListener { task ->
            if (task.isSuccessful) {
                _authState.value = task.result.user?.let { AuthState.Authenticated(it.toAuthUser()) } ?: AuthState.Idle
                onResult(true, null)
            } else {
                fail(task.exception?.message ?: "Email sign-in failed.", onResult)
            }
        }
    }

    override fun createEmailAccount(
        name: String,
        email: String,
        password: String,
        onResult: (Boolean, String?) -> Unit
    ) {
        val firebaseAuth = authOrReport(onResult) ?: return
        _authState.value = AuthState.Loading
        val existingUser = firebaseAuth.currentUser
        val task = if (existingUser?.isAnonymous == true) {
            existingUser.linkWithCredential(
                com.google.firebase.auth.EmailAuthProvider.getCredential(email.trim(), password)
            )
        } else {
            firebaseAuth.createUserWithEmailAndPassword(email.trim(), password)
        }
        task.addOnCompleteListener { result ->
            if (!result.isSuccessful) {
                fail(result.exception?.message ?: "Could not create the account.", onResult)
                return@addOnCompleteListener
            }
            val user = result.result.user
            if (user == null) {
                fail("Account creation completed without a user.", onResult)
                return@addOnCompleteListener
            }
            user.updateProfile(com.google.firebase.auth.UserProfileChangeRequest.Builder()
                .setDisplayName(name.trim())
                .build())
                .addOnCompleteListener { profileTask ->
                    if (!profileTask.isSuccessful) {
                        _authState.value = AuthState.Authenticated(user.toAuthUser())
                        onResult(true, "Account created, but the display name could not be saved.")
                        return@addOnCompleteListener
                    }
                    user.sendEmailVerification().addOnCompleteListener { verificationTask ->
                        _authState.value = AuthState.Authenticated(user.toAuthUser())
                        val message = if (verificationTask.isSuccessful) {
                            "Account created. Check your email to verify it."
                        } else {
                            "Account created, but the verification email could not be sent."
                        }
                        onResult(true, message)
                    }
                }
        }
    }

    override fun sendPasswordReset(email: String, onResult: (Boolean, String?) -> Unit) {
        val firebaseAuth = authOrReport(onResult) ?: return
        firebaseAuth.sendPasswordResetEmail(email.trim()).addOnCompleteListener { task ->
            if (task.isSuccessful) onResult(true, "Password reset email sent.")
            else onResult(false, task.exception?.message ?: "Could not send the password reset email.")
        }
    }

    override fun sendPhoneOtp(phoneNumber: String, activity: Activity, onResult: (Boolean, String?) -> Unit) {
        val firebaseAuth = authOrReport(onResult) ?: return
        val cleanNumber = phoneNumber.trim().filter { it == '+' || it.isDigit() }
        if (!Regex("^\\+[1-9]\\d{7,14}$").matches(cleanNumber)) {
            onResult(false, "Enter a valid phone number in international format, including + and country code.")
            return
        }
        _authState.value = AuthState.Loading
        val callbacks = object : PhoneAuthProvider.OnVerificationStateChangedCallbacks() {
            override fun onVerificationCompleted(credential: com.google.firebase.auth.PhoneAuthCredential) {
                val current = firebaseAuth.currentUser
                val task = if (current?.isAnonymous == true) current.linkWithCredential(credential)
                    else firebaseAuth.signInWithCredential(credential)
                task.addOnCompleteListener { result ->
                    if (result.isSuccessful) {
                        result.result.user?.let { _authState.value = AuthState.Authenticated(it.toAuthUser()) }
                        onResult(true, "Phone number verified.")
                    } else fail(result.exception?.message ?: "Phone verification failed.", onResult)
                }
            }

            override fun onVerificationFailed(error: com.google.firebase.FirebaseException) {
                fail(error.message ?: "Could not send the verification code.", onResult)
            }

            override fun onCodeSent(verificationId: String, token: PhoneAuthProvider.ForceResendingToken) {
                _authState.value = AuthState.CodeSent(verificationId, cleanNumber)
                onResult(true, verificationId)
            }
        }
        PhoneAuthProvider.verifyPhoneNumber(
            PhoneAuthOptions.newBuilder(firebaseAuth)
                .setPhoneNumber(cleanNumber)
                .setTimeout(60, TimeUnit.SECONDS)
                .setActivity(activity)
                .setCallbacks(callbacks)
                .build()
        )
    }

    override fun verifyPhoneOtp(verificationId: String, code: String, onResult: (Boolean, String?) -> Unit) {
        val firebaseAuth = authOrReport(onResult) ?: return
        if (code.length != 6 || !code.all(Char::isDigit)) {
            onResult(false, "Enter the 6-digit code from the SMS.")
            return
        }
        val credential = PhoneAuthProvider.getCredential(verificationId, code)
        val existingUser = firebaseAuth.currentUser
        val task = if (existingUser?.isAnonymous == true) existingUser.linkWithCredential(credential)
            else firebaseAuth.signInWithCredential(credential)
        _authState.value = AuthState.Loading
        task.addOnCompleteListener { result ->
            if (result.isSuccessful) {
                result.result.user?.let { _authState.value = AuthState.Authenticated(it.toAuthUser()) }
                onResult(true, "Phone number verified.")
            } else {
                fail(result.exception?.message ?: "Phone verification failed.", onResult)
            }
        }
    }

    override fun signInAsGuest(onResult: (Boolean, String?) -> Unit) {
        val firebaseAuth = authOrReport(onResult) ?: return
        if (firebaseAuth.currentUser != null) {
            onResult(true, null)
            return
        }
        _authState.value = AuthState.Loading
        firebaseAuth.signInAnonymously().addOnCompleteListener { task ->
            if (task.isSuccessful) {
                task.result.user?.let { _authState.value = AuthState.Authenticated(it.toAuthUser()) }
                onResult(true, null)
            } else {
                fail(task.exception?.message ?: "Guest sign-in failed.", onResult)
            }
        }
    }

    override fun signOut() {
        auth?.signOut()
        _currentUser.value = null
        _authState.value = AuthState.Idle
    }

    override fun deleteAccount(onResult: (Boolean, String?) -> Unit) {
        val firebaseAuth = authOrReport(onResult) ?: return
        val user = firebaseAuth.currentUser
        if (user == null) {
            onResult(false, "There is no signed-in account to delete.")
            return
        }
        _authState.value = AuthState.Loading
        com.google.firebase.functions.FirebaseFunctions.getInstance()
            .getHttpsCallable("deleteAccountData")
            .call()
            .addOnCompleteListener { cleanupTask ->
                if (!cleanupTask.isSuccessful) {
                    fail(cleanupTask.exception?.message ?: "Could not delete the account's data.", onResult)
                    return@addOnCompleteListener
                }
                firebaseAuth.signOut()
                _currentUser.value = null
                _authState.value = AuthState.Idle
                onResult(true, "Your Photovance account was deleted.")
            }
    }

    override fun close() {
        auth?.removeAuthStateListener(authListener)
    }

    private fun requireAuth(): FirebaseAuth =
        auth ?: throw IllegalStateException("Firebase is not configured. Add android/app/google-services.json.")

    private fun authOrReport(onResult: (Boolean, String?) -> Unit): FirebaseAuth? {
        if (auth != null) return auth
        val message = "Firebase is not configured. Add android/app/google-services.json."
        _authState.value = AuthState.Error(message)
        onResult(false, message)
        return null
    }

    private fun fail(message: String, onResult: (Boolean, String?) -> Unit) {
        _authState.value = AuthState.Error(message)
        onResult(false, message)
    }

    private fun FirebaseUser.toAuthUser() = AuthUser(
        uid = uid,
        displayName = displayName,
        email = email,
        phoneNumber = phoneNumber,
        photoUrl = photoUrl?.toString(),
        isGuest = isAnonymous
    )
}
