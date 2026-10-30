package com.photovance.app.ui.screens

import android.app.Activity
import android.util.Patterns
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowForward
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.photovance.app.R
import com.photovance.app.auth.AuthRepository
import com.photovance.app.ui.components.*

@Composable
fun AuthScreen(
    authRepository: AuthRepository? = null,
    onGuest: () -> Unit,
    onSettings: () -> Unit
) {
    var signUp by rememberSaveable { mutableStateOf(false) }
    BackHandler(signUp) { signUp = false }
    Column(Modifier.fillMaxSize()) {
        StudioHeader(onSettings, onExplore = onGuest)
        if (signUp) SignupScreen(authRepository, onGuest, onLogin = { signUp = false })
        else LoginScreen(authRepository, onGuest, onSignUp = { signUp = true })
    }
}

// Overload for backwards compatibility with tests
@Composable
fun AuthScreen(onGuest: () -> Unit, onSettings: () -> Unit) {
    AuthScreen(null, onGuest, onSettings)
}

@Composable
fun LoginScreen(authRepository: AuthRepository? = null, onGuest: () -> Unit, onSignUp: () -> Unit) =
    AuthenticationForm(false, authRepository, onGuest, onSignUp)

@Composable
fun SignupScreen(authRepository: AuthRepository? = null, onGuest: () -> Unit, onLogin: () -> Unit) =
    AuthenticationForm(true, authRepository, onGuest, onLogin)

@Composable
private fun AuthenticationForm(
    signUp: Boolean,
    authRepository: AuthRepository?,
    onGuest: () -> Unit,
    onSwitch: () -> Unit
) {
    val context = LocalContext.current
    val activity = context as? Activity

    var name by rememberSaveable { mutableStateOf("") }
    var email by rememberSaveable { mutableStateOf("") }
    var password by remember { mutableStateOf("") }
    var phone by rememberSaveable { mutableStateOf("") }
    var otpCode by rememberSaveable { mutableStateOf("") }
    var showPhoneAuth by rememberSaveable { mutableStateOf(false) }
    var showOtpDialog by rememberSaveable { mutableStateOf(false) }
    var verificationId by rememberSaveable { mutableStateOf("") }
    var passwordVisible by remember { mutableStateOf(false) }
    var attempted by remember { mutableStateOf(false) }
    var notice by remember { mutableStateOf<String?>(null) }
    var recovery by remember { mutableStateOf(false) }

    val validEmail = Patterns.EMAIL_ADDRESS.matcher(email.trim()).matches()
    val validPassword = if (signUp) password.length >= 8 else password.isNotBlank()
    val validPhone = phone.trim().replace(" ", "").length >= 8
    val fields = Modifier.fillMaxWidth()

    Column(
        Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .imePadding()
            .padding(horizontal = 24.dp)
            .padding(bottom = 20.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp)
    ) {
        AuthHero()
        Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text(if (signUp) "Make room for a new you." else "Welcome back.",
                style = MaterialTheme.typography.headlineLarge)
            Text(if (signUp) "A little inspiration. Endless possibilities." else "Your next look starts here.",
                style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        Text("SIGN-IN PREVIEW", style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.primary, letterSpacing = 1.6.sp)
        Column(verticalArrangement = Arrangement.spacedBy(9.dp)) {
            ProviderButton("Continue with Google", google = true) { notice = "Google sign-in" }
            ProviderButton("Continue with Apple", google = false) { notice = "Apple sign-in" }
        }
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(14.dp)) {
            HorizontalDivider(Modifier.weight(1f), color = MaterialTheme.colorScheme.outlineVariant)
            Text("or use email / phone", style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant)
            HorizontalDivider(Modifier.weight(1f), color = MaterialTheme.colorScheme.outlineVariant)
        }

        if (showPhoneAuth) {
            // Phone SMS OTP Section
            OutlinedTextField(
                value = phone,
                onValueChange = { phone = it },
                label = { Text("Mobile Phone Number") },
                placeholder = { Text("+1 555-123-4567 or +91 9876543210") },
                singleLine = true,
                modifier = fields,
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Phone),
                shape = MaterialTheme.shapes.small,
                leadingIcon = { Icon(Icons.Default.PhoneIphone, null) },
                isError = attempted && !validPhone,
                supportingText = if (attempted && !validPhone) ({ Text("Enter valid phone number with country code") }) else null
            )
            StudioButton("Send 6-Digit SMS Code", onClick = {
                attempted = true
                if (validPhone) {
                    verificationId = "verif_${System.currentTimeMillis()}"
                    showOtpDialog = true
                }
            })
            TextButton(onClick = { showPhoneAuth = false }, modifier = Modifier.align(Alignment.CenterHorizontally)) {
                Text("Switch back to email")
            }
        } else {
            // Email Section
            if (signUp) OutlinedTextField(value = name, onValueChange = { name = it }, label = { Text("Name") },
                singleLine = true, modifier = fields, shape = MaterialTheme.shapes.small,
                leadingIcon = { Icon(Icons.Default.PersonOutline, null) },
                isError = attempted && name.isBlank(),
                supportingText = if (attempted && name.isBlank()) ({ Text("Enter your name") }) else null)
            OutlinedTextField(value = email, onValueChange = { email = it }, label = { Text("Email") },
                placeholder = { Text("you@example.com") }, singleLine = true, modifier = fields,
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Email),
                shape = MaterialTheme.shapes.small, leadingIcon = { Icon(Icons.Default.MailOutline, null) },
                isError = attempted && !validEmail,
                supportingText = if (attempted && !validEmail) ({ Text("Enter a valid email address") }) else null)
            OutlinedTextField(value = password, onValueChange = { password = it }, label = { Text("Password") },
                singleLine = true, modifier = fields, shape = MaterialTheme.shapes.small,
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password),
                visualTransformation = if (passwordVisible) VisualTransformation.None else PasswordVisualTransformation(),
                leadingIcon = { Icon(Icons.Default.Lock, null) },
                trailingIcon = { IconButton(onClick = { passwordVisible = !passwordVisible }) {
                    Icon(if (passwordVisible) Icons.Default.VisibilityOff else Icons.Default.Visibility,
                        if (passwordVisible) "Hide password" else "Show password")
                } }, isError = attempted && !validPassword,
                supportingText = if (signUp || (attempted && !validPassword)) ({
                    Text(if (signUp) "At least 8 characters" else "Enter your password")
                }) else null)
            if (!signUp) TextButton(onClick = { recovery = true }, modifier = Modifier.align(Alignment.End)) {
                Text("Forgot password?")
            }
            StudioButton(if (signUp) "Create account" else "Log in", onClick = {
                attempted = true
                if (validEmail && validPassword && (!signUp || name.isNotBlank())) {
                    authRepository?.signInWithGoogle("token", email, name)
                    notice = if (signUp) "Create account" else "Email sign-in"
                }
            })
            if (!signUp) {
                StudioOutlineButton("Email me a sign-in link", onClick = {
                    attempted = true
                    if (validEmail) {
                        authRepository?.sendEmailMagicLink(email) { _, _ -> }
                        notice = "Email sign-in link"
                    }
                }, icon = Icons.Default.MarkEmailUnread)
                StudioOutlineButton("Sign in with Phone Number (SMS)", onClick = {
                    showPhoneAuth = true
                    attempted = false
                }, icon = Icons.Default.PhoneIphone)
            }
        }

        TextButton(onClick = onSwitch, modifier = Modifier.align(Alignment.CenterHorizontally)) {
            Text(if (signUp) "Already have an account? Log in" else "New here? Create an account")
        }
        HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
        StudioOutlineButton("Explore as a guest", onClick = {
            authRepository?.signInAsGuest()
            onGuest()
        }, icon = Icons.AutoMirrored.Filled.ArrowForward)
        Text("Account features are preview only. Guest access opens the photo studio.",
            style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }

    if (notice != null) AlertDialog(onDismissRequest = { notice = null },
        title = { Text("${notice} is coming soon") },
        text = { Text("This design preview doesn't create accounts or send sign-in emails. You can explore the studio as a guest.") },
        confirmButton = { TextButton(onClick = { notice = null; onGuest() }) { Text("Explore as a guest") } },
        dismissButton = { TextButton(onClick = { notice = null }) { Text("Back") } })

    if (recovery) AlertDialog(onDismissRequest = { recovery = false },
        title = { Text("Reset your password") },
        text = { Text("Password reset will be available when account sign-in launches. This preview hasn't created an account or saved your password.") },
        confirmButton = { TextButton(onClick = { recovery = false; onGuest() }) { Text("Explore as a guest") } },
        dismissButton = { TextButton(onClick = { recovery = false }) { Text("Back") } })

    if (showOtpDialog) {
        AlertDialog(
            onDismissRequest = { showOtpDialog = false },
            title = { Text("Enter 6-Digit SMS Code") },
            text = {
                Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    Text("We sent an SMS code to $phone. Enter it below to sign in:")
                    OutlinedTextField(
                        value = otpCode,
                        onValueChange = { if (it.length <= 6) otpCode = it },
                        label = { Text("6-Digit Code") },
                        placeholder = { Text("123456") },
                        singleLine = true,
                        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                        modifier = Modifier.fillMaxWidth()
                    )
                }
            },
            confirmButton = {
                Button(onClick = {
                    if (otpCode.length == 6) {
                        authRepository?.verifyPhoneOtp(verificationId, otpCode) { _, _ -> }
                        showOtpDialog = false
                        onGuest()
                    }
                }) {
                    Text("Verify & Sign In")
                }
            },
            dismissButton = {
                TextButton(onClick = { showOtpDialog = false }) { Text("Cancel") }
            }
        )
    }
}

@Composable
private fun ProviderButton(label: String, google: Boolean, onClick: () -> Unit) {
    OutlinedButton(onClick, modifier = Modifier.fillMaxWidth().heightIn(min = 52.dp),
        shape = MaterialTheme.shapes.small, border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant)) {
        if (google) Image(painterResource(R.drawable.ic_google), null, Modifier.size(22.dp))
        else Icon(painterResource(R.drawable.ic_apple), null, Modifier.size(24.dp), tint = MaterialTheme.colorScheme.onSurface)
        Spacer(Modifier.width(12.dp))
        Text(label, color = MaterialTheme.colorScheme.onSurface)
    }
}

@Composable
private fun AuthHero() {
    Box(Modifier.fillMaxWidth().height(205.dp).clip(MaterialTheme.shapes.large)
        .background(Color(0xFF494237))) {
        Image(painterResource(R.drawable.cc0_studio), null, contentScale = ContentScale.Crop,
            alignment = Alignment.TopCenter, modifier = Modifier.fillMaxSize())
        Box(Modifier.fillMaxSize().background(Brush.horizontalGradient(
            listOf(Color(0xE81C1A19), Color(0x501C1A19), Color.Transparent))))
        Column(Modifier.align(Alignment.BottomStart).padding(22.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Text("A new look.\nStill you.", style = MaterialTheme.typography.displaySmall, color = Color.White)
            Text("YOUR EVERYDAY, REIMAGINED", style = MaterialTheme.typography.labelSmall,
                color = Color(0xFFE5DBCC), letterSpacing = 1.4.sp)
        }
    }
}
