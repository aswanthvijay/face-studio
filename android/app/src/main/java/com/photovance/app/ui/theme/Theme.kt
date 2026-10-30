package com.photovance.app.ui.theme

import android.app.Activity
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Shapes
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.SideEffect
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.view.WindowCompat

enum class AppAppearance(val title: String) { System("System"), Light("Light"), Dark("Dark") }

private val Obsidian = darkColorScheme(
    primary = WebUiPrimary, onPrimary = Color.White,
    primaryContainer = Color(0xFF1E293B), onPrimaryContainer = Color(0xFFE0F2FE),
    secondary = WebUiAccent, onSecondary = Color.White,
    secondaryContainer = Color(0xFF1E1B4B), onSecondaryContainer = Color(0xFFEDE9FE),
    tertiary = WebUiGold, onTertiary = Color.Black,
    background = WebUiBgDark, onBackground = WebUiTextMain,
    surface = WebUiBgCard, onSurface = WebUiTextMain,
    surfaceVariant = Color(0xFF162035), onSurfaceVariant = WebUiTextMuted,
    outline = WebUiBorderHover, outlineVariant = WebUiBorderCard,
    error = Color(0xFFFFB4AB), onError = Color(0xFF690005),
    errorContainer = Color(0xFF442426), onErrorContainer = Color(0xFFFFDAD6)
)

private val Ivory = lightColorScheme(
    primary = Color(0xFF0284C7), onPrimary = Color.White,
    primaryContainer = Color(0xFFE0F2FE), onPrimaryContainer = Color(0xFF0C4A6E),
    secondary = Color(0xFF7C3AED), onSecondary = Color.White,
    secondaryContainer = Color(0xFFEDE9FE), onSecondaryContainer = Color(0xFF4C1D95),
    tertiary = Color(0xFFD97706), onTertiary = Color.White,
    background = Color(0xFFF8FAFC), onBackground = Color(0xFF0F172A),
    surface = Color.White, onSurface = Color(0xFF0F172A),
    surfaceVariant = Color(0xFFF1F5F9), onSurfaceVariant = Color(0xFF64748B),
    outline = Color(0xFF94A3B8), outlineVariant = Color(0xFFE2E8F0),
    error = Color(0xFF9C3932), onError = Color.White,
    errorContainer = Color(0xFFFFE9E5), onErrorContainer = Color(0xFF6E211C)
)

@Composable
fun PhotovanceTheme(
    appearance: AppAppearance = AppAppearance.System,
    content: @Composable () -> Unit
) {
    val dark = when (appearance) {
        AppAppearance.System -> isSystemInDarkTheme()
        AppAppearance.Light -> false
        AppAppearance.Dark -> true
    }
    val view = LocalView.current
    if (!view.isInEditMode) {
        SideEffect {
            (view.context as? Activity)?.window?.let { window ->
                WindowCompat.getInsetsController(window, view).apply {
                    isAppearanceLightStatusBars = !dark
                    isAppearanceLightNavigationBars = !dark
                }
            }
        }
    }
    val editorial = FontFamily.SansSerif
    val base = Typography()
    MaterialTheme(
        colorScheme = if (dark) Obsidian else Ivory,
        typography = base.copy(
            displaySmall = base.displaySmall.copy(fontFamily = editorial, fontSize = 36.sp,
                lineHeight = 42.sp, fontWeight = FontWeight.Bold, letterSpacing = (-1).sp),
            headlineLarge = base.headlineLarge.copy(fontFamily = editorial, fontSize = 30.sp,
                lineHeight = 36.sp, fontWeight = FontWeight.Bold, letterSpacing = (-0.8).sp),
            headlineMedium = base.headlineMedium.copy(fontFamily = editorial, fontSize = 24.sp,
                lineHeight = 30.sp, fontWeight = FontWeight.Bold),
            titleLarge = base.titleLarge.copy(fontFamily = editorial, fontSize = 20.sp, fontWeight = FontWeight.SemiBold),
            titleMedium = base.titleMedium.copy(fontFamily = editorial, fontSize = 16.sp, fontWeight = FontWeight.SemiBold),
            bodyLarge = base.bodyLarge.copy(fontSize = 15.sp, lineHeight = 22.sp),
            bodyMedium = base.bodyMedium.copy(fontSize = 13.sp, lineHeight = 19.sp),
            labelLarge = base.labelLarge.copy(fontWeight = FontWeight.SemiBold, fontSize = 13.sp),
            labelSmall = base.labelSmall.copy(fontWeight = FontWeight.Medium, fontSize = 11.sp, letterSpacing = 0.5.sp)
        ),
        shapes = Shapes(
            small = RoundedCornerShape(8.dp),
            medium = RoundedCornerShape(14.dp),
            large = RoundedCornerShape(20.dp),
            extraLarge = RoundedCornerShape(28.dp)
        ),
        content = content
    )
}
