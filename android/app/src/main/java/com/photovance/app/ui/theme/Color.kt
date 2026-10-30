package com.photovance.app.ui.theme

import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color

// WebUI 7870 Obsidian Glassmorphism Palette
val WebUiBgDark = Color(0xFF090C10)
val WebUiBgCard = Color(0xBF121826)
val WebUiBgCardHover = Color(0xD9162035)
val WebUiSurfaceGlass = Color(0x800F172A)
val WebUiBorderCard = Color(0x1FFFFFFF)
val WebUiBorderHover = Color(0x5938BDF8)

// Neon & Luxury Brand Accents
val WebUiPrimary = Color(0xFF0EA5E9)         // Electric Cyan
val WebUiPrimaryGlow = Color(0x660EA5E9)
val WebUiAccent = Color(0xFF8B5CF6)          // Vivid Violet
val WebUiPink = Color(0xFFEC4899)
val WebUiGold = Color(0xFFF59E0B)            // Haute Couture Gold
val WebUiStatusGreen = Color(0xFF10B981)     // GPU Active Green

// Typography & Neutrals
val WebUiTextMain = Color(0xFFF1F5F9)
val WebUiTextMuted = Color(0xFF94A3B8)
val WebUiTextDim = Color(0xFF64748B)

// Backward Compatible Tokens
val BgMain = WebUiBgDark
val BgCanvas = WebUiBgDark
val BgCard = Color(0xFF0F1626)
val BgCardHover = Color(0xFF162035)
val GlassSurface = Color(0xCC111827)
val GlassBorder = Color(0x2EFFFFFF)
val Accent = WebUiPrimary
val AccentPink = WebUiPink
val NeonMagenta = WebUiPink
val NeonViolet = WebUiAccent
val NeonCyan = WebUiPrimary
val GoldAmber = WebUiGold
val Success = WebUiStatusGreen
val TextMain = WebUiTextMain
val TextMuted = WebUiTextMuted
val BorderColor = WebUiBorderCard

// Luxury Gradients & Glows
val WebUiGradient = Brush.horizontalGradient(listOf(WebUiPrimary, Color(0xFF6366F1), WebUiAccent))
val WebUiTabActiveGradient = Brush.linearGradient(listOf(Color(0x330EA5E9), Color(0x338B5CF6)))
val WebUiGlassBorderGradient = Brush.verticalGradient(listOf(Color(0x38FFFFFF), Color(0x0DFFFFFF)))
val HeroGradient = WebUiGradient
val NeonGlowGradient = Brush.linearGradient(listOf(WebUiAccent, WebUiPink, WebUiPrimary))
val GoldGradient = Brush.linearGradient(listOf(Color(0xFFFBBF24), WebUiGold))
