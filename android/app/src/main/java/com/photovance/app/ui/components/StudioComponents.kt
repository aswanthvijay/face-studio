package com.photovance.app.ui.components

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.ErrorOutline
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.photovance.app.R
import com.photovance.app.ui.theme.*

@Composable
fun FoldedLightMark(modifier: Modifier = Modifier) {
    Image(painterResource(R.drawable.ic_folded_light), contentDescription = null, modifier = modifier)
}

/**
 * Atmospheric soft radial glow backdrop recreation from WebUI .glow-bg
 */
@Composable
fun AtmosphericGlowBackground(modifier: Modifier = Modifier) {
    Box(modifier = modifier.fillMaxSize()) {
        // Cyan glow (top-left)
        Box(
            modifier = Modifier
                .size(340.dp)
                .offset(x = (-100).dp, y = (-100).dp)
                .clip(CircleShape)
                .background(
                    Brush.radialGradient(
                        listOf(Color(0x2E0EA5E9), Color.Transparent)
                    )
                )
        )
        // Violet glow (bottom-right)
        Box(
            modifier = Modifier
                .size(380.dp)
                .align(Alignment.BottomEnd)
                .offset(x = 100.dp, y = 100.dp)
                .clip(CircleShape)
                .background(
                    Brush.radialGradient(
                        listOf(Color(0x288B5CF6), Color.Transparent)
                    )
                )
        )
    }
}

/**
 * WebUI Glassmorphism Card Style
 */
@Composable
fun WebUiCard(
    modifier: Modifier = Modifier,
    content: @Composable ColumnScope.() -> Unit
) {
    Surface(
        modifier = modifier.fillMaxWidth(),
        shape = MaterialTheme.shapes.large,
        color = WebUiBgCard,
        border = BorderStroke(1.dp, WebUiGlassBorderGradient)
    ) {
        Column(
            modifier = Modifier.padding(18.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp),
            content = content
        )
    }
}

@Composable
fun StudioHeader(
    onSettings: () -> Unit,
    onBack: (() -> Unit)? = null,
    onExplore: (() -> Unit)? = null,
    engineModeTitle: String? = null,
    onToggleEngine: (() -> Unit)? = null,
    vipBadgeTitle: String? = null,
    onTapVip: (() -> Unit)? = null
) {
    Row(
        Modifier
            .fillMaxWidth()
            .heightIn(min = 72.dp)
            .padding(horizontal = 20.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        if (onBack != null) {
            IconButton(onClick = onBack, modifier = Modifier.padding(end = 4.dp)) {
                Icon(Icons.AutoMirrored.Filled.ArrowBack, "Back", tint = WebUiTextMain)
            }
        } else {
            // WebUI Gradient Brand Logo Box
            Box(
                modifier = Modifier
                    .size(38.dp)
                    .clip(RoundedCornerShape(10.dp))
                    .background(WebUiGradient),
                contentAlignment = Alignment.Center
            ) {
                Icon(Icons.Default.AutoAwesome, contentDescription = null, tint = Color.White, modifier = Modifier.size(20.dp))
            }
            Spacer(Modifier.width(12.dp))
        }

        Column(Modifier.weight(1f)) {
            Text(
                "Face Studio AI",
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.Bold,
                color = WebUiTextMain,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
            Text(
                "Photovance Engine · Port 7870",
                style = MaterialTheme.typography.labelSmall,
                color = WebUiTextMuted
            )
        }

        // WebUI Live GPU Engine Status Pill
        Surface(
            shape = CircleShape,
            color = Color(0x331E293B),
            border = BorderStroke(1.dp, Color(0x22FFFFFF))
        ) {
            Row(
                modifier = Modifier.padding(horizontal = 10.dp, vertical = 5.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                Box(
                    modifier = Modifier
                        .size(7.dp)
                        .clip(CircleShape)
                        .background(WebUiStatusGreen)
                )
                Text(
                    "Active",
                    style = MaterialTheme.typography.labelSmall,
                    fontWeight = FontWeight.Bold,
                    color = Color.White
                )
            }
        }

        Spacer(Modifier.width(6.dp))

        if (vipBadgeTitle != null) {
            Surface(
                onClick = { onTapVip?.invoke() },
                shape = MaterialTheme.shapes.small,
                color = WebUiGold.copy(alpha = 0.15f),
                border = BorderStroke(1.dp, WebUiGold.copy(alpha = 0.6f))
            ) {
                Text(
                    text = vipBadgeTitle,
                    style = MaterialTheme.typography.labelSmall,
                    color = WebUiGold,
                    modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
                )
            }
            Spacer(Modifier.width(6.dp))
        }

        if (onExplore != null) TextButton(onClick = onExplore) { Text("Explore") }
        IconButton(onClick = onSettings) {
            Icon(Icons.Default.Settings, "Settings and appearance", tint = WebUiTextMuted)
        }
    }
}

@Composable
fun StudioSteps(current: Int, hasPhoto: Boolean, hasResult: Boolean, enabled: Boolean, onStep: (Int) -> Unit) {
    Row(
        Modifier
            .fillMaxWidth()
            .padding(horizontal = 24.dp, vertical = 6.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        listOf("Photo", "Style", "Result").forEachIndexed { index, title ->
            if (index > 0) {
                HorizontalDivider(
                    Modifier
                        .weight(1f)
                        .padding(horizontal = 8.dp),
                    color = if (current >= index) WebUiPrimary else Color(0x1FFFFFFF)
                )
            }
            val available = enabled && (index == 0 || (index == 1 && hasPhoto) || (index == 2 && hasResult))
            Column(
                Modifier
                    .widthIn(min = 48.dp)
                    .clip(MaterialTheme.shapes.small)
                    .selectable(
                        selected = current == index,
                        enabled = available,
                        role = Role.Tab,
                        onClick = { onStep(index) }
                    )
                    .padding(vertical = 4.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(4.dp)
            ) {
                Box(
                    Modifier
                        .size(22.dp)
                        .border(1.dp, if (index <= current) WebUiPrimary else Color(0x33FFFFFF), CircleShape)
                        .padding(3.dp)
                        .background(
                            if (index <= current) WebUiPrimary else Color.Transparent,
                            CircleShape
                        ),
                    contentAlignment = Alignment.Center
                ) {
                    if (index < current) Icon(
                        Icons.Default.Check, null,
                        tint = Color.White, modifier = Modifier.size(12.dp)
                    )
                }
                Text(
                    title,
                    style = MaterialTheme.typography.labelSmall,
                    fontWeight = if (index == current) FontWeight.Bold else FontWeight.Normal,
                    color = if (index == current) WebUiPrimary else WebUiTextMuted
                )
            }
        }
    }
}

@Composable
fun PageHeading(eyebrow: String, title: String, description: String) {
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Text(
            eyebrow.uppercase(),
            style = MaterialTheme.typography.labelSmall,
            fontWeight = FontWeight.Bold,
            letterSpacing = 1.5.sp,
            color = WebUiPrimary
        )
        Text(
            title,
            style = MaterialTheme.typography.headlineLarge,
            color = WebUiTextMain
        )
        Text(
            description,
            style = MaterialTheme.typography.bodyMedium,
            color = WebUiTextMuted
        )
    }
}

/**
 * WebUI Glowing Neon Gradient Button (.generate-btn)
 */
@Composable
fun StudioButton(
    text: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    icon: ImageVector? = null,
    enabled: Boolean = true,
    loading: Boolean = false
) {
    Box(
        modifier = modifier
            .fillMaxWidth()
            .heightIn(min = 54.dp)
            .clip(MaterialTheme.shapes.medium)
            .background(
                if (enabled) WebUiGradient else Brush.linearGradient(listOf(Color(0xFF334155), Color(0xFF1E293B)))
            )
            .clickable(enabled = enabled && !loading, onClick = onClick),
        contentAlignment = Alignment.Center
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 20.dp, vertical = 14.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.Center
        ) {
            if (loading) {
                CircularProgressIndicator(Modifier.size(20.dp), strokeWidth = 2.dp, color = Color.White)
                Spacer(Modifier.width(10.dp))
            } else if (icon != null) {
                Icon(icon, null, Modifier.size(20.dp), tint = Color.White)
                Spacer(Modifier.width(10.dp))
            }
            Text(
                text,
                style = MaterialTheme.typography.labelLarge,
                fontWeight = FontWeight.Bold,
                color = if (enabled) Color.White else Color(0xFF94A3B8)
            )
        }
    }
}

@Composable
fun StudioOutlineButton(
    text: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    icon: ImageVector? = null,
    enabled: Boolean = true
) {
    Surface(
        onClick = onClick,
        enabled = enabled,
        modifier = modifier
            .fillMaxWidth()
            .heightIn(min = 50.dp),
        shape = MaterialTheme.shapes.medium,
        color = Color(0x331E293B),
        border = BorderStroke(1.dp, WebUiGlassBorderGradient)
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 16.dp, vertical = 12.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.Center
        ) {
            if (icon != null) {
                Icon(icon, null, Modifier.size(18.dp), tint = if (enabled) WebUiTextMain else WebUiTextDim)
                Spacer(Modifier.width(8.dp))
            }
            Text(
                text,
                style = MaterialTheme.typography.labelLarge,
                fontWeight = FontWeight.SemiBold,
                color = if (enabled) WebUiTextMain else WebUiTextDim
            )
        }
    }
}

/**
 * WebUI 4-Metric Studio Canvas Information Strip (.generation-info)
 */
@Composable
fun GenerationInfoBar(
    modelName: String,
    modeName: String,
    elapsed: Double?,
    modifier: Modifier = Modifier
) {
    Surface(
        modifier = modifier.fillMaxWidth(),
        shape = MaterialTheme.shapes.medium,
        color = Color(0xB30F172A),
        border = BorderStroke(1.dp, WebUiGlassBorderGradient)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 12.dp, vertical = 10.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(horizontalAlignment = Alignment.CenterHorizontally, modifier = Modifier.weight(1f)) {
                Text("MODEL", style = MaterialTheme.typography.labelSmall, color = WebUiTextDim, fontSize = 9.sp)
                Text(modelName.take(12), style = MaterialTheme.typography.labelSmall, fontWeight = FontWeight.Bold, color = WebUiTextMain, maxLines = 1)
            }
            Box(Modifier.width(1.dp).height(24.dp).background(Color(0x22FFFFFF)))
            Column(horizontalAlignment = Alignment.CenterHorizontally, modifier = Modifier.weight(1f)) {
                Text("MODE", style = MaterialTheme.typography.labelSmall, color = WebUiTextDim, fontSize = 9.sp)
                Text(modeName, style = MaterialTheme.typography.labelSmall, fontWeight = FontWeight.Bold, color = WebUiTextMain, maxLines = 1)
            }
            Box(Modifier.width(1.dp).height(24.dp).background(Color(0x22FFFFFF)))
            Column(horizontalAlignment = Alignment.CenterHorizontally, modifier = Modifier.weight(1f)) {
                Text("RENDER", style = MaterialTheme.typography.labelSmall, color = WebUiTextDim, fontSize = 9.sp)
                Text(if (elapsed != null) "${String.format(java.util.Locale.US, "%.1f", elapsed)}s" else "Fast", style = MaterialTheme.typography.labelSmall, fontWeight = FontWeight.Bold, color = WebUiTextMain)
            }
            Box(Modifier.width(1.dp).height(24.dp).background(Color(0x22FFFFFF)))
            Column(horizontalAlignment = Alignment.CenterHorizontally, modifier = Modifier.weight(1.2f)) {
                Text("FACE LOCK", style = MaterialTheme.typography.labelSmall, color = WebUiTextDim, fontSize = 9.sp)
                Text("100% Locked", style = MaterialTheme.typography.labelSmall, fontWeight = FontWeight.Bold, color = WebUiPrimary)
            }
        }
    }
}

@Composable
fun ErrorNotice(message: String, onSettings: () -> Unit) {
    Surface(
        color = MaterialTheme.colorScheme.errorContainer.copy(alpha = 0.9f),
        shape = MaterialTheme.shapes.medium,
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.error.copy(alpha = 0.5f))
    ) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                Icon(
                    Icons.Default.ErrorOutline, null,
                    tint = MaterialTheme.colorScheme.onErrorContainer,
                    modifier = Modifier.size(20.dp)
                )
                Text(
                    message,
                    color = MaterialTheme.colorScheme.onErrorContainer,
                    style = MaterialTheme.typography.bodyMedium,
                    modifier = Modifier.weight(1f)
                )
            }
            TextButton(onClick = onSettings) {
                Text("Connection settings", color = MaterialTheme.colorScheme.onErrorContainer, fontWeight = FontWeight.Bold)
            }
        }
    }
}
