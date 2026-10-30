package com.photovance.app.ui.screens

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Stars
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.photovance.app.billing.VipPlan
import com.photovance.app.ui.components.StudioButton

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PaywallSheet(
    onDismiss: () -> Unit,
    onSubscribe: (VipPlan) -> Unit,
    onRestore: () -> Unit
) {
    var selectedPlan by remember { mutableStateOf(VipPlan.YEARLY) }
    val sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)

    ModalBottomSheet(
        onDismissRequest = onDismiss,
        sheetState = sheetState,
        containerColor = Color(0xFF0F0E13),
        contentColor = Color.White,
        dragHandle = null
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 24.dp)
                .padding(top = 16.dp, bottom = 32.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            // Header with Close
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.End
            ) {
                IconButton(
                    onClick = onDismiss,
                    modifier = Modifier
                        .size(36.dp)
                        .background(Color.White.copy(alpha = 0.08f), CircleShape)
                ) {
                    Icon(
                        Icons.Default.Close,
                        contentDescription = "Close paywall",
                        tint = Color.White.copy(alpha = 0.8f),
                        modifier = Modifier.size(18.dp)
                    )
                }
            }

            Spacer(Modifier.height(8.dp))

            // Crown / Sparkle Emblem
            Box(
                modifier = Modifier
                    .size(64.dp)
                    .background(
                        Brush.radialGradient(
                            colors = listOf(Color(0xFFFFB800), Color(0xFF9333EA))
                        ),
                        CircleShape
                    )
                    .border(2.dp, Color(0xFFFFD700).copy(alpha = 0.6f), CircleShape),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    Icons.Default.Stars,
                    contentDescription = null,
                    modifier = Modifier.size(34.dp),
                    tint = Color.White
                )
            }

            Spacer(Modifier.height(16.dp))

            Text(
                text = "PHOTOVANCE VIP",
                style = MaterialTheme.typography.labelMedium.copy(
                    letterSpacing = 3.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color(0xFFFFB800)
                )
            )

            Spacer(Modifier.height(4.dp))

            Text(
                text = "Unlimited 8K Studio Power",
                style = MaterialTheme.typography.headlineSmall.copy(
                    fontWeight = FontWeight.Bold
                ),
                textAlign = TextAlign.Center
            )

            Spacer(Modifier.height(6.dp))

            Text(
                text = "Transform your portraits into editorial perfection with unlimited generations and zero watermarks.",
                style = MaterialTheme.typography.bodyMedium.copy(
                    color = Color.White.copy(alpha = 0.7f)
                ),
                textAlign = TextAlign.Center
            )

            Spacer(Modifier.height(24.dp))

            // VIP Perks List
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(Color.White.copy(alpha = 0.04f), RoundedCornerShape(16.dp))
                    .border(1.dp, Color.White.copy(alpha = 0.08f), RoundedCornerShape(16.dp))
                    .padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                VipPerkItem("⚡ Unlimited 8K Photovance Ultra-Portraits")
                VipPerkItem("👗 Virtual Wardrobe: Try on ANY outfit or gown")
                VipPerkItem("👑 Unlock all Editorial & Runway couture tiers")
                VipPerkItem("🚫 Pristine 8K exports with ZERO watermark")
                VipPerkItem("⚡ Dedicated priority GPU cloud cluster")
            }

            Spacer(Modifier.height(24.dp))

            // Plan Selection Cards
            Column(
                modifier = Modifier.fillMaxWidth(),
                verticalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                VipPlanCard(
                    plan = VipPlan.YEARLY,
                    isSelected = selectedPlan == VipPlan.YEARLY,
                    onSelect = { selectedPlan = VipPlan.YEARLY }
                )

                VipPlanCard(
                    plan = VipPlan.WEEKLY,
                    isSelected = selectedPlan == VipPlan.WEEKLY,
                    onSelect = { selectedPlan = VipPlan.WEEKLY }
                )
            }

            Spacer(Modifier.height(24.dp))

            // CTA Button
            val ctaText = if (selectedPlan == VipPlan.WEEKLY) "Start 3-Day Free Trial" else "Unlock Annual VIP Studio"
            StudioButton(
                text = ctaText,
                onClick = { onSubscribe(selectedPlan) },
                icon = Icons.Default.Stars,
                modifier = Modifier.fillMaxWidth()
            )

            Spacer(Modifier.height(14.dp))

            // Footer / Restore & Terms
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.Center,
                verticalAlignment = Alignment.CenterVertically
            ) {
                TextButton(onClick = onRestore) {
                    Text(
                        "Restore Purchases",
                        style = MaterialTheme.typography.bodySmall.copy(color = Color.White.copy(alpha = 0.6f))
                    )
                }

                Text(
                    "•",
                    style = MaterialTheme.typography.bodySmall.copy(color = Color.White.copy(alpha = 0.3f)),
                    modifier = Modifier.padding(horizontal = 4.dp)
                )

                Text(
                    "Cancel anytime in Play Store",
                    style = MaterialTheme.typography.bodySmall.copy(color = Color.White.copy(alpha = 0.4f))
                )
            }
        }
    }
}

@Composable
private fun VipPerkItem(text: String) {
    Row(
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        Box(
            modifier = Modifier
                .size(20.dp)
                .background(Color(0xFF9333EA).copy(alpha = 0.3f), CircleShape),
            contentAlignment = Alignment.Center
        ) {
            Icon(
                Icons.Default.Check,
                contentDescription = null,
                tint = Color(0xFFA855F7),
                modifier = Modifier.size(13.dp)
            )
        }
        Text(
            text = text,
            style = MaterialTheme.typography.bodyMedium.copy(
                color = Color.White.copy(alpha = 0.9f),
                fontSize = 13.5.sp
            )
        )
    }
}

@Composable
private fun VipPlanCard(
    plan: VipPlan,
    isSelected: Boolean,
    onSelect: () -> Unit
) {
    val borderColor = if (isSelected) Color(0xFFFFB800) else Color.White.copy(alpha = 0.12f)
    val bgColor = if (isSelected) Color(0xFF231C10) else Color.White.copy(alpha = 0.04f)

    Surface(
        onClick = onSelect,
        shape = RoundedCornerShape(14.dp),
        color = bgColor,
        border = BorderStroke(if (isSelected) 2.dp else 1.dp, borderColor),
        modifier = Modifier.fillMaxWidth()
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 14.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            // Radio Indicator
            Box(
                modifier = Modifier
                    .size(20.dp)
                    .border(
                        2.dp,
                        if (isSelected) Color(0xFFFFB800) else Color.White.copy(alpha = 0.4f),
                        CircleShape
                    ),
                contentAlignment = Alignment.Center
            ) {
                if (isSelected) {
                    Box(
                        modifier = Modifier
                            .size(10.dp)
                            .background(Color(0xFFFFB800), CircleShape)
                    )
                }
            }

            Spacer(Modifier.width(14.dp))

            Column(modifier = Modifier.weight(1f)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        text = plan.title,
                        style = MaterialTheme.typography.titleMedium.copy(
                            fontWeight = FontWeight.SemiBold,
                            color = Color.White
                        )
                    )
                    plan.badge?.let { badge ->
                        Spacer(Modifier.width(8.dp))
                        Box(
                            modifier = Modifier
                                .background(Color(0xFFFFB800), RoundedCornerShape(4.dp))
                                .padding(horizontal = 6.dp, vertical = 2.dp)
                        ) {
                            Text(
                                text = badge,
                                style = MaterialTheme.typography.labelSmall.copy(
                                    color = Color.Black,
                                    fontWeight = FontWeight.Bold,
                                    fontSize = 9.sp
                                )
                            )
                        }
                    }
                }
                Text(
                    text = "${plan.price} ${plan.billingPeriod}",
                    style = MaterialTheme.typography.bodySmall.copy(
                        color = Color.White.copy(alpha = 0.65f)
                    )
                )
            }
        }
    }
}
