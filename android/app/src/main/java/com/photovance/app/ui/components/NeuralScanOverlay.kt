package com.photovance.app.ui.components

import androidx.compose.animation.core.*
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.photovance.app.ui.theme.*

@Composable
fun NeuralScanOverlay(
    elapsedSeconds: Int,
    modelName: String,
    modifier: Modifier = Modifier
) {
    // 1. Vertical Laser Sweep Animation
    val infiniteTransition = rememberInfiniteTransition(label = "laserSweepTransition")
    val sweepProgress by infiniteTransition.animateFloat(
        initialValue = 0.05f,
        targetValue = 0.95f,
        animationSpec = infiniteRepeatable(
            animation = tween(durationMillis = 2200, easing = LinearEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "laserSweepY"
    )

    // Milestone text mapping
    val (stepTitle, stepDetail) = when (elapsedSeconds) {
        in 0..3 -> "Step 1/5: Analyzing Facial Geometry" to "Detecting 68 landmarks & eye vector alignment..."
        in 4..8 -> "Step 2/5: Identity Locking" to "Preserving skin texture, micro-pores & facial likeness..."
        in 9..15 -> "Step 3/5: Latent DiT Synthesis" to "Draping haute couture fabric in unrestricted latent space..."
        in 16..23 -> "Step 4/5: Studio Lighting & Shadow" to "Rendering 8K photographic illumination & environment reflection..."
        else -> "Step 5/5: Ultra-HD Mastering" to "Finalizing high-frequency details & composite pass..."
    }

    Box(
        modifier = modifier
            .fillMaxSize()
            .background(Color(0x7707090E))
    ) {
        // Laser Sweep Line
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .fillMaxHeight(sweepProgress)
                .align(Alignment.TopCenter)
        ) {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(3.dp)
                    .align(Alignment.BottomCenter)
                    .background(
                        Brush.horizontalGradient(
                            listOf(
                                Color.Transparent,
                                NeonCyan.copy(alpha = 0.4f),
                                NeonCyan,
                                NeonViolet,
                                NeonCyan,
                                NeonCyan.copy(alpha = 0.4f),
                                Color.Transparent
                            )
                        )
                    )
            )
        }

        // Futuristic Center / Bottom Progress HUD Card
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(14.dp)
                .align(Alignment.BottomCenter)
                .clip(RoundedCornerShape(16.dp))
                .background(Color(0xDD0B0F19))
                .border(1.dp, NeonCyan.copy(alpha = 0.6f), RoundedCornerShape(16.dp))
                .padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(16.dp),
                        color = NeonCyan,
                        strokeWidth = 2.dp
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = "🧬 NEURAL SYNTHESIS",
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Bold,
                        color = NeonCyan,
                        letterSpacing = 1.sp
                    )
                }

                Text(
                    text = "${elapsedSeconds}s",
                    fontSize = 13.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color.White
                )
            }

            Text(
                text = stepTitle,
                fontSize = 13.sp,
                fontWeight = FontWeight.Bold,
                color = TextMain
            )

            Text(
                text = stepDetail,
                fontSize = 11.sp,
                color = TextMuted,
                lineHeight = 15.sp
            )

            Spacer(modifier = Modifier.height(2.dp))

            // Subtitle status badge
            Text(
                text = "⚡ Engine: $modelName · High-Fidelity Studio",
                fontSize = 10.sp,
                fontWeight = FontWeight.SemiBold,
                color = NeonViolet
            )
        }
    }
}
