package com.photovance.app.ui.components

import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.gestures.detectHorizontalDragGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.SwapHoriz
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawWithContent
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.clipRect
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.semantics.*
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import coil.compose.AsyncImage
import com.photovance.app.ui.theme.*

@Composable
fun BeforeAfterSlider(
    beforeUrl: String,
    afterUrl: String,
    modifier: Modifier = Modifier,
    isHoldPeekOriginal: Boolean = false
) {
    var rawSplit by remember(beforeUrl, afterUrl) { mutableFloatStateOf(0.5f) }
    val split by animateFloatAsState(
        if (isHoldPeekOriginal) 1f else rawSplit,
        animationSpec = tween(150),
        label = "comparison"
    )

    BoxWithConstraints(
        modifier = modifier
            .clip(MaterialTheme.shapes.large)
            .background(WebUiBgDark)
            .border(1.dp, WebUiGlassBorderGradient, MaterialTheme.shapes.large)
            .semantics(mergeDescendants = true) {
                contentDescription = "Before and after photo comparison"
                stateDescription = "${(split * 100).toInt()} percent original"
                progressBarRangeInfo = ProgressBarRangeInfo(rawSplit, 0f..1f)
                setProgress { rawSplit = it.coerceIn(0f, 1f); true }
            }
            .pointerInput(beforeUrl, afterUrl, isHoldPeekOriginal) {
                if (!isHoldPeekOriginal) detectHorizontalDragGestures { change, distance ->
                    change.consume()
                    rawSplit = (rawSplit + distance / size.width).coerceIn(0f, 1f)
                }
            }
    ) {
        // Transformed (After) Layer
        AsyncImage(afterUrl, null, contentScale = ContentScale.Crop, modifier = Modifier.fillMaxSize())

        // Original (Before) Layer with clip
        AsyncImage(
            beforeUrl, null, contentScale = ContentScale.Crop,
            modifier = Modifier
                .fillMaxSize()
                .drawWithContent {
                    clipRect(right = size.width * split) { this@drawWithContent.drawContent() }
                }
        )

        val divider = maxWidth * split
        if (!isHoldPeekOriginal) {
            // Neon Divider Line
            Box(
                Modifier
                    .offset(x = divider - 1.dp)
                    .width(2.dp)
                    .fillMaxHeight()
                    .background(Color.White)
            )

            // Center Sleek Handle
            Box(
                Modifier
                    .align(Alignment.CenterStart)
                    .offset(x = divider - 20.dp)
                    .size(40.dp)
                    .clip(CircleShape)
                    .background(Color(0xE6090C10))
                    .border(1.5.dp, WebUiPrimary, CircleShape),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    Icons.Default.SwapHoriz,
                    null,
                    tint = WebUiPrimary,
                    modifier = Modifier.size(22.dp)
                )
            }
        }

        // WebUI Floating Layer Badges
        Row(
            Modifier
                .fillMaxWidth()
                .align(Alignment.TopCenter)
                .padding(14.dp),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            // Original Badge
            Box(
                modifier = Modifier
                    .clip(RoundedCornerShape(8.dp))
                    .background(Color(0xCC090C10))
                    .border(1.dp, Color(0x33FFFFFF), RoundedCornerShape(8.dp))
                    .padding(horizontal = 10.dp, vertical = 5.dp)
            ) {
                Text("Original", color = Color.White, fontSize = 11.sp, fontWeight = FontWeight.SemiBold)
            }

            // Transformed Badge
            if (!isHoldPeekOriginal) {
                Box(
                    modifier = Modifier
                        .clip(RoundedCornerShape(8.dp))
                        .background(WebUiGradient)
                        .padding(horizontal = 10.dp, vertical = 5.dp)
                ) {
                    Text("Transformed", color = Color.White, fontSize = 11.sp, fontWeight = FontWeight.Bold)
                }
            }
        }
    }
}
