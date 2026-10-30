package com.photovance.app.ui.screens

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import coil.compose.AsyncImage
import com.photovance.app.ui.Creation
import com.photovance.app.ui.StudioViewModel
import com.photovance.app.ui.components.*
import com.photovance.app.ui.theme.*

@Composable
fun HomeScreen(studio: StudioViewModel, onSettings: () -> Unit) {
    val scroll = rememberScrollState()
    LaunchedEffect(studio.step, studio.error, studio.isGenerating) { scroll.scrollTo(0) }
    val picker = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        if (uri != null) studio.selectPhoto(uri)
    }
    val choosePhoto = { picker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) }
    val context = LocalContext.current
    val savePermission = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) studio.save() else studio.message = "Allow storage access to save photos, or use Share."
    }
    val savePhoto = {
        if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P &&
            ContextCompat.checkSelfPermission(context, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            savePermission.launch(Manifest.permission.WRITE_EXTERNAL_STORAGE)
        } else studio.save()
    }

    if (studio.showPaywall) {
        PaywallSheet(
            onDismiss = studio::dismissPaywall,
            onSubscribe = studio::subscribeVip,
            onRestore = studio::restorePurchases
        )
    }

    Box(modifier = Modifier.fillMaxSize().background(WebUiBgDark)) {
        // Atmospheric Ambient Glow (WebUI .glow-bg recreation)
        AtmosphericGlowBackground()

        Scaffold(
            containerColor = Color.Transparent,
            topBar = {
                val engineMode by studio.engineMode.collectAsState()
                val isVip by studio.isVip.collectAsState()
                val credits by studio.remainingCredits.collectAsState()
                Column {
                    StudioHeader(
                        onSettings = onSettings,
                        onBack = if (studio.step > 0 && !studio.isGenerating) ({ studio.goToStep(studio.step - 1) }) else null,
                        vipBadgeTitle = if (isVip) "👑 VIP" else "✨ $credits Free",
                        onTapVip = { studio.openPaywall() },
                        engineModeTitle = if (studio.hybridRouter.deviceSpec.supportsLocalNpuStudio) {
                            if (engineMode == com.photovance.app.engine.StudioEngineMode.LOCAL_OFFLINE) "⚡ Local NPU" else "☁️ Cloud 8K"
                        } else "☁️ Cloud 8K",
                        onToggleEngine = { studio.hybridRouter.toggleMode() }
                    )
                    StudioSteps(studio.step, studio.hasPhoto, studio.result != null, !studio.isGenerating, studio::goToStep)
                }
            },
            bottomBar = {
                val isVip by studio.isVip.collectAsState()
                Surface(
                    color = WebUiBgDark.copy(alpha = 0.95f),
                    border = BorderStroke(1.dp, WebUiGlassBorderGradient)
                ) {
                    Column(Modifier.fillMaxWidth().imePadding(),
                        horizontalAlignment = Alignment.CenterHorizontally) {
                        Column(Modifier.fillMaxWidth().padding(horizontal = 24.dp, vertical = 12.dp),
                            verticalArrangement = Arrangement.spacedBy(8.dp)) {
                            when (studio.step) {
                                0 -> StudioButton(
                                    if (studio.hasPhoto) "Continue to styling" else "Choose a photo",
                                    onClick = if (studio.hasPhoto) ({ studio.goToStep(1) }) else choosePhoto,
                                    icon = if (studio.hasPhoto) Icons.AutoMirrored.Filled.ArrowForward else Icons.Default.AddPhotoAlternate
                                )
                                1 -> StudioButton(
                                    if (studio.isGenerating) "Creating your look · ${studio.elapsedSeconds}s" else "Generate photo",
                                    onClick = studio::generate,
                                    icon = Icons.Default.AutoAwesome,
                                    enabled = studio.prompt.isNotBlank(),
                                    loading = studio.isGenerating
                                )
                                2 -> StudioButton(
                                    if (studio.isExporting) "Preparing photo…" else "Save photo",
                                    onClick = savePhoto,
                                    icon = Icons.Default.Download,
                                    loading = studio.isExporting
                                )
                            }
                            if (studio.step == 1) Text(
                                "Your photo is securely processed with 1:1 facial biometric lock.",
                                style = MaterialTheme.typography.bodySmall, color = WebUiTextDim
                            )
                        }
                        com.photovance.app.ads.AdBanner(isVip = isVip)
                    }
                }
            }
        ) { padding ->
            Column(
                Modifier
                    .fillMaxSize()
                    .padding(padding)
                    .verticalScroll(scroll)
                    .padding(horizontal = 20.dp)
                    .padding(top = 16.dp, bottom = 12.dp),
                verticalArrangement = Arrangement.spacedBy(20.dp)
            ) {
                studio.error?.let { ErrorNotice(it, onSettings) }
                if (studio.isGenerating) GenerationProgress(studio.progressLabel, studio.elapsedSeconds)
                when (studio.step) {
                    0 -> PhotoScreen(studio.photoModel, studio.hasPhoto, choosePhoto, studio::selectSample)
                    1 -> StyleScreen(studio, choosePhoto, onSettings)
                    2 -> studio.result?.let { ResultScreen(it, studio.isExporting, studio, studio::share, studio::shareStory,
                        { studio.goToStep(if (studio.hasPhoto) 1 else 0) }) }
                }
                if (studio.step == 0) {
                    ConnectionStatus(studio.online, studio.checkingConnection, onSettings)
                    if (studio.recent.isNotEmpty()) RecentCreations(studio.recent, studio::openCreation)
                }
            }
        }
    }
}

@Composable
fun PhotoScreen(photo: Any?, hasPhoto: Boolean, onChoose: () -> Unit, onSample: () -> Unit) {
    PageHeading("01 / Reference Photo", "Start with your photo.",
        "High-fidelity biometric lock preserves 100% facial features and skin pores.")

    Surface(
        onClick = onChoose,
        shape = MaterialTheme.shapes.large,
        color = WebUiBgCard,
        border = BorderStroke(1.5.dp, if (hasPhoto) WebUiBorderHover else Color(0x330EA5E9)),
        modifier = Modifier
            .fillMaxWidth()
            .heightIn(min = 280.dp, max = 420.dp)
            .aspectRatio(0.88f)
    ) {
        if (hasPhoto) {
            Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                AsyncImage(
                    model = photo,
                    contentDescription = "Your selected photo. Tap to change.",
                    contentScale = ContentScale.Fit,
                    modifier = Modifier.fillMaxSize()
                )
                // WebUI Segformer tag
                Surface(
                    modifier = Modifier
                        .align(Alignment.BottomStart)
                        .padding(12.dp),
                    shape = RoundedCornerShape(8.dp),
                    color = Color(0xCC0EA5E9)
                ) {
                    Text(
                        "Segformer Mask Ready",
                        modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp),
                        style = MaterialTheme.typography.labelSmall,
                        fontWeight = FontWeight.Bold,
                        color = Color.White
                    )
                }
            }
        } else {
            Column(
                Modifier.padding(28.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(14.dp)
            ) {
                Box(
                    Modifier
                        .size(68.dp)
                        .background(Color(0x1F0EA5E9), CircleShape)
                        .border(1.dp, Color(0x330EA5E9), CircleShape),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        Icons.Default.AddPhotoAlternate, null,
                        Modifier.size(30.dp),
                        tint = WebUiPrimary
                    )
                }
                Text("Your next look starts here", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                Text(
                    "Choose a selfie or full-body photo.\nKeep your face and outfit in view.",
                    color = WebUiTextMuted,
                    style = MaterialTheme.typography.bodyMedium,
                    textAlign = androidx.compose.ui.text.style.TextAlign.Center
                )
                Surface(
                    onClick = onSample,
                    shape = CircleShape,
                    color = Color(0x1A0EA5E9),
                    border = BorderStroke(1.dp, Color(0x4D0EA5E9))
                ) {
                    Text(
                        "Just exploring? Try a sample",
                        style = MaterialTheme.typography.labelMedium,
                        fontWeight = FontWeight.SemiBold,
                        color = WebUiPrimary,
                        modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp)
                    )
                }
            }
        }
    }
    if (hasPhoto) {
        StudioOutlineButton("Change photo", onClick = onChoose, icon = Icons.Default.PhotoLibrary)
        TextButton(onClick = onSample, modifier = Modifier.fillMaxWidth()) {
            Icon(Icons.Default.AutoAwesome, null, Modifier.size(17.dp), tint = WebUiPrimary)
            Spacer(Modifier.width(8.dp))
            Text("Use the sample portrait instead", color = WebUiPrimary)
        }
    }
}

@Composable
private fun ConnectionStatus(online: Boolean, checking: Boolean, onSettings: () -> Unit) {
    Surface(
        shape = MaterialTheme.shapes.medium,
        color = WebUiBgCard,
        border = BorderStroke(1.dp, WebUiBorderCard)
    ) {
        Row(
            Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 12.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Box(
                Modifier
                    .size(8.dp)
                    .background(if (online) WebUiStatusGreen else Color(0xFFEF4444), CircleShape)
            )
            Spacer(Modifier.width(10.dp))
            Text(
                if (checking) "Connecting to your studio…" else if (online) "Your studio is ready (Port 7870)" else "Connect your studio to generate",
                style = MaterialTheme.typography.bodySmall,
                color = WebUiTextMuted,
                modifier = Modifier.weight(1f)
            )
            TextButton(onClick = onSettings) { Text("Settings", color = WebUiPrimary) }
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
fun StyleScreen(studio: StudioViewModel, onChoose: () -> Unit, onOptions: () -> Unit) {
    Column(verticalArrangement = Arrangement.spacedBy(18.dp)) {
        PageHeading("02 / Make it yours", "Find your next look.", "Choose a direction. Add a little imagination.")

        // Reference Photo Mini Bar
        Surface(
            onClick = onChoose,
            enabled = !studio.isGenerating,
            shape = MaterialTheme.shapes.medium,
            color = WebUiBgCard,
            border = BorderStroke(1.dp, WebUiGlassBorderGradient)
        ) {
            Row(
                Modifier.fillMaxWidth().padding(12.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(14.dp)
            ) {
                AsyncImage(
                    studio.photoModel, "Your selected photo",
                    contentScale = ContentScale.Crop,
                    modifier = Modifier.size(56.dp, 62.dp).clip(MaterialTheme.shapes.small)
                )
                Column(Modifier.weight(1f)) {
                    Text("Your photo", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.Bold, color = WebUiTextMain)
                    Text("Ready for a new perspective", style = MaterialTheme.typography.bodySmall, color = WebUiTextMuted)
                }
                Icon(Icons.Default.ChevronRight, "Change photo", tint = WebUiTextMuted)
            }
        }

        // Section 1: Studio Mode Switcher
        WebUiCard {
            Text("1. Generation Mode", style = MaterialTheme.typography.labelLarge, color = WebUiTextMuted)

            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                // Mode 1: Outfit Swap
                val isOutfitSwap = studio.mode == "in_place"
                Surface(
                    onClick = { studio.mode = "in_place" },
                    modifier = Modifier.weight(1f),
                    shape = MaterialTheme.shapes.medium,
                    color = if (isOutfitSwap) Color(0x330EA5E9) else Color(0x400F172A),
                    border = BorderStroke(if (isOutfitSwap) 1.5.dp else 1.dp, if (isOutfitSwap) WebUiPrimary else WebUiBorderCard)
                ) {
                    Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                            Text("👗", fontSize = 18.sp)
                            Text("Outfit Swap", style = MaterialTheme.typography.labelLarge, fontWeight = FontWeight.Bold, color = WebUiTextMain)
                        }
                        Text("100% original pose", style = MaterialTheme.typography.labelSmall, color = WebUiTextDim, fontSize = 10.sp)
                    }
                }

                // Mode 2: Full Scene
                val isFullScene = studio.mode == "face_studio" || studio.mode == "full_scene"
                Surface(
                    onClick = { studio.mode = "face_studio" },
                    modifier = Modifier.weight(1f),
                    shape = MaterialTheme.shapes.medium,
                    color = if (isFullScene) Color(0x338B5CF6) else Color(0x400F172A),
                    border = BorderStroke(if (isFullScene) 1.5.dp else 1.dp, if (isFullScene) WebUiAccent else WebUiBorderCard)
                ) {
                    Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                            Text("✨", fontSize = 18.sp)
                            Text("Full Scene", style = MaterialTheme.typography.labelLarge, fontWeight = FontWeight.Bold, color = WebUiTextMain)
                        }
                        Text("New background", style = MaterialTheme.typography.labelSmall, color = WebUiTextDim, fontSize = 10.sp)
                    }
                }
            }

            // Virtual Wardrobe optional pill
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.SpaceBetween, modifier = Modifier.fillMaxWidth()) {
                FilterChip(
                    selected = studio.mode == "wardrobe",
                    onClick = { studio.mode = "wardrobe" },
                    enabled = !studio.isGenerating,
                    label = { Text("Virtual Wardrobe 👗") }
                )
                Text(
                    when (studio.mode) {
                        "face_studio" -> "Generates a brand new scene with 1:1 facial lock."
                        "wardrobe" -> "Upload any garment photo to drape directly."
                        else -> "Mask inpaint; 100% original face & body."
                    },
                    style = MaterialTheme.typography.labelSmall, color = WebUiTextMuted
                )
            }
        }

        if (studio.mode == "wardrobe") {
            VirtualWardrobePicker(
                garmentUri = studio.garmentUri,
                onSelectGarment = studio::setGarment,
                onApplyTag = { tag ->
                    studio.editPrompt("${studio.prompt.trimEnd().trimEnd(',')}, wearing $tag".trimStart(',', ' '))
                }
            )
        }

        // Section 2: Competitive Model Suite
        WebUiCard {
            Text("2. Competitive Model Suite", style = MaterialTheme.typography.labelLarge, color = WebUiTextMuted)
            FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                listOf(
                    "cyberrealistic_xl" to "CyberRealistic XL ★ King",
                    "epicrealism_xl" to "epiCRealism XL",
                    "realvis_xl" to "RealVisXL V5.0",
                    "flux_klein" to "Flux-2-Klein 4B Turbo"
                ).forEach { (key, label) ->
                    FilterChip(
                        selected = studio.modelKey == key,
                        onClick = { studio.modelKey = key },
                        enabled = !studio.isGenerating,
                        label = { Text(label) }
                    )
                }
            }
            Surface(
                shape = MaterialTheme.shapes.small,
                color = Color(0x331E293B),
                border = BorderStroke(1.dp, WebUiBorderCard)
            ) {
                Text(
                    when (studio.modelKey) {
                        "cyberrealistic_xl" -> "True DSLR micro-pores, natural epidermal depth, zero AI plastic smoothing."
                        "epicrealism_xl" -> "Studio lighting, cinematic contrast, lifelike hair & authentic fabric drape."
                        "realvis_xl" -> "Reliable, razor-sharp all-rounder portrait generation with lifelike detail."
                        "flux_klein" -> "Next-gen DiT with Qwen3 Heretic text encoder for unrestricted prompt adherence."
                        else -> "Competitive neural generation model."
                    },
                    modifier = Modifier.padding(10.dp),
                    style = MaterialTheme.typography.bodySmall, color = WebUiTextMuted
                )
            }
        }

        // Section 3: Custom Style & Fashion LoRA Suite
        WebUiCard {
            Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.SpaceBetween) {
                Text("3. Custom Style & Fashion LoRA", style = MaterialTheme.typography.labelLarge, color = WebUiTextMuted)
                val badgeText = studio.selectedLoraId?.let { studio.lorasMap[it]?.badge ?: "Active" } ?: "Optional"
                Surface(color = WebUiPrimary.copy(alpha = 0.15f), shape = MaterialTheme.shapes.extraSmall) {
                    Text(badgeText, modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp),
                        style = MaterialTheme.typography.labelSmall, color = WebUiPrimary, fontWeight = FontWeight.Bold)
                }
            }

            val loraOptions = listOf(
                null to "None (Base Model)",
                "fs_couture" to (studio.lorasMap["fs_couture"]?.name ?: "Haute Couture Gowns"),
                "Hyper-SDXL-8steps-CFG-lora" to (studio.lorasMap["Hyper-SDXL-8steps-CFG-lora"]?.name ?: "Hyper-SDXL 8-Step Turbo"),
                "sd_xl_offset_example-lora_1.0" to (studio.lorasMap["sd_xl_offset_example-lora_1.0"]?.name ?: "Dynamic Contrast & Shadow")
            ) + studio.lorasMap.keys.filter { it !in listOf("fs_couture", "Hyper-SDXL-8steps-CFG-lora", "sd_xl_offset_example-lora_1.0") }
                .map { it to (studio.lorasMap[it]?.name ?: it) }

            FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                loraOptions.forEach { (key, label) ->
                    FilterChip(
                        selected = studio.selectedLoraId == key,
                        onClick = { studio.selectLora(key) },
                        enabled = !studio.isGenerating,
                        label = { Text(label) }
                    )
                }
            }

            if (studio.selectedLoraId != null) {
                val activeLora = studio.lorasMap[studio.selectedLoraId]
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(MaterialTheme.shapes.small)
                        .background(Color(0x331E293B))
                        .border(1.dp, WebUiBorderCard, MaterialTheme.shapes.small)
                        .padding(12.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
                        Text("LoRA Strength", style = MaterialTheme.typography.bodySmall, fontWeight = FontWeight.Bold, color = WebUiTextMain)
                        Text(String.format(java.util.Locale.US, "%.2f", studio.loraWeight), style = MaterialTheme.typography.bodySmall, color = WebUiPrimary, fontWeight = FontWeight.Bold)
                    }
                    Slider(
                        value = studio.loraWeight,
                        onValueChange = studio::setLoraWeightValue,
                        valueRange = 0.1f..1.2f,
                        steps = 21,
                        enabled = !studio.isGenerating
                    )
                    if (!activeLora?.trigger.isNullOrBlank()) {
                        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            Text("Trigger Tag:", style = MaterialTheme.typography.labelSmall, color = WebUiTextMuted)
                            SuggestionChip(
                                onClick = studio::injectLoraTrigger,
                                enabled = !studio.isGenerating,
                                label = { Text("+ Add ${activeLora?.id ?: "Trigger"}", color = WebUiPrimary, fontWeight = FontWeight.Bold) }
                            )
                        }
                    }
                    Text(
                        activeLora?.desc ?: "Enhance garments with trained bespoke fashion lines or couture fabrics.",
                        style = MaterialTheme.typography.bodySmall,
                        color = WebUiTextMuted
                    )
                }
            } else {
                Text(
                    "Enhance garments with trained bespoke fashion lines, embroidery, or couture fabrics.",
                    style = MaterialTheme.typography.bodySmall, color = WebUiTextDim
                )
            }
        }

        // Section 4: Target Garment (when in outfit swap mode)
        if (studio.mode == "in_place") {
            WebUiCard {
                Text("4. Target Garment", style = MaterialTheme.typography.labelLarge, color = WebUiTextMuted)
                FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    listOf("dress" to "Dress / Gown", "upper" to "Tops / Blouse", "lower" to "Bottoms / Pants", "all" to "Full Outfit").forEach { (key, title) ->
                        FilterChip(selected = studio.target == key, onClick = { studio.target = key },
                            enabled = !studio.isGenerating, label = { Text(title) })
                    }
                }
            }
        }

        // Section 5: Outfit Prompt & Pre-prompts
        WebUiCard {
            Text("5. Outfit & Scene Description", style = MaterialTheme.typography.labelLarge, color = WebUiTextMuted)
            OutlinedTextField(
                value = studio.prompt,
                onValueChange = studio::editPrompt,
                label = { Text("Describe your look") },
                placeholder = { Text("Describe outfit, fabric, color, cut, or styling...") },
                enabled = !studio.isGenerating,
                minLines = 3,
                maxLines = 6,
                modifier = Modifier.fillMaxWidth(),
                shape = MaterialTheme.shapes.medium
            )

            Text("Quick Ideas (Old Hollywood & Sultry Glamour):", style = MaterialTheme.typography.labelMedium, fontWeight = FontWeight.Bold, color = WebUiPrimary)
            LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                val quickIdeas = listOf(
                    "Old Hollywood Noir" to "sensual vintage 1950s Old Hollywood black velvet off-shoulder gown, sweetheart neckline, form-fitting mermaid silhouette, sheer black opera gloves, diamond collar necklace, moody classic studio backlight, sultry expression, cinematic grain, Kodak Tri-X 400 aesthetic",
                    "Riviera Silk Slip" to "alluring old money Amalfi aesthetic, backless ivory liquid silk slip dress, draped cowl neckline, high thigh slit, delicate gold chain anklet, golden hour sunset over Mediterranean balcony, soft coastal breeze, rich vintage glamour",
                    "Couture Corset" to "sultry French sheer black lace corset gown with structured boning, semi-translucent chiffon floor-length skirt, delicate floral lace embroidery, subtle cleavage, dramatic red carpet spotlight, seductive posture, high-fashion editorial Vogue",
                    "Studio 54 Gold" to "1970s vintage Studio 54 glamour, plunge V-neck liquid gold lamé halter dress, open back, pleated flowing silhouette, retro disco party lighting, shimmering golden highlights, seductive alluring gaze, vintage 35mm film photograph",
                    "Scarlet Siren" to "breathtaking scarlet red silk satin mermaid gown, low-cut sweetheart bodice, thigh-high slit, delicate strap detailing, glamorous retro wavy hair, warm paparazzi flash and grand chandelier ambient bokeh, seductive confident posture",
                    "Parisian Boudoir" to "seductive 1920s vintage Parisian boudoir silk robe and lace slip, marabou feather trim cuffs, deep emerald green satin, candlelit vintage suite, sultry captivating mood, soft warm shadows, luxury retro aesthetic",
                    "Cashmere & Mini" to "effortlessly sexy old-money aesthetic, deep V-neck slouchy ribbed cream cashmere sweater slightly off one shoulder, fitted black lambskin leather mini skirt, sheer stockings, warm vintage library lighting, seductive candid portrait",
                    "Emerald Velvet" to "emerald green velvet evening gown with intricate golden floral embroidery, tailored silhouette, plunging neckline, high-fashion silk",
                    "Parisian Suit" to "tailored Parisian double-breasted cream tweed blazer and high-waisted wide-leg trousers with gold buttons"
                )
                items(quickIdeas) { (title, promptText) ->
                    SuggestionChip(
                        onClick = { studio.editPrompt(promptText) },
                        enabled = !studio.isGenerating,
                        label = { Text(title) }
                    )
                }
            }

            Text("The Inspiration Edit", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
            LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                items(listOf("All", "Vintage & Retro", "Sultry Glamour", "Old Money", "Red Carpet", "Vacation", "Street Style", "Casual")) { category ->
                    FilterChip(selected = studio.category == category, onClick = { studio.category = category },
                        enabled = !studio.isGenerating, label = { Text(category) })
                }
            }
            LazyRow(horizontalArrangement = Arrangement.spacedBy(12.dp), contentPadding = PaddingValues(vertical = 4.dp)) {
                items(VISUAL_STYLE_PRESETS.filter { studio.category == "All" || it.category == studio.category }, key = { it.id }) { preset ->
                    StyleInspirationCard(preset, studio.selectedPresetId == preset.id,
                        onClick = { if (!studio.isGenerating) studio.selectPreset(preset.id) })
                }
            }

            LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                items(listOf("Golden hour", "Studio lighting", "Soft film", "City at night", "Vintage Grain", "Candlelight")) { modifier ->
                    SuggestionChip(onClick = { studio.editPrompt("${studio.prompt.trimEnd().trimEnd(',')}, $modifier".trimStart(',', ' ')) },
                        enabled = !studio.isGenerating, label = { Text("+ $modifier") })
                }
            }
        }

        // Section 6: Format & Variations
        WebUiCard {
            Text("Photo format", style = MaterialTheme.typography.labelLarge, color = WebUiTextMuted)
            FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                listOf("portrait" to "4:5 Portrait", "story" to "9:16 Story", "square" to "1:1 Square").forEach { (key, label) ->
                    FilterChip(selected = studio.aspectRatio == key, onClick = { studio.aspectRatio = key },
                        enabled = !studio.isGenerating, label = { Text(label) })
                }
            }

            Text("Studio variations", style = MaterialTheme.typography.labelLarge, color = WebUiTextMuted)
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                listOf(1 to "1 Master Photo", 4 to "4 Variations ✨").forEach { (count, label) ->
                    FilterChip(
                        selected = studio.variationsCount == count,
                        onClick = { studio.variationsCount = count },
                        enabled = !studio.isGenerating,
                        label = { Text(label) }
                    )
                }
            }
        }

        TextButton(onClick = onOptions, enabled = !studio.isGenerating) {
            Icon(Icons.Default.Tune, null, Modifier.size(18.dp), tint = WebUiPrimary)
            Spacer(Modifier.width(8.dp))
            Text("More options", color = WebUiPrimary)
        }
    }
}

@Composable
private fun GenerationProgress(label: String, elapsed: Int) {
    Surface(
        color = WebUiBgCard,
        shape = MaterialTheme.shapes.large,
        border = BorderStroke(1.dp, WebUiGlassBorderGradient)
    ) {
        Column(
            Modifier.fillMaxWidth().padding(22.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            CircularProgressIndicator(Modifier.size(36.dp), strokeWidth = 3.dp, color = WebUiPrimary)
            Text("Synthesizing Outfit...", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold, color = WebUiTextMain)
            Text(label, style = MaterialTheme.typography.bodySmall, color = WebUiPrimary)
            LinearProgressIndicator(
                Modifier.fillMaxWidth().clip(RoundedCornerShape(4.dp)),
                color = WebUiPrimary,
                trackColor = Color(0x33FFFFFF)
            )
            Text("Elapsed: ${elapsed}s · Neural engine active on port 7870",
                style = MaterialTheme.typography.bodySmall, color = WebUiTextDim)
        }
    }
}

@Composable
fun ResultScreen(
    creation: Creation,
    exporting: Boolean,
    studio: StudioViewModel,
    onShare: () -> Unit,
    onShareStory: () -> Unit,
    onAnother: () -> Unit
) {
    var showOriginal by rememberSaveable(creation.imageUrl) { mutableStateOf(false) }
    Column(verticalArrangement = Arrangement.spacedBy(18.dp)) {
        Row(
            Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            PageHeading("03 / The reveal", "Your new look.", "Same you. A whole new perspective.")
            Surface(
                shape = CircleShape,
                color = WebUiPrimary.copy(alpha = 0.15f),
                border = BorderStroke(1.dp, WebUiPrimary.copy(alpha = 0.5f))
            ) {
                Text(
                    "100% Face Locked",
                    modifier = Modifier.padding(horizontal = 10.dp, vertical = 5.dp),
                    style = MaterialTheme.typography.labelSmall,
                    fontWeight = FontWeight.Bold,
                    color = WebUiPrimary
                )
            }
        }

        // Before & After Interactive Split Slider
        BeforeAfterSlider(
            creation.beforeUrl, creation.imageUrl,
            modifier = Modifier.fillMaxWidth().aspectRatio(0.8f),
            isHoldPeekOriginal = showOriginal
        )

        Row(
            Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text("Drag to compare", style = MaterialTheme.typography.bodySmall, color = WebUiTextMuted)
            TextButton(onClick = { showOriginal = !showOriginal }) {
                Text(if (showOriginal) "Show comparison" else "View original", color = WebUiPrimary)
            }
        }

        // WebUI 4-Metric Studio Canvas Information Strip
        GenerationInfoBar(
            modelName = creation.model ?: studio.modelKey,
            modeName = if (studio.mode == "face_studio") "Full Scene" else "Outfit Swap",
            elapsed = creation.elapsed
        )

        if (creation.variationUrls.size > 1) {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("Studio Variations (${creation.variationUrls.size})", style = MaterialTheme.typography.labelLarge, color = WebUiTextMuted)
                LazyRow(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    itemsIndexed(creation.variationUrls) { idx, url ->
                        val isSelected = url == creation.imageUrl
                        Box(
                            modifier = Modifier
                                .size(74.dp, 94.dp)
                                .clip(MaterialTheme.shapes.small)
                                .border(
                                    if (isSelected) 2.dp else 1.dp,
                                    if (isSelected) WebUiPrimary else WebUiBorderCard,
                                    MaterialTheme.shapes.small
                                )
                                .clickable { studio.selectVariation(idx) }
                        ) {
                            AsyncImage(
                                model = url,
                                contentDescription = "Variation ${idx + 1}",
                                contentScale = ContentScale.Crop,
                                modifier = Modifier.fillMaxSize()
                            )
                            if (isSelected) {
                                Box(
                                    modifier = Modifier
                                        .align(Alignment.BottomCenter)
                                        .background(WebUiGradient)
                                        .fillMaxWidth()
                                        .padding(vertical = 1.dp),
                                    contentAlignment = Alignment.Center
                                ) {
                                    Text("Active", fontSize = 9.sp, fontWeight = FontWeight.Bold, color = Color.White)
                                }
                            }
                        }
                    }
                }
            }
        }

        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            StudioOutlineButton("Share photo", onClick = onShare, icon = Icons.Default.Share, enabled = !exporting, modifier = Modifier.weight(1f))
            StudioOutlineButton("Story (9:16)", onClick = onShareStory, icon = Icons.Default.AutoStories, enabled = !exporting, modifier = Modifier.weight(1f))
        }

        TextButton(onClick = onAnother, enabled = !exporting, modifier = Modifier.fillMaxWidth()) {
            Icon(Icons.Default.Refresh, null, Modifier.size(18.dp), tint = WebUiPrimary)
            Spacer(Modifier.width(8.dp))
            Text("Try another style", color = WebUiPrimary, fontWeight = FontWeight.Bold)
        }
    }
}

@Composable
private fun RecentCreations(creations: List<Creation>, onOpen: (Creation) -> Unit) {
    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Text("Your recent looks", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
        LazyRow(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            items(creations, key = { it.imageUrl }) { creation ->
                AsyncImage(
                    creation.imageUrl, "Open recent creation: ${creation.prompt}",
                    contentScale = ContentScale.Crop,
                    modifier = Modifier
                        .size(112.dp, 142.dp)
                        .clip(MaterialTheme.shapes.medium)
                        .border(1.dp, WebUiBorderCard, MaterialTheme.shapes.medium)
                        .clickable { onOpen(creation) }
                )
            }
        }
        Text("Recent looks are kept for this session.", style = MaterialTheme.typography.bodySmall, color = WebUiTextDim)
    }
}
