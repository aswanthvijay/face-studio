package com.photovance.app

import android.app.Application
import android.content.Context
import android.os.Looper
import androidx.activity.ComponentActivity
import androidx.compose.foundation.layout.*
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.Density
import androidx.compose.ui.test.*
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.cancel
import com.google.gson.JsonParser
import com.github.takahirom.roborazzi.captureRoboImage
import com.photovance.app.network.ApiClient
import com.photovance.app.ui.PhotovanceApp
import com.photovance.app.ui.StudioViewModel
import com.photovance.app.ui.screens.HomeScreen
import com.photovance.app.ui.theme.AppAppearance
import com.photovance.app.ui.theme.PhotovanceTheme
import okhttp3.mockwebserver.Dispatcher
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import okhttp3.mockwebserver.RecordedRequest
import okio.Buffer
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment
import org.robolectric.Shadows.shadowOf
import org.robolectric.annotation.Config
import org.robolectric.annotation.GraphicsMode
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34], qualifiers = "w393dp-h851dp-xhdpi")
@GraphicsMode(GraphicsMode.Mode.NATIVE)
class StudioFlowTest {
    @get:Rule val compose = createAndroidComposeRule<MainActivity>()
    private lateinit var server: MockWebServer
    private lateinit var studio: StudioViewModel
    private val generationFails = AtomicBoolean(false)
    private val requestBody = AtomicReference("")
    private lateinit var previousUrl: String

    @Before fun setUp() {
        val app = RuntimeEnvironment.getApplication()
        app.getSharedPreferences("studio_design", Context.MODE_PRIVATE).edit().clear().putString("appearance", "Dark").commit()
        val portrait = app.resources.openRawResource(R.drawable.cc0_studio).use { it.readBytes() }
        server = MockWebServer()
        server.dispatcher = object : Dispatcher() {
            override fun dispatch(request: RecordedRequest): MockResponse = when (request.path) {
                "/api/status" -> json("""{"ready":true,"default_negative_prompt":"blurry"}""")
                "/api/upload" -> json("""{"image_id":"test-photo","image_url":"/inputs/original.jpg"}""")
                "/api/change-dress" -> {
                    requestBody.set(request.body.readUtf8())
                    if (generationFails.get()) json("""{"success":false,"error":"Please try another description."}""")
                    else json("""{"success":true,"result_url":"/outputs/result.jpg","elapsed":1.2}""")
                }
                "/inputs/original.jpg", "/outputs/result.jpg" -> MockResponse().setHeader("Content-Type", "image/jpeg")
                    .setBody(Buffer().write(portrait))
                else -> MockResponse().setResponseCode(404)
            }
        }
        server.start()
        previousUrl = ApiClient.baseUrl
        ApiClient.baseUrl = server.url("/").toString()
        compose.runOnUiThread { studio = StudioViewModel(app) }
    }

    @After fun tearDown() {
        compose.runOnUiThread { studio.viewModelScope.cancel() }
        server.shutdown()
        ApiClient.baseUrl = previousUrl
    }

    @Test fun guestCanStyleGenerateAndCompareWithMatchingOriginal() {
        compose.setContent { PhotovanceApp(studio) }
        awaitStudio { studio.online }
        capture("login-dark")
        compose.onNodeWithText("Explore as a guest").performScrollTo().performClick()
        compose.onNodeWithText("Start with your photo.").assertIsDisplayed()
        capture("photo-dark")
        compose.onNodeWithText("Just exploring? Try a sample").performScrollTo().performClick()
        compose.onNodeWithText("Continue to styling").performClick()
        compose.onNodeWithText("Find your next look.").assertIsDisplayed()
        capture("style-dark")
        compose.onNodeWithText("Outfit Swap").performClick()
        compose.onNodeWithText("Evening Edit").performClick()
        compose.runOnIdle { assertEquals("in_place", studio.mode) }
        compose.onNodeWithText("Generate photo").performClick()
        awaitStudio { studio.result != null }
        compose.onNodeWithText("Your new look.").assertIsDisplayed()
        val sent = JsonParser.parseString(requestBody.get()).asJsonObject
        assertEquals("test-photo", sent["image_id"].asString)
        assertEquals("in_place", sent["mode"].asString)
        assertEquals("portrait", sent["aspect_ratio"].asString)
        assertTrue(sent["keep_face"].asBoolean)
        assertEquals(server.url("/inputs/original.jpg").toString(), studio.result?.beforeUrl)
        // These result screenshots use local fixture images, not a live AI generation.
        capture("result-dark-fixture")
        compose.onNodeWithText("View original").performScrollTo().performClick()
        compose.mainClock.advanceTimeBy(250)
        compose.onNodeWithContentDescription("Before and after photo comparison")
            .assert(SemanticsMatcher.expectValue(androidx.compose.ui.semantics.SemanticsProperties.StateDescription,
                "100 percent original"))
    }

    @Test fun lightThemeAndSignupRemainAvailableWithoutPretendingToAuthenticate() {
        RuntimeEnvironment.getApplication().getSharedPreferences("studio_design", Context.MODE_PRIVATE)
            .edit().putString("appearance", "Light").commit()
        compose.setContent { PhotovanceApp(studio) }
        awaitStudio { studio.online }
        capture("login-light")
        compose.onNodeWithText("New here? Create an account").performScrollTo().performClick()
        compose.onNodeWithText("Make room for a new you.").assertIsDisplayed()
        capture("signup-light")
        compose.onNodeWithText("Create account", substring = false).performScrollTo().performClick()
        compose.onNodeWithText("Enter a valid email address").assertExists()
        compose.onNodeWithText("Continue with Google").performScrollTo().performClick()
        compose.onNodeWithText("Google sign-in is coming soon").assertIsDisplayed()
        compose.onNode(hasText("Explore as a guest") and hasAnyAncestor(isDialog())).performClick()
        compose.onNodeWithText("Start with your photo.").assertIsDisplayed()
        capture("photo-light")
        compose.onNodeWithText("Just exploring? Try a sample").performScrollTo().performClick()
        compose.onNodeWithText("Continue to styling").performClick()
        capture("style-light")
    }

    @Test fun generationFailureKeepsPhotoPromptAndAllowsRetry() {
        generationFails.set(true)
        compose.runOnUiThread { studio.selectSample(); studio.goToStep(1) }
        compose.setContent {
            PhotovanceTheme(AppAppearance.Dark) {
                Surface(color = MaterialTheme.colorScheme.background) { HomeScreen(studio, onSettings = {}) }
            }
        }
        val prompt = studio.prompt
        compose.onNodeWithText("Generate photo").performClick()
        awaitStudio { studio.error != null && !studio.isGenerating }
        compose.runOnIdle {
            assertEquals(1, studio.step)
            assertTrue(studio.hasPhoto)
            assertEquals(prompt, studio.prompt)
            assertNull(studio.result)
        }
        compose.onNodeWithText("Please try another description.").performScrollTo().assertIsDisplayed()
        generationFails.set(false)
        compose.onNodeWithText("Generate photo").performClick()
        awaitStudio { studio.result != null }
        compose.runOnIdle { assertEquals(2, studio.step) }
    }

    @Test fun appearanceChoiceIsAppliedAndSaved() {
        compose.setContent { PhotovanceApp(studio) }
        compose.onNodeWithContentDescription("Settings and appearance").performClick()
        compose.onNodeWithText("Light", substring = false).performClick()
        compose.onNodeWithText("Light", substring = false).assertIsSelected()
        assertEquals("Light", RuntimeEnvironment.getApplication()
            .getSharedPreferences("studio_design", Context.MODE_PRIVATE).getString("appearance", null))
        compose.onNodeWithText("Done", substring = false).performScrollTo().performClick()
        compose.onNodeWithText("Welcome back.").assertExists()
    }

    @Test
    @Config(qualifiers = "w360dp-h800dp-xhdpi")
    fun smallScreenWithLargeTextKeepsPhotoAndStyleActionsReachable() {
        compose.setContent {
            CompositionLocalProvider(LocalDensity provides Density(LocalDensity.current.density, 1.4f)) {
                PhotovanceTheme(AppAppearance.Light) {
                    Surface { HomeScreen(studio, onSettings = {}) }
                }
            }
        }
        compose.onNodeWithText("Choose a photo").assertIsDisplayed()
        compose.onNodeWithText("Just exploring? Try a sample").performScrollTo().performClick()
        compose.onNodeWithText("Continue to styling").assertIsDisplayed().performClick()
        compose.onNodeWithText("Generate photo").assertIsDisplayed()
        compose.onNodeWithText("Describe your look").performScrollTo().assertIsDisplayed()
        capture("style-light-large-text")
    }

    private fun capture(name: String) {
        compose.waitForIdle()
        val file = File("build/design-preview/$name.png").apply { parentFile?.mkdirs() }
        compose.onRoot().captureRoboImage(file.absolutePath)
    }

    private fun awaitStudio(condition: () -> Boolean) {
        compose.waitUntil(15_000) { shadowOf(Looper.getMainLooper()).idle(); condition() }
        compose.waitForIdle()
    }

    private fun json(body: String) = MockResponse().setHeader("Content-Type", "application/json").setBody(body)
}

