package com.photovance.app.media

import android.content.ClipData
import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.Typeface
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.util.Base64
import androidx.core.content.FileProvider
import androidx.exifinterface.media.ExifInterface
import com.photovance.app.network.ApiClient
import com.photovance.app.network.UploadRequest
import com.photovance.app.network.UploadResponse
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.IOException

/** Decode at a bounded size, including the orientation written by phone cameras. */
suspend fun uploadPortrait(context: Context, uri: Uri): UploadResponse = withContext(Dispatchers.IO) {
    val resolver = context.contentResolver
    val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
    resolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, bounds) }
    require(bounds.outWidth > 0 && bounds.outHeight > 0) { "This photo couldn't be read. Choose another image." }
    var sample = 1
    while (maxOf(bounds.outWidth, bounds.outHeight) / sample > 3072) sample *= 2
    val decoded = resolver.openInputStream(uri)?.use {
        BitmapFactory.decodeStream(it, null, BitmapFactory.Options().apply { inSampleSize = sample })
    } ?: throw IOException("This photo couldn't be opened.")
    val orientation = resolver.openInputStream(uri)?.use {
        ExifInterface(it).getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
    } ?: ExifInterface.ORIENTATION_NORMAL
    val transform = Matrix().apply {
        when (orientation) {
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> setScale(-1f, 1f)
            ExifInterface.ORIENTATION_ROTATE_180 -> setRotate(180f)
            ExifInterface.ORIENTATION_FLIP_VERTICAL -> setScale(1f, -1f)
            ExifInterface.ORIENTATION_TRANSPOSE -> { setRotate(90f); postScale(-1f, 1f) }
            ExifInterface.ORIENTATION_ROTATE_90 -> setRotate(90f)
            ExifInterface.ORIENTATION_TRANSVERSE -> { setRotate(-90f); postScale(-1f, 1f) }
            ExifInterface.ORIENTATION_ROTATE_270 -> setRotate(-90f)
        }
    }
    val oriented = Bitmap.createBitmap(decoded, 0, 0, decoded.width, decoded.height, transform, true)
    val scale = minOf(1f, 1536f / maxOf(oriented.width, oriented.height))
    val resized = Bitmap.createScaledBitmap(oriented, (oriented.width * scale).toInt().coerceAtLeast(1),
        (oriented.height * scale).toInt().coerceAtLeast(1), true)
    val bytes = try {
        ByteArrayOutputStream().use {
            check(resized.compress(Bitmap.CompressFormat.JPEG, 92, it)) { "Couldn't prepare this photo." }
            it.toByteArray()
        }
    } finally {
        if (resized !== oriented) resized.recycle()
        if (oriented !== decoded) oriented.recycle()
        decoded.recycle()
    }
    ApiClient.api.uploadImage(UploadRequest("data:image/jpeg;base64,${Base64.encodeToString(bytes, Base64.NO_WRAP)}"))
}

private val downloadClient = OkHttpClient()

private fun downloadImage(url: String): ByteArray = downloadClient.newCall(Request.Builder().url(url).build())
    .execute().use { response ->
        if (!response.isSuccessful) throw IOException("The photo couldn't be downloaded. Try again.")
        response.body?.bytes() ?: throw IOException("The photo is empty.")
    }
fun applyWatermarkIfFree(bitmap: Bitmap, isVip: Boolean): Bitmap {
    if (isVip) return bitmap
    val mutable = bitmap.copy(Bitmap.Config.ARGB_8888, true)
    val canvas = Canvas(mutable)
    val text = "✦ PHOTOVANCE STUDIO"
    val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = android.graphics.Color.WHITE
        textSize = (bitmap.width * 0.028f).coerceIn(24f, 48f)
        typeface = Typeface.create(Typeface.SANS_SERIF, Typeface.BOLD)
        setShadowLayer(8f, 2f, 2f, android.graphics.Color.BLACK)
        alpha = 210
    }
    val bounds = Rect()
    paint.getTextBounds(text, 0, text.length, bounds)
    val margin = (bitmap.width * 0.04f).coerceAtLeast(24f)
    val x = bitmap.width - bounds.width() - margin
    val y = bitmap.height - margin
    canvas.drawText(text, x, y, paint)
    return mutable
}

fun createInstagramStoryBitmap(source: Bitmap, isVip: Boolean): Bitmap {
    val targetWidth = 1080
    val targetHeight = 1920
    val storyBitmap = Bitmap.createBitmap(targetWidth, targetHeight, Bitmap.Config.ARGB_8888)
    val canvas = Canvas(storyBitmap)

    val bgPaint = Paint().apply {
        shader = android.graphics.LinearGradient(
            0f, 0f, 0f, targetHeight.toFloat(),
            intArrayOf(android.graphics.Color.parseColor("#0F0E13"), android.graphics.Color.parseColor("#1C1628")),
            null, android.graphics.Shader.TileMode.CLAMP
        )
    }
    canvas.drawRect(0f, 0f, targetWidth.toFloat(), targetHeight.toFloat(), bgPaint)

    val scale = minOf((targetWidth - 96).toFloat() / source.width, (targetHeight - 260).toFloat() / source.height)
    val scaledW = (source.width * scale).toInt()
    val scaledH = (source.height * scale).toInt()
    val left = ((targetWidth - scaledW) / 2).toFloat()
    val top = ((targetHeight - scaledH) / 2).toFloat()

    val scaled = Bitmap.createScaledBitmap(source, scaledW, scaledH, true)
    canvas.drawBitmap(scaled, left, top, null)
    if (scaled !== source) scaled.recycle()

    val brandPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = android.graphics.Color.parseColor("#FFB800")
        textSize = 32f
        typeface = Typeface.create(Typeface.SANS_SERIF, Typeface.BOLD)
        textAlign = Paint.Align.CENTER
        letterSpacing = 0.15f
    }
    canvas.drawText("PHOTOVANCE AI STUDIO", (targetWidth / 2).toFloat(), 110f, brandPaint)

    return applyWatermarkIfFree(storyBitmap, isVip)
}

suspend fun savePortrait(context: Context, url: String, isVip: Boolean = false) = withContext(Dispatchers.IO) {
    val bytes = downloadImage(url)
    val rawBitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
        ?: throw IOException("The result isn't a readable image.")
    val bitmap = applyWatermarkIfFree(rawBitmap, isVip)
    val resolver = context.contentResolver
    val values = ContentValues().apply {
        put(MediaStore.Images.Media.DISPLAY_NAME, "Photovance_${System.currentTimeMillis()}.png")
        put(MediaStore.Images.Media.MIME_TYPE, "image/png")
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            put(MediaStore.Images.Media.RELATIVE_PATH, "${Environment.DIRECTORY_PICTURES}/Photovance")
            put(MediaStore.Images.Media.IS_PENDING, 1)
        }
    }
    var saved: Uri? = null
    try {
        saved = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
            ?: throw IOException("Couldn't create a photo in your gallery.")
        resolver.openOutputStream(saved)?.use {
            check(bitmap.compress(Bitmap.CompressFormat.PNG, 100, it)) { "Couldn't save this photo." }
        } ?: throw IOException("Couldn't write to your gallery.")
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            resolver.update(saved, ContentValues().apply { put(MediaStore.Images.Media.IS_PENDING, 0) }, null, null)
        }
    } catch (error: Exception) {
        saved?.let { resolver.delete(it, null, null) }
        throw error
    } finally {
        if (bitmap !== rawBitmap) bitmap.recycle()
        rawBitmap.recycle()
    }
}

suspend fun sharePortrait(context: Context, url: String, isVip: Boolean = false, asStory: Boolean = false) {
    val uri = withContext(Dispatchers.IO) {
        val bytes = downloadImage(url)
        val rawBitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
            ?: throw IOException("The result isn't a readable image.")
        val bitmap = if (asStory) createInstagramStoryBitmap(rawBitmap, isVip) else applyWatermarkIfFree(rawBitmap, isVip)
        val folder = File(context.cacheDir, "shared_photos").apply { mkdirs() }
        folder.listFiles()?.filter { System.currentTimeMillis() - it.lastModified() > 86_400_000 }?.forEach { it.delete() }
        val file = File(folder, "Photovance_${System.currentTimeMillis()}.png")
        try {
            file.outputStream().use { bitmap.compress(Bitmap.CompressFormat.PNG, 100, it) }
        } finally {
            if (bitmap !== rawBitmap) bitmap.recycle()
            rawBitmap.recycle()
        }
        FileProvider.getUriForFile(context, "${context.packageName}.fileprovider", file)
    }
    val intent = Intent(Intent.ACTION_SEND).apply {
        type = "image/png"
        putExtra(Intent.EXTRA_STREAM, uri)
        clipData = ClipData.newRawUri("Photovance portrait", uri)
        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
    }
    context.startActivity(Intent.createChooser(intent, "Share photo").addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
}
