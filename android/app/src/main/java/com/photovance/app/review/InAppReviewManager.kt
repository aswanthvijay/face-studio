package com.photovance.app.review

import android.app.Activity
import android.content.Context
import com.google.android.play.core.review.ReviewManagerFactory

class InAppReviewManager(private val context: Context) {
    private val reviewManager = ReviewManagerFactory.create(context)
    private val prefs = context.getSharedPreferences("photovance_reviews", Context.MODE_PRIVATE)

    fun recordGenerationSuccess(activity: Activity) {
        val count = prefs.getInt(KEY_SUCCESSFUL_GENERATIONS, 0) + 1
        prefs.edit().putInt(KEY_SUCCESSFUL_GENERATIONS, count).apply()

        // Prompt for review on the 2nd and 6th generation
        if (count == 2 || count == 6) {
            promptReview(activity)
        }
    }

    private fun promptReview(activity: Activity) {
        val request = reviewManager.requestReviewFlow()
        request.addOnCompleteListener { task ->
            if (task.isSuccessful) {
                val reviewInfo = task.result
                reviewManager.launchReviewFlow(activity, reviewInfo)
            }
        }
    }

    companion object {
        private const val KEY_SUCCESSFUL_GENERATIONS = "successful_generations_count"
    }
}
