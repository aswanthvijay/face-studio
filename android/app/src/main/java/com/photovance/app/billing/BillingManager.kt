package com.photovance.app.billing

import android.content.Context
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

enum class VipPlan(val id: String, val title: String, val price: String, val billingPeriod: String, val badge: String?) {
    WEEKLY("vip_weekly", "Weekly Access", "$4.99", "/ week (3-day free trial)", null),
    YEARLY("vip_yearly", "Annual VIP Studio", "$29.99", "/ year (~$0.58/wk)", "SAVE 88% · BEST VALUE")
}

class BillingManager(context: Context) {
    private val prefs = context.getSharedPreferences("photovance_billing", Context.MODE_PRIVATE)

    private val _isVip = MutableStateFlow(prefs.getBoolean(KEY_IS_VIP, false))
    val isVip: StateFlow<Boolean> = _isVip.asStateFlow()

    private val _remainingCredits = MutableStateFlow(prefs.getInt(KEY_CREDITS, DEFAULT_FREE_CREDITS))
    val remainingCredits: StateFlow<Int> = _remainingCredits.asStateFlow()

    fun canGenerate(): Boolean {
        return _isVip.value || _remainingCredits.value > 0
    }

    fun consumeCreditIfFree(): Boolean {
        if (_isVip.value) return true
        val current = _remainingCredits.value
        if (current <= 0) return false
        val next = current - 1
        _remainingCredits.value = next
        prefs.edit().putInt(KEY_CREDITS, next).apply()
        return true
    }

    fun subscribe(plan: VipPlan) {
        _isVip.value = true
        prefs.edit()
            .putBoolean(KEY_IS_VIP, true)
            .putString(KEY_ACTIVE_PLAN, plan.id)
            .apply()
    }

    fun restorePurchases(): Boolean {
        // When StoreKit or Google Play Billing restores an active subscription
        val hasActive = prefs.getBoolean(KEY_IS_VIP, false)
        if (hasActive) {
            _isVip.value = true
        }
        return hasActive
    }

    fun resetForTesting() {
        _isVip.value = false
        _remainingCredits.value = DEFAULT_FREE_CREDITS
        prefs.edit().clear().apply()
    }

    companion object {
        private const val KEY_IS_VIP = "is_vip"
        private const val KEY_CREDITS = "remaining_credits"
        private const val KEY_ACTIVE_PLAN = "active_plan"
        const val DEFAULT_FREE_CREDITS = 3
    }
}
