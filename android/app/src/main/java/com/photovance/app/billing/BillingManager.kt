package com.photovance.app.billing

import android.app.Activity
import android.content.Context
import com.android.billingclient.api.AcknowledgePurchaseParams
import com.android.billingclient.api.BillingClient
import com.android.billingclient.api.BillingClientStateListener
import com.android.billingclient.api.BillingFlowParams
import com.android.billingclient.api.PendingPurchasesParams
import com.android.billingclient.api.ProductDetails
import com.android.billingclient.api.Purchase
import com.android.billingclient.api.PurchasesUpdatedListener
import com.android.billingclient.api.QueryProductDetailsParams
import com.android.billingclient.api.QueryPurchasesParams
import com.google.firebase.functions.FirebaseFunctions
import com.google.firebase.auth.FirebaseAuth
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.tasks.await
import java.security.MessageDigest
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlin.coroutines.resume

enum class VipPlan(val id: String, val title: String, val price: String, val billingPeriod: String, val badge: String?) {
    WEEKLY("vip_weekly", "Weekly Access", "$4.99", "/ week · 3-day trial when eligible", null),
    YEARLY("vip_yearly", "Annual VIP Studio", "$29.99", "/ year (~$0.58/wk)", "SAVE 88% · BEST VALUE")
}

class BillingManager(context: Context) : BillingClientStateListener, PurchasesUpdatedListener {
    private val appContext = context.applicationContext
    private val prefs = appContext.getSharedPreferences("photovance_billing", Context.MODE_PRIVATE)
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val _isVip = MutableStateFlow(false)
    val isVip: StateFlow<Boolean> = _isVip.asStateFlow()

    private val _remainingCredits = MutableStateFlow(prefs.getInt(KEY_CREDITS, DEFAULT_FREE_CREDITS))
    val remainingCredits: StateFlow<Int> = _remainingCredits.asStateFlow()

    private val _productDetails = MutableStateFlow<Map<String, ProductDetails>>(emptyMap())
    val productDetails: StateFlow<Map<String, ProductDetails>> = _productDetails.asStateFlow()

    private val _message = MutableStateFlow<String?>(null)
    val message: StateFlow<String?> = _message.asStateFlow()

    private var purchaseResult: ((Boolean, String?) -> Unit)? = null
    private val billingClient = BillingClient.newBuilder(appContext)
        .setListener(this)
        .enablePendingPurchases(
            PendingPurchasesParams.newBuilder()
                .enableOneTimeProducts()
                .build()
        )
        .build()

    init {
        withConnection {
            queryProductDetails()
            queryOwnedSubscriptions()
        }
    }

    override fun onBillingSetupFinished(result: com.android.billingclient.api.BillingResult) {
        if (result.responseCode == BillingClient.BillingResponseCode.OK) {
            queryProductDetails()
            queryOwnedSubscriptions()
        } else {
            _message.value = "Google Play Billing is unavailable: ${result.debugMessage}"
        }
    }

    override fun onBillingServiceDisconnected() {
        _message.value = "Google Play disconnected. Reopen subscriptions to retry."
    }

    override fun onPurchasesUpdated(
        result: com.android.billingclient.api.BillingResult,
        purchases: MutableList<Purchase>?
    ) {
        when (result.responseCode) {
            BillingClient.BillingResponseCode.OK -> {
                if (purchases.isNullOrEmpty()) {
                    finishPurchase(false, "Google Play returned no purchase to verify.")
                    return
                }
                scope.launch {
                    for (purchase in purchases) {
                        processPurchase(purchase)
                    }
                }
            }
            BillingClient.BillingResponseCode.USER_CANCELED ->
                finishPurchase(false, "Purchase canceled.")
            else ->
                finishPurchase(false, "Purchase failed: ${result.debugMessage}")
        }
    }

    fun canGenerate(): Boolean = _isVip.value || _remainingCredits.value > 0

    fun consumeCreditIfFree(): Boolean {
        if (_isVip.value) return true
        val current = _remainingCredits.value
        if (current <= 0) return false
        val next = current - 1
        _remainingCredits.value = next
        prefs.edit().putInt(KEY_CREDITS, next).apply()
        return true
    }

    fun subscribe(activity: Activity, plan: VipPlan, onResult: (Boolean, String?) -> Unit) {
        purchaseResult = onResult
        val uid = FirebaseAuth.getInstance().currentUser?.uid
        if (uid == null) {
            finishPurchase(false, "Sign in before starting a subscription purchase.")
            return
        }
        val accountId = MessageDigest.getInstance("SHA-256")
            .digest(uid.toByteArray(Charsets.UTF_8))
            .joinToString("") { byte -> "%02x".format(byte.toInt() and 0xff) }
        withConnection(onFailure = { finishPurchase(false, it) }) {
            val query = QueryProductDetailsParams.newBuilder()
                .setProductList(
                    listOf(
                        QueryProductDetailsParams.Product.newBuilder()
                            .setProductId(plan.id)
                            .setProductType(BillingClient.ProductType.SUBS)
                            .build()
                    )
                )
                .build()
            billingClient.queryProductDetailsAsync(query) { result, details ->
                if (result.responseCode != BillingClient.BillingResponseCode.OK) {
                    finishPurchase(false, "Could not load subscription details: ${result.debugMessage}")
                    return@queryProductDetailsAsync
                }
                val product = details.firstOrNull()
                val offers = product?.subscriptionOfferDetails.orEmpty()
                val offerToken = offers.firstOrNull { offer ->
                    plan == VipPlan.WEEKLY && offer.pricingPhases.pricingPhaseList.any { phase ->
                        phase.priceAmountMicros == 0L && phase.billingPeriod == "P3D"
                    }
                }?.offerToken ?: offers.firstOrNull()?.offerToken
                if (product == null || offerToken == null) {
                    finishPurchase(false, "This subscription is not available in Google Play yet.")
                    return@queryProductDetailsAsync
                }
                _productDetails.value = _productDetails.value + (product.productId to product)
                val flowParams = BillingFlowParams.newBuilder()
                    .setObfuscatedAccountId(accountId)
                    .setProductDetailsParamsList(
                        listOf(
                            BillingFlowParams.ProductDetailsParams.newBuilder()
                                .setProductDetails(product)
                                .setOfferToken(offerToken)
                                .build()
                        )
                    )
                    .build()
                val launchResult = billingClient.launchBillingFlow(activity, flowParams)
                if (launchResult.responseCode != BillingClient.BillingResponseCode.OK) {
                    finishPurchase(false, "Google Play could not start the purchase: ${launchResult.debugMessage}")
                } else {
                    _message.value = "Complete the purchase in Google Play."
                }
            }
        }
    }

    fun restorePurchases(onResult: (Boolean, String?) -> Unit) {
        withConnection(onFailure = { onResult(false, it) }) {
            billingClient.queryPurchasesAsync(
                QueryPurchasesParams.newBuilder()
                    .setProductType(BillingClient.ProductType.SUBS)
                    .build()
            ) { result, purchases ->
                if (result.responseCode != BillingClient.BillingResponseCode.OK) {
                    onResult(false, "Could not restore purchases: ${result.debugMessage}")
                    return@queryPurchasesAsync
                }
                if (purchases.isEmpty()) {
                    _isVip.value = false
                    onResult(false, "No active subscription found.")
                    return@queryPurchasesAsync
                }
                scope.launch {
                    var verified = false
                    purchases.forEach { verified = processPurchase(it, reportResult = false) || verified }
                    _isVip.value = verified
                    val message = if (verified) "Subscription restored."
                    else "No active subscription could be verified."
                    launch(Dispatchers.Main) { onResult(verified, message) }
                }
            }
        }
    }

    fun close() {
        scope.cancel()
        if (billingClient.isReady) billingClient.endConnection()
    }

    private fun queryProductDetails() {
        val products = VipPlan.entries.map { plan ->
            QueryProductDetailsParams.Product.newBuilder()
                .setProductId(plan.id)
                .setProductType(BillingClient.ProductType.SUBS)
                .build()
        }
        billingClient.queryProductDetailsAsync(
            QueryProductDetailsParams.newBuilder().setProductList(products).build()
        ) { result, details ->
            if (result.responseCode == BillingClient.BillingResponseCode.OK) {
                _productDetails.value = details.associateBy { it.productId }
            } else {
                _message.value = "Could not load subscription prices: ${result.debugMessage}"
            }
        }
    }

    private fun queryOwnedSubscriptions() {
        billingClient.queryPurchasesAsync(
            QueryPurchasesParams.newBuilder()
                .setProductType(BillingClient.ProductType.SUBS)
                .build()
        ) { result, purchases ->
            if (result.responseCode != BillingClient.BillingResponseCode.OK) {
                _message.value = "Could not check existing subscriptions: ${result.debugMessage}"
                return@queryPurchasesAsync
            }
            scope.launch {
                var active = false
                purchases.filter { it.purchaseState == Purchase.PurchaseState.PURCHASED }
                    .forEach { active = processPurchase(it, reportResult = false) || active }
                _isVip.value = active
            }
        }
    }

    private suspend fun processPurchase(purchase: Purchase, reportResult: Boolean = true): Boolean {
        if (purchase.purchaseState != Purchase.PurchaseState.PURCHASED) {
            if (reportResult) finishPurchase(false, "The purchase is still pending.")
            return false
        }
        val productId = purchase.products.firstOrNull { id -> VipPlan.entries.any { it.id == id } }
            ?: return false
        return try {
            val response = FirebaseFunctions.getInstance()
                .getHttpsCallable("verifyPlayPurchase")
                .call(mapOf("productId" to productId, "purchaseToken" to purchase.purchaseToken))
                .await()
            val active = (response.data as? Map<*, *>)?.get("active") as? Boolean ?: false
            if (!active) {
                if (reportResult) finishPurchase(false, "Google Play could not confirm an active subscription.")
                return false
            }
            if (!purchase.isAcknowledged) {
                val ack = suspendCancellableCoroutine { continuation ->
                    billingClient.acknowledgePurchase(
                        AcknowledgePurchaseParams.newBuilder()
                            .setPurchaseToken(purchase.purchaseToken)
                            .build()
                    ) { result ->
                        if (continuation.isActive) continuation.resume(result)
                    }
                }
                if (ack.responseCode != BillingClient.BillingResponseCode.OK) {
                    if (reportResult) finishPurchase(false, "Purchase verified, but acknowledgement failed: ${ack.debugMessage}")
                    return false
                }
            }
            _isVip.value = true
            if (reportResult) finishPurchase(true, "Subscription verified and active.")
            true
        } catch (error: Exception) {
            if (reportResult) finishPurchase(false, error.message ?: "Could not verify the purchase.")
            false
        }
    }

    private fun withConnection(
        onFailure: (String) -> Unit = { _message.value = it },
        action: () -> Unit
    ) {
        if (billingClient.isReady) {
            action()
        } else {
            billingClient.startConnection(object : BillingClientStateListener {
                override fun onBillingSetupFinished(result: com.android.billingclient.api.BillingResult) {
                    if (result.responseCode == BillingClient.BillingResponseCode.OK) action()
                    else onFailure("Google Play Billing is unavailable: ${result.debugMessage}")
                }

                override fun onBillingServiceDisconnected() {
                    onFailure("Google Play disconnected. Try again.")
                }
            })
        }
    }

    private fun finishPurchase(success: Boolean, message: String?) {
        _message.value = message
        val callback = purchaseResult
        purchaseResult = null
        if (callback != null) {
            scope.launch(Dispatchers.Main) { callback(success, message) }
        }
    }

    companion object {
        private const val KEY_CREDITS = "remaining_credits"
        const val DEFAULT_FREE_CREDITS = 3
    }
}
