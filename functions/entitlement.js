const SUBSCRIPTION_PRODUCTS = new Set(["vip_weekly", "vip_yearly"]);
const ACTIVE_STATES = new Set([
  "SUBSCRIPTION_STATE_ACTIVE",
  "SUBSCRIPTION_STATE_IN_GRACE_PERIOD",
  "SUBSCRIPTION_STATE_CANCELED",
]);

function isAllowedProduct(productId) {
  return SUBSCRIPTION_PRODUCTS.has(productId);
}

function isActiveSubscription(subscription, productId, accountId, now = Date.now()) {
  if (!ACTIVE_STATES.has(subscription.subscriptionState)) return false;
  if (subscription.externalAccountIdentifiers?.obfuscatedExternalAccountId !== accountId) return false;
  return (subscription.lineItems || []).some((lineItem) =>
    lineItem.productId === productId &&
    Date.parse(lineItem.expiryTime || "") > now
  );
}

module.exports = { isAllowedProduct, isActiveSubscription };
