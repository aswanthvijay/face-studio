const test = require("node:test");
const assert = require("node:assert/strict");
const { isAllowedProduct, isActiveSubscription } = require("./entitlement");

test("only the configured Android subscription ids are eligible", () => {
  assert.equal(isAllowedProduct("vip_weekly"), true);
  assert.equal(isAllowedProduct("vip_yearly"), true);
  assert.equal(isAllowedProduct("vip_unconfigured"), false);
});

test("active subscriptions must match a supported product and an unexpired line item", () => {
  const now = Date.now();
  assert.equal(isActiveSubscription({
    subscriptionState: "SUBSCRIPTION_STATE_ACTIVE",
    externalAccountIdentifiers: { obfuscatedExternalAccountId: "account-a" },
    lineItems: [{ productId: "vip_weekly", expiryTime: new Date(now + 60_000).toISOString() }],
  }, "vip_weekly", "account-a", now), true);
  assert.equal(isActiveSubscription({
    subscriptionState: "SUBSCRIPTION_STATE_ACTIVE",
    externalAccountIdentifiers: { obfuscatedExternalAccountId: "account-a" },
    lineItems: [{ productId: "vip_yearly", expiryTime: new Date(now + 60_000).toISOString() }],
  }, "vip_weekly", "account-a", now), false);
  assert.equal(isActiveSubscription({
    subscriptionState: "SUBSCRIPTION_STATE_EXPIRED",
    externalAccountIdentifiers: { obfuscatedExternalAccountId: "account-a" },
    lineItems: [{ productId: "vip_weekly", expiryTime: new Date(now + 60_000).toISOString() }],
  }, "vip_weekly", "account-a", now), false);
  assert.equal(isActiveSubscription({
    subscriptionState: "SUBSCRIPTION_STATE_CANCELED",
    externalAccountIdentifiers: { obfuscatedExternalAccountId: "account-a" },
    lineItems: [{ productId: "vip_weekly", expiryTime: new Date(now - 1).toISOString() }],
  }, "vip_weekly", "account-a", now), false);
  assert.equal(isActiveSubscription({
    subscriptionState: "SUBSCRIPTION_STATE_ACTIVE",
    externalAccountIdentifiers: { obfuscatedExternalAccountId: "another-account" },
    lineItems: [{ productId: "vip_weekly", expiryTime: new Date(now + 60_000).toISOString() }],
  }, "vip_weekly", "account-a", now), false);
  assert.equal(isActiveSubscription({
    subscriptionState: "SUBSCRIPTION_STATE_ACTIVE",
    lineItems: [{ productId: "vip_weekly", expiryTime: new Date(now + 60_000).toISOString() }],
  }, "vip_weekly", "account-a", now), false);
});
