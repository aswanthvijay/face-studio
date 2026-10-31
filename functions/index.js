const crypto = require("node:crypto");
const { google } = require("googleapis");
const { getAuth } = require("firebase-admin/auth");
const { initializeApp } = require("firebase-admin/app");
const { getFirestore, FieldValue } = require("firebase-admin/firestore");
const { HttpsError, onCall } = require("firebase-functions/v2/https");
const { isAllowedProduct, isActiveSubscription } = require("./entitlement");

initializeApp();

const ANDROID_PACKAGE_NAME = "com.photovance.app";
const publisher = google.androidpublisher({
  version: "v3",
  auth: new google.auth.GoogleAuth({
    scopes: ["https://www.googleapis.com/auth/androidpublisher"],
  }),
});

exports.verifyPlayPurchase = onCall(async (request) => {
  if (!request.auth) {
    throw new HttpsError("unauthenticated", "Sign in before verifying a subscription.");
  }

  const productId = request.data?.productId;
  const purchaseToken = request.data?.purchaseToken;
  if (!isAllowedProduct(productId) ||
      typeof purchaseToken !== "string" ||
      purchaseToken.length < 20 ||
      purchaseToken.length > 4096) {
    throw new HttpsError("invalid-argument", "A valid subscription product and purchase token are required.");
  }

  try {
    const response = await publisher.purchases.subscriptionsv2.get({
      packageName: ANDROID_PACKAGE_NAME,
      token: purchaseToken,
    });
    const subscription = response.data;
    const accountId = crypto.createHash("sha256").update(request.auth.uid).digest("hex");
    const active = isActiveSubscription(subscription, productId, accountId);
    const tokenHash = crypto.createHash("sha256").update(purchaseToken).digest("hex");
    await getFirestore()
      .collection("users")
      .doc(request.auth.uid)
      .collection("subscriptions")
      .doc(tokenHash)
      .set({
        productId,
        active,
        expiryTime: subscription.lineItems?.find((item) => item.productId === productId)?.expiryTime || null,
        updatedAt: FieldValue.serverTimestamp(),
      }, { merge: true });

    return { active, productId };
  } catch (error) {
    console.error("Play subscription verification failed", {
      uid: request.auth.uid,
      productId,
      status: error?.response?.status,
    });
    throw new HttpsError("internal", "The subscription could not be verified with Google Play.");
  }
});

exports.deleteAccountData = onCall(async (request) => {
  if (!request.auth) {
    throw new HttpsError("unauthenticated", "Sign in before deleting account data.");
  }
  const authTime = Number(request.auth.token.auth_time || 0);
  if (Date.now() / 1000 - authTime > 5 * 60) {
    throw new HttpsError("failed-precondition", "Sign in again before deleting your account.");
  }
  try {
    await getFirestore().recursiveDelete(
      getFirestore().collection("users").doc(request.auth.uid)
    );
    await getAuth().deleteUser(request.auth.uid);
    return { deleted: true };
  } catch (error) {
    console.error("Account data deletion failed", { uid: request.auth.uid });
    throw new HttpsError("internal", "Account data could not be deleted. Try again later.");
  }
});
