import { randomBytes, createHash } from "node:crypto";
import assert from "node:assert/strict";
import { Pool } from "pg";

const url = new URL(process.env.DATABASE_URL || "");
const origin = process.env.WAX_TEST_URL || "http://localhost:3000";
const preview = /^https:\/\/wax-[a-z0-9]+-coolition\.vercel\.app$/.test(origin);
if (origin !== "http://localhost:3000" && !preview)
  throw new Error(
    "Shop smoke tests require localhost or an exact protected WAX preview URL.",
  );
const database = preview ? "wax_preview" : "wax_development";
if (
  url.pathname !== `/${database}` ||
  url.username !== database ||
  url.searchParams.has("user")
)
  throw new Error(
    "Shop smoke tests require the matching isolated nonproduction database.",
  );
if (preview && !process.env.VERCEL_OIDC_TOKEN)
  throw new Error(
    "A short-lived project OIDC credential is required for the protected preview.",
  );
const pool = new Pool({ connectionString: url.href, max: 1 });
const token = randomBytes(32).toString("hex");
const digest = createHash("sha256").update(token).digest("hex");
const email = `wax-shop-smoke-${randomBytes(8).toString("hex")}@example.test`;
const password = randomBytes(24).toString("base64url");
let cookie = "";
const clientIp = `shops-smoke-${randomBytes(6).toString("hex")}`;
async function request(path, body, authenticated = true) {
  const response = await fetch(origin + path, {
    method: "POST",
    redirect: "error",
    headers: {
      Origin: origin,
      "Content-Type": "application/json",
      "x-real-ip": clientIp,
      ...(preview
        ? { "x-vercel-trusted-oidc-idp-token": process.env.VERCEL_OIDC_TOKEN }
        : {}),
      ...(authenticated && cookie ? { Cookie: cookie } : {}),
    },
    body: JSON.stringify(body),
  });
  const result = await response.json();
  return { response, result };
}
try {
  assert.equal(
    (await pool.query("SELECT current_user")).rows[0].current_user,
    database,
  );
  const denied = await request(
    "/api/shops/nearby",
    { latitude: 39.95, longitude: -75.16, radiusKm: 10 },
    false,
  );
  assert.equal(denied.response.status, 401);
  await pool.query(
    "INSERT INTO wax.invitations(token_hash,email,kind,expires_at) VALUES($1,$2,'collector',now()+interval '15 minutes')",
    [digest, email],
  );
  const accepted = await request("/api/invitations/accept", {
    token,
    email,
    name: "Shop finder verification",
    password,
  });
  assert.equal(accepted.response.status, 200);
  const signedIn = await request("/api/auth/sign-in/email", {
    email,
    password,
  });
  assert.equal(signedIn.response.status, 200);
  cookie = signedIn.response.headers
    .getSetCookie()
    .map((value) => value.split(";")[0])
    .join("; ");
  assert.ok(cookie);
  console.log(
    "PASS: invited fixture signs in; anonymous shop requests are denied.",
  );

  const places = await request("/api/shops/places", {
    query: "Philadelphia, Pennsylvania",
  });
  assert.equal(places.response.status, 200, JSON.stringify(places.result));
  assert.ok(
    places.result.places.some((place) => /Philadelphia/i.test(place.label)),
  );
  console.log("PASS: live city lookup returns reviewable place candidates.");

  const input = { latitude: 39.951234, longitude: -75.161234, radiusKm: 10 };
  const nearby = await request("/api/shops/nearby", input);
  assert.equal(nearby.response.status, 200, JSON.stringify(nearby.result));
  assert.deepEqual(nearby.result.center, {
    latitude: 39.95,
    longitude: -75.16,
  });
  assert.ok(nearby.result.shops.length > 0);
  for (const [index, shop] of nearby.result.shops.entries()) {
    assert.ok(
      shop.name &&
        /^https:\/\/www\.openstreetmap\.org\/(node|way|relation)\/\d+$/.test(
          shop.osmUrl,
        ),
    );
    assert.ok(shop.distanceKm <= 10);
    if (index)
      assert.ok(shop.distanceKm >= nearby.result.shops[index - 1].distanceKm);
  }
  console.log(
    `PASS: ${nearby.result.shops.length} live record-shop listings, rounded search center, nearest-first distances.`,
  );
  const repeated = await request("/api/shops/nearby", input);
  assert.equal(repeated.response.status, 200);
  assert.equal(repeated.result.fetchedAt, nearby.result.fetchedAt);
  assert.deepEqual(repeated.result.shops, nearby.result.shops);
  console.log("PASS: repeated searches reuse the shared provider cache.");

  const columbus = await request("/api/shops/nearby", {
    latitude: 40.02,
    longitude: -83.04,
    radiusKm: 25,
  });
  assert.equal(columbus.response.status, 200, JSON.stringify(columbus.result));
  const names = columbus.result.shops.map((shop) => shop.name);
  for (const name of [
    "Lost Weekend Records",
    "Elizabeth Records",
    "RPM",
    "Used Kids Records",
    "Magnolia Thunderpussy",
  ])
    assert.ok(names.includes(name), `Known record seller retained: ${name}`);
  for (const name of [
    "Columbus Percussion",
    "Sam Ash Music",
    "Stanton's Sheet Music",
    "German Village Music Haus",
  ])
    assert.ok(
      !names.includes(name),
      `Unconfirmed vinyl seller excluded: ${name}`,
    );
  console.log(
    `PASS: ${names.length} Columbus record shops; known vinyl sellers retained and broad music-store false positives excluded.`,
  );
  assert.ok(
    names.includes("Spoonful Records"),
    "Missing-map retailer is included from checked sources",
  );
  assert.equal(columbus.result.discovery.mapAvailable, true);
  let previous = columbus.result;
  for (const radiusKm of [50, 100]) {
    const expanded = await request("/api/shops/nearby", {
      latitude: 40.02,
      longitude: -83.04,
      radiusKm,
    });
    assert.equal(
      expanded.response.status,
      200,
      JSON.stringify(expanded.result),
    );
    assert.equal(expanded.result.discovery.mapAvailable, true);
    assert.ok(
      expanded.result.shops.length > previous.shops.length,
      `Radius ${radiusKm}km discovers additional real shops`,
    );
    assert.ok(
      expanded.result.shops.some((shop) => shop.distanceKm > previous.radiusKm),
      "New results extend beyond the previous search circle",
    );
    assert.ok(
      expanded.result.shops.every((shop) => shop.distanceKm <= radiusKm),
    );
    const expected =
      radiusKm === 50 ? "Pat's Endangered Species" : "Legend Records";
    assert.ok(
      expanded.result.shops.some((shop) => shop.name === expected),
      `Retains sourced regional seller ${expected}`,
    );
    console.log(
      `PASS: ${radiusKm}km search expands to ${expanded.result.shops.length} live results, including ${expected}.`,
    );
    previous = expanded.result;
  }
} finally {
  await pool.query("DELETE FROM wax.invitations WHERE token_hash=$1", [digest]);
  await pool.query("DELETE FROM wax.users WHERE email=$1", [email]);
  await pool.end();
  console.log("Temporary shop-finder collector cleaned up.");
}
