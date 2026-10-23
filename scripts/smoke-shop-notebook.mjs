import { randomBytes, randomUUID, createHash } from "node:crypto";
import assert from "node:assert/strict";
import { Pool } from "pg";

const url = new URL(process.env.DATABASE_URL || "");
const origin = process.env.WAX_TEST_URL || "http://localhost:3000";
const preview = /^https:\/\/wax-[a-z0-9]+-coolition\.vercel\.app$/.test(origin);
assert.ok(
  origin === "http://localhost:3000" || preview,
  "Use localhost or an exact WAX preview.",
);
const database = preview ? "wax_preview" : "wax_development";
assert.equal(url.pathname, `/${database}`);
assert.equal(url.username, database);
assert.ok(!url.searchParams.has("user"));
if (preview)
  assert.ok(
    process.env.VERCEL_OIDC_TOKEN,
    "Refresh the project preview credential.",
  );
const pool = new Pool({ connectionString: url.href, max: 1 });
const fixtures = Array.from({ length: 2 }, () => {
  const token = randomBytes(32).toString("hex");
  return {
    token,
    digest: createHash("sha256").update(token).digest("hex"),
    email: `wax-shop-notebook-${randomBytes(8).toString("hex")}@example.test`,
    password: randomBytes(24).toString("base64url"),
    cookie: "",
    ip: `notebook-${randomBytes(8).toString("hex")}`,
  };
});
async function request(path, method, body, fixture, requestOrigin = origin) {
  const response = await fetch(origin + path, {
    method,
    redirect: "error",
    headers: {
      Origin: requestOrigin,
      ...(body ? { "Content-Type": "application/json" } : {}),
      ...(fixture ? { "x-real-ip": fixture.ip, Cookie: fixture.cookie } : {}),
      ...(preview
        ? { "x-vercel-trusted-oidc-idp-token": process.env.VERCEL_OIDC_TOKEN }
        : {}),
    },
    ...(body ? { body: JSON.stringify(body) } : {}),
  });
  return { response, result: await response.json() };
}
const path = "/api/shops/notebook";
try {
  assert.equal(
    (await pool.query("SELECT current_user")).rows[0].current_user,
    database,
  );
  assert.equal((await request(path, "GET")).response.status, 401);
  for (const fixture of fixtures) {
    await pool.query(
      "INSERT INTO wax.invitations(token_hash,email,kind,expires_at) VALUES($1,$2,'collector',now()+interval '15 minutes')",
      [fixture.digest, fixture.email],
    );
    const accepted = await request(
      "/api/invitations/accept",
      "POST",
      {
        token: fixture.token,
        email: fixture.email,
        password: fixture.password,
        name: "Shop notebook verification",
      },
      fixture,
    );
    assert.equal(accepted.response.status, 200);
    const signedIn = await request(
      "/api/auth/sign-in/email",
      "POST",
      {
        email: fixture.email,
        password: fixture.password,
      },
      fixture,
    );
    assert.equal(signedIn.response.status, 200);
    fixture.cookie = signedIn.response.headers
      .getSetCookie()
      .map((value) => value.split(";")[0])
      .join("; ");
    assert.ok(fixture.cookie);
  }
  const [alice, bob] = fixtures;
  // Regression: an overnight cache expiry must not erase collection artwork.
  const releaseId = 2825456;
  const artworkPath = `/api/catalog/${releaseId}/artwork`;
  assert.equal((await request(artworkPath, "GET")).response.status, 401);
  const release = await request(
    `/api/catalog/${releaseId}`,
    "GET",
    undefined,
    alice,
  );
  assert.equal(release.response.status, 200);
  assert.ok(release.result.coverUrl);
  const copy = await request(
    "/api/records",
    "POST",
    {
      idempotencyKey: randomUUID(),
      candidate: release.result,
      copy: {
        mediaCondition: "Not graded",
        sleeveCondition: "Not graded",
        purchasePrice: null,
        currency: "USD",
        acquiredAt: null,
        notes: "",
        rating: 0,
        location: "",
        intent: "keep",
        pressingConfirmed: false,
      },
      allowDuplicate: false,
    },
    alice,
  );
  assert.equal(copy.response.status, 200);
  await pool.query(
    "UPDATE wax.catalog_cache SET fetched_at=now()-interval '7 hours' WHERE cache_key=$1",
    [`release:${releaseId}`],
  );
  const library = await request("/api/library", "GET", undefined, alice);
  assert.equal(library.result.records[0].coverUrl, artworkPath);
  assert.equal(
    (await request("/api/library", "GET", undefined, bob)).result.records
      .length,
    0,
  );
  const artwork = await fetch(origin + artworkPath, {
    redirect: "manual",
    headers: {
      Cookie: alice.cookie,
      ...(preview
        ? { "x-vercel-trusted-oidc-idp-token": process.env.VERCEL_OIDC_TOKEN }
        : {}),
    },
  });
  assert.equal(artwork.status, 302);
  assert.equal(artwork.headers.get("cache-control"), "private, no-store");
  const imageUrl = new URL(artwork.headers.get("location"));
  assert.ok(["i.discogs.com", "img.discogs.com"].includes(imageUrl.hostname));
  // Never forward the app's session or preview token to the image CDN.
  const image = await fetch(imageUrl, {
    redirect: "error",
    signal: AbortSignal.timeout(15000),
  });
  assert.equal(image.status, 200);
  assert.ok(image.headers.get("content-type")?.startsWith("image/"));
  await image.body?.cancel();
  assert.ok(
    (
      await pool.query(
        "SELECT fetched_at>now()-interval '5 minutes' AS fresh FROM wax.catalog_cache WHERE cache_key=$1",
        [`release:${releaseId}`],
      )
    ).rows[0].fresh,
  );
  console.log(
    "PASS: overnight artwork expiry refreshes through the authenticated API to a live cover image; collection data stays private.",
  );
  const input = {
    shop: {
      id: "node/12919585334",
      name: "WAX shop notebook test fixture",
      latitude: 40.02,
      longitude: -83.01,
      address: "Test fixture, Columbus",
      website: "https://example.test/",
    },
    rating: 5,
    employees: [
      {
        id: randomUUID(),
        name: "Test Pat",
        role: "Record buyer",
        avatar: "groove",
      },
    ],
    version: null,
  };
  assert.equal(
    (await request(path, "PUT", input, alice, "https://untrusted.test"))
      .response.status,
    403,
  );
  const created = await request(path, "PUT", input, alice);
  assert.equal(created.response.status, 200, JSON.stringify(created.result));
  assert.equal(created.result.version, 1);
  const persisted = await request(path, "GET", undefined, alice);
  assert.equal(persisted.response.status, 200);
  assert.equal(
    persisted.response.headers.get("cache-control"),
    "private, no-store",
  );
  assert.equal(persisted.result.shops[0].rating, 5);
  assert.deepEqual(persisted.result.shops[0].employees, input.employees);
  console.log(
    "PASS: rating, employee name, role and avatar persist through the authenticated API in Neon.",
  );

  assert.deepEqual(
    (await request(path, "GET", undefined, bob)).result.shops,
    [],
  );
  assert.equal(
    (await request(path, "DELETE", { shopId: input.shop.id, version: 1 }, bob))
      .response.status,
    404,
  );
  assert.equal(
    (await request(path, "PUT", { ...input, version: 1 }, bob)).response.status,
    404,
  );
  const other = await request(
    path,
    "PUT",
    { ...input, rating: 2, employees: [] },
    bob,
  );
  assert.equal(other.response.status, 200);
  assert.equal(
    (await request(path, "GET", undefined, alice)).result.shops[0].rating,
    5,
  );
  console.log(
    "PASS: a second collector cannot read, edit, or delete the first collector’s shop notebook.",
  );

  assert.equal((await request(path, "PUT", input, alice)).response.status, 409);
  const editedInput = {
    ...input,
    version: 1,
    rating: null,
    employees: [
      { ...input.employees[0], role: "Shop manager", avatar: "cosmic" },
    ],
  };
  const edited = await request(path, "PUT", editedInput, alice);
  assert.equal(edited.response.status, 200);
  assert.equal(edited.result.version, 2);
  assert.equal(
    (await request(path, "PUT", editedInput, alice)).response.status,
    409,
  );
  const reloaded = (await request(path, "GET", undefined, alice)).result.shops;
  assert.equal(reloaded.length, 1);
  assert.equal(reloaded[0].rating, null);
  assert.equal(reloaded[0].employees[0].role, "Shop manager");
  assert.equal(reloaded[0].employees[0].avatar, "cosmic");
  const removedPerson = await request(
    path,
    "PUT",
    { ...editedInput, version: 2, employees: [] },
    alice,
  );
  assert.equal(removedPerson.response.status, 200);
  assert.deepEqual(removedPerson.result.employees, []);
  assert.equal(
    (
      await request(
        path,
        "DELETE",
        { shopId: input.shop.id, version: 2 },
        alice,
      )
    ).response.status,
    409,
  );
  assert.equal(
    (
      await request(
        path,
        "DELETE",
        { shopId: input.shop.id, version: 3 },
        alice,
      )
    ).response.status,
    200,
  );
  assert.deepEqual(
    (await request(path, "GET", undefined, alice)).result.shops,
    [],
  );
  assert.equal(
    (await request(path, "GET", undefined, bob)).result.shops[0].rating,
    2,
  );
  console.log(
    "PASS: clear ratings, edit/remove people, stale-write protection and shop removal work without duplicate saves.",
  );
} finally {
  for (const fixture of fixtures) {
    await pool.query("DELETE FROM wax.invitations WHERE token_hash=$1", [
      fixture.digest,
    ]);
    await pool.query("DELETE FROM wax.users WHERE email=$1", [fixture.email]);
  }
  await pool.end();
  console.log(
    "Temporary shop-notebook collectors and their private data cleaned up.",
  );
}
