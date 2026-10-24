import assert from "node:assert/strict";
import { createHash, randomBytes, randomUUID } from "node:crypto";
import { Pool } from "pg";

// This probe creates only disposable collectors in the isolated test database.
const url = new URL(process.env.DATABASE_URL || "");
const origin = process.env.WAX_TEST_URL || "http://localhost:3000";
const preview = /^https:\/\/wax-[a-z0-9]+-coolition\.vercel\.app$/.test(origin);
assert.ok(origin === "http://localhost:3000" || preview);
const database = preview ? "wax_preview" : "wax_development";
assert.equal(url.pathname, `/${database}`);
assert.equal(url.username, database);
assert.ok(!url.searchParams.has("user"));
if (preview) assert.ok(process.env.VERCEL_OIDC_TOKEN);
const pool = new Pool({ connectionString: url.href, max: 1 });
const fixtures = Array.from({ length: 2 }, () => {
  const token = randomBytes(32).toString("hex");
  return {
    token,
    digest: createHash("sha256").update(token).digest("hex"),
    email: `wax-catalog-${randomBytes(8).toString("hex")}@example.test`,
    password: randomBytes(24).toString("base64url"),
    cookie: "",
    ip: `catalog-${randomBytes(8).toString("hex")}`,
  };
});
async function request(path, method = "GET", body, fixture) {
  const response = await fetch(origin + path, {
    method,
    redirect: "error",
    headers: {
      Origin: origin,
      ...(body ? { "Content-Type": "application/json" } : {}),
      ...(fixture ? { Cookie: fixture.cookie, "x-real-ip": fixture.ip } : {}),
      ...(preview
        ? { "x-vercel-trusted-oidc-idp-token": process.env.VERCEL_OIDC_TOKEN }
        : {}),
    },
    ...(body ? { body: JSON.stringify(body) } : {}),
  });
  return { response, result: await response.json() };
}
try {
  assert.equal(
    (await pool.query("SELECT current_user")).rows[0].current_user,
    database,
  );
  for (const path of [
    "/api/catalog/albums?query=beatles%20white%20album",
    "/api/catalog/albums/46402/versions?country=US",
  ])
    assert.equal((await request(path)).response.status, 401);

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
        name: "Catalog discovery verification",
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
  const albums = await request(
    "/api/catalog/albums?query=beatles%20white%20album",
    "GET",
    undefined,
    alice,
  );
  assert.equal(albums.response.status, 200);
  assert.equal(
    albums.response.headers.get("cache-control"),
    "private, no-store",
  );
  assert.equal(albums.result.albums[0].masterId, 46402);
  assert.equal(albums.result.albums[0].originalYear, 1968);
  assert.match(albums.result.albums[0].artist, /Beatles/i);

  const base = "/api/catalog/albums/46402/versions?country=US";
  const first = await request(base, "GET", undefined, alice);
  assert.equal(first.response.status, 200);
  assert.ok(first.result.pagination.hasMore);
  assert.ok(
    first.result.candidates.length > 0 && first.result.candidates.length <= 20,
  );
  assert.ok(
    first.result.candidates.every(
      (item) => item.country === "US" && /Vinyl/i.test(item.format),
    ),
  );
  const second = await request(base + "&page=2", "GET", undefined, alice);
  assert.equal(second.response.status, 200);
  assert.equal(second.result.pagination.page, 2);
  assert.notDeepEqual(
    second.result.candidates.map((item) => item.discogsId),
    first.result.candidates.map((item) => item.discogsId),
  );
  const filtered = await request(
    base + "&year=1968&label=Apple%20Records",
    "GET",
    undefined,
    alice,
  );
  assert.equal(filtered.response.status, 200);
  assert.ok(filtered.result.candidates.length > 0);
  assert.ok(filtered.result.pagination.total < first.result.pagination.total);
  assert.ok(
    filtered.result.candidates.every(
      (item) => item.country === "US" && item.year === 1968,
    ),
  );
  assert.ok(
    filtered.result.candidates.some((item) =>
      /Apple Records/i.test(item.label),
    ),
  );
  // Discogs can match a secondary label while versions show the primary label.
  const release = await request(
    `/api/catalog/${filtered.result.candidates[0].discogsId}`,
    "GET",
    undefined,
    alice,
  );
  assert.equal(release.response.status, 200);
  assert.equal(release.result.originalYear, 1968);
  assert.equal(release.result.country, "US");
  console.log(
    "PASS: Beatles White Album resolves its canonical album; real US vinyl pressings paginate and narrow by 1968 and Apple Records (which may be a secondary label).",
  );

  const fields = {
    mediaCondition: "VG+",
    sleeveCondition: "VG",
    purchasePrice: "24.50",
    currency: "USD",
    acquiredAt: "2026-10-01",
    notes: "Private catalog smoke note",
    rating: 4,
    location: "Test shelf",
    intent: "keep",
    pressingConfirmed: false,
  };
  const manualCandidate = {
    discogsId: null,
    artist: "The Beatles",
    title: "The White Album",
    year: null,
    originalYear: null,
    genres: [],
    label: "",
    catalogNumber: "",
    country: "",
    format: "Vinyl, LP",
    coverUrl: null,
    sourceUrl: null,
  };
  const idempotencyKey = randomUUID();
  const body = {
    idempotencyKey,
    candidate: manualCandidate,
    copy: fields,
    allowDuplicate: false,
  };
  const manual = await request("/api/records", "POST", body, alice);
  assert.equal(manual.response.status, 200);
  const repeat = await request("/api/records", "POST", body, alice);
  assert.equal(repeat.response.status, 200);
  assert.equal(repeat.result.id, manual.result.id);
  const path = `/api/records/${manual.result.id}`;
  const details = await request(
    "/api/catalog/albums?artist=The%20Beatles&title=The%20White%20Album",
    "GET",
    undefined,
    alice,
  );
  assert.equal(details.response.status, 200);
  assert.equal(details.result.albums[0].masterId, 46402);
  const untouched = await request(path, "GET", undefined, alice);
  assert.equal(untouched.result.discogsId, null);
  assert.equal(untouched.result.version, 1);
  const update = { version: 1, candidate: release.result, ...fields };
  assert.equal(
    (await request(path, "PATCH", update, bob)).response.status,
    404,
  );
  const saved = await request(path, "PATCH", update, alice);
  assert.equal(saved.response.status, 200);
  const reloaded = await request(path, "GET", undefined, alice);
  assert.equal(reloaded.result.discogsId, release.result.discogsId);
  assert.equal(reloaded.result.originalYear, 1968);
  assert.equal(reloaded.result.version, 2);
  for (const [key, value] of Object.entries(fields))
    assert.equal(reloaded.result[key], value);
  assert.equal(
    (await request(path, "GET", undefined, bob)).response.status,
    404,
  );
  const library = await request("/api/library", "GET", undefined, alice);
  assert.equal(library.result.records.length, 1);
  assert.equal(
    (await request(base + "&page=0", "GET", undefined, alice)).response.status,
    400,
  );
  assert.equal(
    (await request(base + "&year=abc", "GET", undefined, alice)).response
      .status,
    400,
  );
  console.log(
    "PASS: manual enrichment persists only after save, keeps copy details, remains private, and repeated creation cannot duplicate the copy.",
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
    "Temporary catalog collectors and their private data cleaned up.",
  );
}
