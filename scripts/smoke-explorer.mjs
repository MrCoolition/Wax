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
    email: `wax-explorer-${randomBytes(8).toString("hex")}@example.test`,
    password: randomBytes(24).toString("base64url"),
    cookie: "",
    ip: `explorer-${randomBytes(8).toString("hex")}`,
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
const path = "/api/explore/releases/2825456";
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
        name: "Artist Tree verification",
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
  const releaseId = 2825456;
  const release = await request(
    `/api/catalog/${releaseId}`,
    "GET",
    undefined,
    alice,
  );
  assert.equal(release.response.status, 200);
  const fields = {
    mediaCondition: "Not graded",
    sleeveCondition: "Not graded",
    purchasePrice: null,
    currency: "USD",
    acquiredAt: null,
    notes: "Private explorer fixture",
    rating: 0,
    location: "",
    intent: "keep",
    pressingConfirmed: false,
  };
  const copy = await request(
    "/api/records",
    "POST",
    {
      idempotencyKey: randomUUID(),
      candidate: release.result,
      copy: fields,
      allowDuplicate: false,
    },
    alice,
  );
  assert.equal(copy.response.status, 200);
  const entryPath = `/api/records/${copy.result.id}/explore`;
  assert.equal((await request(entryPath, "GET")).response.status, 401);
  assert.equal(
    (await request(entryPath, "GET", undefined, bob)).response.status,
    404,
  );
  const entry = await request(entryPath, "GET", undefined, alice);
  assert.equal(entry.response.status, 200);
  assert.equal(
    entry.response.headers.get("cache-control"),
    "private, no-store",
  );
  assert.equal(entry.result.copyId, copy.result.id);
  assert.equal(entry.result.release.id, releaseId);
  assert.ok(!JSON.stringify(entry.result).includes(fields.notes));
  const album = entry.result.release;
  const first = album.tracks.find((track) => track.position === "A1");
  const second = album.tracks.find((track) => track.position === "A2");
  assert.ok(first && second);
  assert.ok(
    first.credits.some(
      (credit) => /Bill Evans/i.test(credit.name) && /Piano/i.test(credit.role),
    ),
  );
  const pianist = second.credits.find(
    (credit) => /Wynton Kelly/i.test(credit.name) && /Piano/i.test(credit.role),
  );
  assert.ok(pianist?.artistId);
  assert.ok(!second.credits.some((credit) => /Bill Evans/i.test(credit.name)));
  assert.ok(
    album.releaseCredits.some((credit) => /Trumpet/i.test(credit.role)),
  );
  console.log(
    "PASS: owned album resolves live song credits; Bill Evans and Wynton Kelly stay on their credited tracks, release-wide credits stay separate.",
  );

  const artistPath = `/api/explore/artists/${pianist.artistId}`;
  assert.equal((await request(artistPath, "GET")).response.status, 401);
  const artist = await request(artistPath, "GET", undefined, alice);
  assert.equal(artist.response.status, 200);
  assert.match(artist.result.name, /Wynton Kelly/i);
  const discography = await request(
    artistPath + "/releases?page=1",
    "GET",
    undefined,
    alice,
  );
  assert.equal(discography.response.status, 200);
  assert.ok(discography.result.releases.length > 0);
  assert.ok(discography.result.releases.length <= 20);
  const related = discography.result.releases.find(
    (item) => item.type === "release" || item.mainReleaseId,
  );
  assert.ok(related);
  const relatedId =
    related.type === "master" ? related.mainReleaseId : related.id;
  const next = await request(
    `/api/explore/releases/${relatedId}`,
    "GET",
    undefined,
    alice,
  );
  assert.equal(next.response.status, 200);
  assert.equal(next.result.id, relatedId);
  assert.ok(next.result.tracks.length > 0);
  console.log(
    "PASS: live Artist Tree profile and bounded discography link to a real related release using the correct release ID.",
  );

  const manualCandidate = {
    ...release.result,
    discogsId: null,
    sourceUrl: null,
    coverUrl: null,
    title: "My handwritten album",
  };
  const manual = await request(
    "/api/records",
    "POST",
    {
      idempotencyKey: randomUUID(),
      candidate: manualCandidate,
      copy: fields,
      allowDuplicate: false,
    },
    alice,
  );
  assert.equal(manual.response.status, 200);
  const manualStart = await request(
    `/api/records/${manual.result.id}/explore`,
    "GET",
    undefined,
    alice,
  );
  assert.equal(manualStart.response.status, 200);
  assert.equal(manualStart.result.release, null);
  assert.equal(manualStart.result.discogsId, null);
  const unchanged = await request(
    `/api/records/${manual.result.id}`,
    "GET",
    undefined,
    alice,
  );
  assert.equal(unchanged.result.version, 1);
  assert.equal(unchanged.result.title, manualCandidate.title);
  assert.equal(unchanged.result.pressingConfirmed, false);
  assert.equal(
    (await request("/api/explore/releases/0", "GET", undefined, alice)).response
      .status,
    400,
  );
  assert.equal(
    (await request(artistPath + "/releases?page=0", "GET", undefined, alice))
      .response.status,
    400,
  );
  console.log(
    "PASS: manual records remain untouched, all explorer routes require sign-in, and another collector cannot open the owned-record entry.",
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
    "Temporary Artist Tree collectors and their private data cleaned up.",
  );
}
