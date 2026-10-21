import assert from "node:assert/strict";
import { createHash, randomBytes, randomUUID } from "node:crypto";
import { Pool } from "pg";
import { del } from "@vercel/blob";

// Run with: node --env-file-if-exists=.env.local scripts/smoke-live.mjs
// This script intentionally refuses any target except the isolated local development database.
const origin = "http://localhost:3000";
const raw = process.env.DATABASE_URL || process.env.neon_connect || "";
const connectionString = raw.match(/postgres(?:ql)?:\/\/[^\s'"`]+/)?.[0];
if (
  !connectionString ||
  new URL(connectionString).pathname !== "/wax_development"
)
  throw new Error("Smoke tests require wax_development.");
if (process.env.OPENAI_API_KEY || process.env.chaz_gpt)
  throw new Error(
    "This smoke scenario expects the development AI key to be absent.",
  );

const pool = new Pool({
  connectionString,
  max: 2,
  connectionTimeoutMillis: 15_000,
});
const runId = randomUUID();
const people = ["a", "b"].map((letter) => ({
  email: `wax-smoke-${runId}-${letter}@example.test`,
  name: `WAX smoke ${letter}`,
  token: randomBytes(32).toString("base64url"),
  password: randomBytes(24).toString("base64url"),
  cookies: new Map(),
  id: null,
}));
const tokenHashes = people.map((person) =>
  createHash("sha256").update(person.token).digest("hex"),
);
const passed = [];
const initialReleaseIds = new Set();
let databaseVerified = false;
class SmokeFailure extends Error {}
function check(condition, message) {
  if (!condition) throw new SmokeFailure(message);
}
function pass(name) {
  passed.push(name);
  console.log(JSON.stringify({ check: name, status: "passed" }));
}

async function request(
  person,
  path,
  { method = "GET", body, expected = 200, binary = false } = {},
) {
  const headers = {
    Origin: origin,
    Accept: binary ? "image/png" : "application/json",
  };
  if (person?.cookies.size)
    headers.Cookie = [...person.cookies]
      .map(([key, value]) => `${key}=${value}`)
      .join("; ");
  if (body && !(body instanceof FormData))
    headers["Content-Type"] = "application/json";
  const response = await fetch(origin + path, {
    method,
    headers,
    body:
      body instanceof FormData ? body : body ? JSON.stringify(body) : undefined,
    redirect: "manual",
    signal: AbortSignal.timeout(90_000),
  });
  for (const cookie of response.headers.getSetCookie()) {
    const pair = cookie.split(";", 1)[0];
    const split = pair.indexOf("=");
    if (person && split > 0)
      person.cookies.set(pair.slice(0, split), pair.slice(split + 1));
  }
  check(
    response.status === expected,
    `${method} ${path.split("?")[0]} returned ${response.status}; expected ${expected}.`,
  );
  if (binary)
    return { data: Buffer.from(await response.arrayBuffer()), response };
  const text = await response.text();
  const data = response.headers
    .get("content-type")
    ?.includes("application/json")
    ? JSON.parse(text)
    : text;
  return { data, response };
}

async function cleanup() {
  if (!databaseVerified) return;
  const users = (
    await pool.query("SELECT id FROM wax.users WHERE email=ANY($1::text[])", [
      people.map((person) => person.email),
    ])
  ).rows.map((row) => row.id);
  const releases = users.length
    ? (
        await pool.query(
          "SELECT DISTINCT release_id FROM wax.copies WHERE owner_id=ANY($1::text[])",
          [users],
        )
      ).rows
        .map((row) => row.release_id)
        .filter((id) => !initialReleaseIds.has(id))
    : [];
  const photos = users.length
    ? (
        await pool.query(
          "SELECT owner_id,pathname FROM wax.photos WHERE owner_id=ANY($1::text[])",
          [users],
        )
      ).rows
    : [];
  for (const photo of photos) {
    check(
      users.includes(photo.owner_id) &&
        photo.pathname.startsWith(`wax/${encodeURIComponent(photo.owner_id)}/`),
      "Refusing to remove an unscoped Blob path.",
    );
    await del(photo.pathname, { abortSignal: AbortSignal.timeout(25_000) });
  }
  const client = await pool.connect();
  try {
    await client.query("BEGIN");
    await client.query(
      "DELETE FROM wax.invitations WHERE token_hash=ANY($1::text[])",
      [tokenHashes],
    );
    await client.query(
      "DELETE FROM wax.users WHERE id=ANY($1::text[]) AND email=ANY($2::text[])",
      [users, people.map((person) => person.email)],
    );
    if (releases.length)
      await client.query(
        "DELETE FROM wax.releases r WHERE r.id=ANY($1::uuid[]) AND NOT EXISTS(SELECT 1 FROM wax.copies c WHERE c.release_id=r.id)",
        [releases],
      );
    await client.query("COMMIT");
  } catch (error) {
    await client.query("ROLLBACK");
    throw error;
  } finally {
    client.release();
  }
  check(
    (
      await pool.query("SELECT id FROM wax.users WHERE email=ANY($1::text[])", [
        people.map((person) => person.email),
      ])
    ).rowCount === 0,
    "Smoke account cleanup failed.",
  );
  console.log(
    JSON.stringify({
      cleanup: "complete",
      accounts: users.length,
      privateBlobs: photos.length,
    }),
  );
}

try {
  const target = (
    await pool.query("SELECT current_database() AS db,current_user AS role")
  ).rows[0];
  check(
    target.db === "wax_development" && target.role === "wax_development",
    "Refusing a database or role outside isolated development.",
  );
  databaseVerified = true;
  for (const row of (await pool.query("SELECT id FROM wax.releases")).rows)
    initialReleaseIds.add(row.id);
  await request(null, "/api/health");
  await request(null, "/api/library", { expected: 401 });
  pass("isolated database and unauthenticated protection");

  for (const [index, person] of people.entries()) {
    await pool.query(
      "INSERT INTO wax.invitations(token_hash,email,kind,expires_at) VALUES($1,$2,'collector',now()+interval '15 minutes')",
      [tokenHashes[index], person.email],
    );
    const accepted = await request(null, "/api/invitations/accept", {
      method: "POST",
      body: {
        token: person.token,
        email: person.email,
        name: person.name,
        password: person.password,
      },
    });
    person.id = accepted.data.collector.id;
    await request(person, "/api/auth/sign-in/email", {
      method: "POST",
      body: { email: person.email, password: person.password },
    });
    check(person.cookies.size > 0, "Sign-in did not establish a cookie.");
    const library = (await request(person, "/api/library")).data;
    check(
      library.collector.id === person.id && library.records.length === 0,
      "New invited collector did not receive an empty private collection.",
    );
  }
  const [alice, bob] = people;
  pass("two invitation redemptions and authenticated empty collections");

  const found = (
    await request(
      alice,
      "/api/catalog?artist=Miles%20Davis&title=Kind%20Of%20Blue",
    )
  ).data;
  check(
    found.candidates.length > 0 &&
      found.candidates.every((candidate) => candidate.discogsId > 0),
    "Discogs search returned no reviewable releases.",
  );
  const barcode = (await request(alice, "/api/catalog?barcode=886976805715"))
    .data;
  check(
    barcode.candidates.length > 0,
    "Barcode lookup returned no reviewable releases.",
  );
  const candidate = (await request(alice, "/api/catalog/2825456")).data;
  check(
    candidate.discogsId === 2825456 &&
      candidate.artist &&
      candidate.title &&
      candidate.catalogNumber,
    "Exact Discogs pressing metadata is incomplete.",
  );
  pass("live Discogs text, barcode, and exact pressing lookup");

  const png = Buffer.from(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aY7cAAAAASUVORK5CYII=",
    "base64",
  );
  const form = new FormData();
  form.append(
    "file",
    new File([png], "smoke-fixture.png", { type: "image/png" }),
  );
  const upload = (
    await request(alice, "/api/photos", { method: "POST", body: form })
  ).data;
  check(
    upload.id && upload.url === `/api/photos/${upload.id}`,
    "Private upload did not return an authenticated photo route.",
  );
  const photo = await request(alice, upload.url, { binary: true });
  assert.deepEqual(photo.data, png);
  check(
    photo.response.headers.get("cache-control")?.includes("no-store"),
    "Private photo response permits caching.",
  );
  await request(bob, upload.url, { expected: 404 });
  await request(null, upload.url, { expected: 401 });
  const recognition = await request(alice, "/api/identify", {
    method: "POST",
    body: { photoId: upload.id },
    expected: 503,
  });
  check(
    typeof recognition.data.error === "string",
    "AI outage did not return a recoverable message.",
  );
  await request(alice, upload.url, { binary: true });
  pass(
    "live private Blob upload/read, cross-user denial, AI outage preserves photo",
  );

  const privateNote = `private-smoke-note-${runId}`;
  const copyFields = {
    mediaCondition: "VG+",
    sleeveCondition: "VG",
    purchasePrice: "123.45",
    currency: "USD",
    acquiredAt: "2026-01-01",
    notes: privateNote,
    rating: 4,
    location: "Smoke shelf",
    intent: "keep",
    pressingConfirmed: false,
  };
  const saveBody = {
    idempotencyKey: randomUUID(),
    candidate,
    copy: copyFields,
    allowDuplicate: false,
    photoIds: [upload.id],
  };
  const saved = (
    await request(alice, "/api/records", { method: "POST", body: saveBody })
  ).data;
  const retry = (
    await request(alice, "/api/records", { method: "POST", body: saveBody })
  ).data;
  check(
    saved.id === retry.id && saved.photoIds.includes(upload.id),
    "Repeated save created another copy or lost its photo.",
  );
  await request(alice, "/api/records", {
    method: "POST",
    body: { ...saveBody, idempotencyKey: randomUUID(), photoIds: [] },
    expected: 409,
  });
  const duplicate = (
    await request(alice, "/api/records", {
      method: "POST",
      body: {
        ...saveBody,
        idempotencyKey: randomUUID(),
        photoIds: [],
        allowDuplicate: true,
      },
    })
  ).data;
  check(
    saved.id !== duplicate.id,
    "Intentional duplicate was not a separate physical copy.",
  );
  const persisted = (await request(alice, "/api/library")).data;
  check(
    persisted.records.length === 2 &&
      persisted.records.some(
        (record) => record.id === saved.id && record.notes === privateNote,
      ),
    "Reload did not return persisted copies.",
  );
  const dbCopy = (
    await pool.query("SELECT owner_id,notes FROM wax.copies WHERE id=$1", [
      saved.id,
    ])
  ).rows[0];
  check(
    dbCopy?.owner_id === alice.id && dbCopy.notes === privateNote,
    "HTTP save was not persisted to the isolated Neon database.",
  );
  await request(bob, `/api/records/${saved.id}`, { expected: 404 });
  await request(bob, `/api/records/${saved.id}`, {
    method: "PATCH",
    body: { version: saved.version, notes: "Unauthorized" },
    expected: 404,
  });
  pass(
    "real record persistence, retry idempotency, intentional duplicate, ownership",
  );

  const bobTitle = `Bob smoke album ${runId}`;
  const bobRecord = (
    await request(bob, "/api/records", {
      method: "POST",
      body: {
        idempotencyKey: randomUUID(),
        candidate: {
          discogsId: null,
          artist: "Fixture artist",
          title: bobTitle,
          year: null,
          genres: [],
          label: "",
          catalogNumber: "",
          country: "",
          format: "Vinyl, LP",
          coverUrl: null,
          sourceUrl: null,
        },
        copy: { ...copyFields, notes: "Bob private note", purchasePrice: null },
        allowDuplicate: false,
      },
    })
  ).data;
  const crate = (
    await request(alice, "/api/crates", {
      method: "POST",
      body: {
        name: `Smoke crate ${runId}`,
        description: "Scoped live smoke fixture",
        color: "#d85a2a",
      },
    })
  ).data;
  await request(alice, `/api/crates/${crate.id}`, {
    method: "PATCH",
    body: { recordIds: [saved.id] },
  });
  await request(alice, `/api/crates/${crate.id}/shares`, {
    method: "POST",
    body: { email: bob.email },
  });
  const shared = (await request(bob, "/api/library")).data.sharedCrates.find(
    (item) => item.id === crate.id,
  );
  check(
    shared?.records.length === 1 && shared.records[0].id === saved.id,
    "Explicit crate sharing did not expose the selected copy.",
  );
  check(
    !JSON.stringify(shared).includes(privateNote),
    "Crate sharing leaked private notes.",
  );
  for (const field of [
    "notes",
    "purchasePrice",
    "currency",
    "photoIds",
    "location",
  ])
    check(
      !(field in shared.records[0]),
      `Crate sharing leaked the private ${field} field.`,
    );
  await request(bob, `/api/records/${saved.id}`, { expected: 404 });
  await request(bob, upload.url, { expected: 404 });
  await request(bob, `/api/crates/${crate.id}`, {
    method: "PATCH",
    body: { name: "Unauthorized" },
    expected: 404,
  });
  await request(alice, `/api/crates/${crate.id}/shares`, {
    method: "DELETE",
    body: { userId: bob.id },
  });
  check(
    !(await request(bob, "/api/library")).data.sharedCrates.some(
      (item) => item.id === crate.id,
    ),
    "Revoking a grant left shared access active.",
  );
  pass(
    "selective read-only sharing, private-field redaction, grant revocation",
  );

  const aliceExport = (await request(alice, "/api/export")).data;
  const bobExport = (await request(bob, "/api/export")).data;
  check(
    typeof aliceExport === "string" &&
      aliceExport.includes(saved.id) &&
      aliceExport.includes(privateNote) &&
      !aliceExport.includes(bobRecord.id),
    "First collector export was not limited to their library.",
  );
  check(
    typeof bobExport === "string" &&
      bobExport.includes(bobRecord.id) &&
      !bobExport.includes(saved.id) &&
      !bobExport.includes(privateNote),
    "Second collector export leaked another library.",
  );
  await request(alice, "/api/listens", {
    method: "POST",
    body: {
      recordId: saved.id,
      notes: "Smoke spin",
      playedAt: new Date().toISOString(),
    },
  });
  await request(bob, "/api/listens", {
    method: "POST",
    body: { recordId: saved.id, notes: "Unauthorized" },
    expected: 404,
  });
  const listened = (await request(alice, "/api/library")).data;
  check(
    listened.records.find((record) => record.id === saved.id)?.spinCount ===
      1 &&
      listened.listens.some(
        (listen) =>
          listen.recordId === saved.id && listen.notes === "Smoke spin",
      ),
    "Listening history did not persist.",
  );
  await request(alice, "/api/wizard", {
    method: "POST",
    body: { message: "What should I play?" },
    expected: 503,
  });
  check(
    (await request(alice, "/api/library")).data.records.length === 2,
    "AI outage changed the collection.",
  );
  pass("private CSV exports, persisted spins, graceful Wizard outage");
} catch (error) {
  console.error(
    JSON.stringify({
      smoke: "failed",
      reason:
        error instanceof SmokeFailure
          ? error.message
          : error instanceof assert.AssertionError
            ? "Fixture bytes did not match."
            : "Unexpected smoke execution error.",
      errorType: error?.name || "UnknownError",
      code: /^[A-Z0-9_]{2,40}$/.test(error?.code || "")
        ? error.code
        : undefined,
      passed: passed.length,
    }),
  );
  process.exitCode = 1;
} finally {
  try {
    await cleanup();
  } catch (error) {
    console.error(
      JSON.stringify({
        cleanup: "failed",
        errorType: error?.name || "UnknownError",
        fixtureRun: runId,
      }),
    );
    process.exitCode = 1;
  }
  await pool.end();
  console.log(
    JSON.stringify({
      smoke: process.exitCode ? "failed" : "passed",
      checks: passed.length,
      target: "isolated local development",
      fixtureRun: runId,
    }),
  );
}
