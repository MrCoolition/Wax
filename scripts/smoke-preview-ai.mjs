import { readFile } from "node:fs/promises";
import { createHash, randomBytes, randomUUID } from "node:crypto";
import { deflateSync } from "node:zlib";
import { Pool } from "pg";
import { del } from "@vercel/blob";

// Run with .env.local loaded and WAX_PREVIEW_HOST set to the approved deployment hostname.
// Only preview database credentials are read from the ignored local setup file.
const host = process.env.WAX_PREVIEW_HOST;
if (!host || !/^wax-[a-z0-9]+-coolition\.vercel\.app$/.test(host))
  throw new Error("An explicit WAX preview deployment hostname is required.");
const origin = `https://${host}`;
const oidc = process.env.VERCEL_OIDC_TOKEN;
if (!oidc) throw new Error("Local Vercel OIDC credentials are required.");
const setup = JSON.parse(
  await readFile(new URL("../.wax-local/setup.json", import.meta.url), "utf8"),
);
const preview = setup.databases?.find(
  (database) => database.name === "wax_preview",
);
if (!preview || !/^[a-f0-9]{64}$/.test(preview.password))
  throw new Error("Isolated preview credentials are unavailable.");
const dbUrl = new URL(
  "postgresql://ep-square-scene-b40q9y2w-pooler.c-6.us-east-2.aws.neon.tech/wax_preview?sslmode=verify-full",
);
dbUrl.username = "wax_preview";
dbUrl.password = preview.password;
const pool = new Pool({
  connectionString: dbUrl.href,
  max: 2,
  connectionTimeoutMillis: 15_000,
});

const runId = randomUUID();
const person = {
  email: `wax-preview-ai-${runId}@example.test`,
  name: "WAX preview AI smoke",
  token: randomBytes(32).toString("base64url"),
  password: randomBytes(24).toString("base64url"),
  cookies: new Map(),
  id: null,
};
const tokenHash = createHash("sha256").update(person.token).digest("hex");
const initialReleaseIds = new Set();
const passed = [];
let verified = false;
class SmokeFailure extends Error {}
function check(condition, message) {
  if (!condition) throw new SmokeFailure(message);
}
function pass(name) {
  passed.push(name);
  console.log(JSON.stringify({ check: name, status: "passed" }));
}

async function request(
  path,
  { method = "GET", body, expected = 200, authenticated = true } = {},
) {
  check(path.startsWith("/api/"), "Only application API paths are allowed.");
  const url = new URL(path, origin);
  check(url.origin === origin, "Refusing credentials for another origin.");
  const headers = {
    Origin: origin,
    Accept: "application/json",
    "x-vercel-trusted-oidc-idp-token": oidc,
  };
  if (authenticated && person.cookies.size)
    headers.Cookie = [...person.cookies]
      .map(([key, value]) => `${key}=${value}`)
      .join("; ");
  if (body && !(body instanceof FormData))
    headers["Content-Type"] = "application/json";
  const response = await fetch(url, {
    method,
    headers,
    body:
      body instanceof FormData ? body : body ? JSON.stringify(body) : undefined,
    redirect: "manual",
    signal: AbortSignal.timeout(100_000),
  });
  for (const cookie of response.headers.getSetCookie()) {
    const pair = cookie.split(";", 1)[0];
    const split = pair.indexOf("=");
    if (split > 0)
      person.cookies.set(pair.slice(0, split), pair.slice(split + 1));
  }
  check(
    response.status === expected,
    `${method} ${url.pathname} returned ${response.status}; expected ${expected}.`,
  );
  check(
    response.headers.get("content-type")?.includes("application/json"),
    `${url.pathname} did not return application JSON.`,
  );
  return response.json();
}

// An original text-only fixture exercises image reading without external artwork downloads.
function labelPng() {
  const glyphs = {
    A: ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    B: ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    C: ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
    D: ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    E: ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    F: ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    G: ["01111", "10000", "10000", "10111", "10001", "10001", "01110"],
    H: ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    I: ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
    J: ["00111", "00010", "00010", "00010", "10010", "10010", "01100"],
    K: ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    L: ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    M: ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    N: ["10001", "11001", "11001", "10101", "10011", "10011", "10001"],
    O: ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    P: ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    Q: ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    R: ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    S: ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    T: ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    U: ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    V: ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    W: ["10001", "10001", "10001", "10101", "10101", "11011", "10001"],
    X: ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    Y: ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    Z: ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
    0: ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    1: ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    2: ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    3: ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    4: ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    5: ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    6: ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
    7: ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    8: ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    9: ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
  };
  const width = 1050,
    height = 420,
    scale = 7,
    stride = width * 3 + 1;
  const pixels = Buffer.alloc(height * stride, 255);
  for (let y = 0; y < height; y++) pixels[y * stride] = 0;
  [
    "MILES DAVIS",
    "KIND OF BLUE",
    "COLUMBIA",
    "CATALOG 88697680571",
    "BARCODE 886976805715",
  ].forEach((line, row) => {
    [...line].forEach((character, index) => {
      glyphs[character]?.forEach((bits, gy) =>
        [...bits].forEach((bit, gx) => {
          if (bit !== "1") return;
          for (let dy = 0; dy < scale; dy++)
            for (let dx = 0; dx < scale; dx++) {
              const x = 35 + index * 6 * scale + gx * scale + dx;
              const y = 35 + row * 75 + gy * scale + dy;
              pixels.fill(
                0,
                y * stride + 1 + x * 3,
                y * stride + 1 + x * 3 + 3,
              );
            }
        }),
      );
    });
  });
  function chunk(type, payload) {
    const name = Buffer.from(type);
    let crc = 0xffffffff;
    for (const byte of Buffer.concat([name, payload])) {
      crc ^= byte;
      for (let i = 0; i < 8; i++)
        crc = (crc >>> 1) ^ (crc & 1 ? 0xedb88320 : 0);
    }
    const length = Buffer.alloc(4);
    length.writeUInt32BE(payload.length);
    const checksum = Buffer.alloc(4);
    checksum.writeUInt32BE((crc ^ 0xffffffff) >>> 0);
    return Buffer.concat([length, name, payload, checksum]);
  }
  const header = Buffer.alloc(13);
  header.writeUInt32BE(width);
  header.writeUInt32BE(height, 4);
  header[8] = 8;
  header[9] = 2;
  return Buffer.concat([
    Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]),
    chunk("IHDR", header),
    chunk("IDAT", deflateSync(pixels)),
    chunk("IEND", Buffer.alloc(0)),
  ]);
}

async function cleanup() {
  if (!verified) return;
  const users = (
    await pool.query("SELECT id FROM wax.users WHERE email=$1", [person.email])
  ).rows.map((row) => row.id);
  const releaseIds = users.length
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
      "Refusing cleanup of an unscoped Blob.",
    );
    await del(photo.pathname, { abortSignal: AbortSignal.timeout(25_000) });
  }
  const client = await pool.connect();
  try {
    await client.query("BEGIN");
    await client.query("DELETE FROM wax.invitations WHERE token_hash=$1", [
      tokenHash,
    ]);
    await client.query(
      "DELETE FROM wax.users WHERE id=ANY($1::text[]) AND email=$2",
      [users, person.email],
    );
    if (releaseIds.length)
      await client.query(
        "DELETE FROM wax.releases r WHERE r.id=ANY($1::uuid[]) AND NOT EXISTS(SELECT 1 FROM wax.copies c WHERE c.release_id=r.id)",
        [releaseIds],
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
      await pool.query("SELECT id FROM wax.users WHERE email=$1", [
        person.email,
      ])
    ).rowCount === 0,
    "Preview fixture cleanup failed.",
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
    target.db === "wax_preview" && target.role === "wax_preview",
    "Refusing a database or role outside isolated preview.",
  );
  verified = true;
  for (const row of (await pool.query("SELECT id FROM wax.releases")).rows)
    initialReleaseIds.add(row.id);
  await request("/api/health", { authenticated: false });
  await request("/api/library", { authenticated: false, expected: 401 });
  await pool.query(
    "INSERT INTO wax.invitations(token_hash,email,kind,expires_at) VALUES($1,$2,'collector',now()+interval '15 minutes')",
    [tokenHash, person.email],
  );
  person.id = (
    await request("/api/invitations/accept", {
      method: "POST",
      body: {
        token: person.token,
        email: person.email,
        name: person.name,
        password: person.password,
      },
      authenticated: false,
    })
  ).collector.id;
  await request("/api/auth/sign-in/email", {
    method: "POST",
    body: { email: person.email, password: person.password },
    authenticated: false,
  });
  check(
    person.cookies.size > 0 &&
      (await request("/api/library")).collector.id === person.id,
    "Preview authentication did not establish the intended collector.",
  );
  pass("protected preview access and isolated invitation sign-in");

  const image = labelPng();
  const form = new FormData();
  form.append(
    "file",
    new File([image], "synthetic-record-label.png", { type: "image/png" }),
  );
  const upload = await request("/api/photos", { method: "POST", body: form });
  const identification = await request("/api/identify", {
    method: "POST",
    body: { photoId: upload.id },
  });
  check(
    identification.photoId === upload.id &&
      /miles\s+davis/i.test(identification.clues.artist || "") &&
      /kind\s+of\s+blue/i.test(identification.clues.title || ""),
    "Live image recognition did not identify the printed artist/title.",
  );
  check(
    identification.clues.uncertainty &&
      Array.isArray(identification.candidates) &&
      identification.candidates.length > 0,
    "Photo clues did not resolve to reviewable real catalog candidates.",
  );
  check(
    identification.candidates.every(
      (candidate) =>
        candidate.discogsId > 0 && !("pressingConfirmed" in candidate),
    ),
    "Photo recognition asserted a confirmed pressing.",
  );
  pass("live OpenAI photo recognition and reviewable Discogs candidates");

  const candidate = await request("/api/catalog/2825456");
  const copy = await request("/api/records", {
    method: "POST",
    body: {
      idempotencyKey: randomUUID(),
      candidate,
      allowDuplicate: false,
      photoIds: [upload.id],
      copy: {
        mediaCondition: "VG+",
        sleeveCondition: "VG",
        purchasePrice: null,
        currency: "USD",
        acquiredAt: null,
        notes: `Preview smoke ${runId}`,
        rating: 4,
        location: "Original smoke shelf",
        intent: "keep",
        pressingConfirmed: false,
      },
    },
  });
  check(
    (await pool.query("SELECT owner_id FROM wax.copies WHERE id=$1", [copy.id]))
      .rows[0]?.owner_id === person.id,
    "Preview copy did not persist to the isolated database.",
  );
  const answer = await request("/api/wizard", {
    method: "POST",
    body: {
      message:
        "Which overlooked record from my collection should I play? Give one recommendation with a reason.",
    },
  });
  check(
    answer.recordIds.includes(copy.id) &&
      answer.recordIds.every((id) => id === copy.id) &&
      answer.proposal === null &&
      answer.reply.length > 0,
    "Wizard answer was not grounded in the actual owned record.",
  );
  pass("live collection-grounded Wizard recommendation");

  const proposed = await request("/api/wizard", {
    method: "POST",
    body: {
      message: `Please propose changing only the location of my copy of Kind Of Blue, record ID ${copy.id}, to "Preview smoke shelf". Leave every other field unchanged.`,
    },
  });
  check(
    proposed.proposal?.changes.length === 1,
    "Wizard did not return a single reviewable edit proposal.",
  );
  const change = proposed.proposal.changes[0];
  check(
    change.recordId === copy.id &&
      change.version === copy.version &&
      change.changes.location === "Preview smoke shelf" &&
      Object.keys(change.changes).length === 1,
    "Proposed edit exceeded the requested record or field.",
  );
  const before = await request(`/api/records/${copy.id}`);
  check(
    before.location === "Original smoke shelf" &&
      before.version === copy.version,
    "An unconfirmed Wizard proposal changed the collection.",
  );
  await request(`/api/wizard/proposals/${proposed.proposal.id}`, {
    method: "POST",
    body: { action: "approve" },
  });
  await request(`/api/wizard/proposals/${proposed.proposal.id}`, {
    method: "POST",
    body: { action: "approve" },
  });
  const after = await request(`/api/records/${copy.id}`);
  check(
    after.location === "Preview smoke shelf" &&
      after.version === copy.version + 1,
    "Reviewed proposal did not apply exactly once.",
  );
  pass("live AI proposal remains pending until confirmation and applies once");
} catch (error) {
  console.error(
    JSON.stringify({
      smoke: "failed",
      reason:
        error instanceof SmokeFailure
          ? error.message
          : "Unexpected preview smoke execution error.",
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
      target: "isolated preview",
      fixtureRun: runId,
    }),
  );
}
