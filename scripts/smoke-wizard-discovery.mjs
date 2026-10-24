import assert from "node:assert/strict";
import { createHash, randomBytes, randomUUID } from "node:crypto";
import { Pool } from "pg";

// Explicit isolated-dev/preview opt-in: this performs five billable model calls.
assert.equal(
  process.env.WAX_RUN_PAID_SMOKE,
  "1",
  "Set WAX_RUN_PAID_SMOKE=1 to run the five live Wizard requests.",
);
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
    email: `wax-wizard-${randomBytes(8).toString("hex")}@example.test`,
    password: randomBytes(24).toString("base64url"),
    cookie: "",
    ip: `wizard-${randomBytes(8).toString("hex")}`,
  };
});
async function request(path, method = "GET", body, fixture) {
  const response = await fetch(origin + path, {
    method,
    redirect: "error",
    signal: AbortSignal.timeout(100_000),
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
async function library(fixture) {
  const result = await request("/api/library", "GET", undefined, fixture);
  assert.equal(result.response.status, 200);
  return result.result;
}
async function ask(fixture, input) {
  const result = await request("/api/wizard", "POST", input, fixture);
  assert.equal(result.response.status, 200, result.result.error);
  assert.equal(
    result.response.headers.get("cache-control"),
    "private, no-store",
  );
  assert.ok(result.result.reply.length > 20);
  return result.result;
}
async function verifyMichaelJacksonPicks(reply, fixture) {
  assert.match(reply.reply, /Off the Wall|Thriller|Bad/i);
  assert.ok(
    reply.recommendations.length >= 1 && reply.recommendations.length <= 3,
  );
  assert.deepEqual(reply.recordIds, []);
  assert.equal(reply.proposal, null);
  assert.ok(
    reply.recommendations.some((album) =>
      /Michael Jackson/i.test(album.artist),
    ),
  );
  for (const album of reply.recommendations) {
    assert.equal(
      album.sourceUrl,
      `https://www.discogs.com/master/${album.masterId}`,
    );
    assert.ok(Number.isSafeInteger(album.masterId) && album.masterId > 0);
    assert.ok(album.reason.length > 10);
    const lookup = await request(
      "/api/catalog/albums?" +
        new URLSearchParams({ artist: album.artist, title: album.title }),
      "GET",
      undefined,
      fixture,
    );
    assert.equal(lookup.response.status, 200);
    assert.ok(
      lookup.result.albums.some((found) => found.masterId === album.masterId),
      "Recommendation must match real catalog results.",
    );
  }
}
async function addManual(fixture, artist, title, notes) {
  const result = await request(
    "/api/records",
    "POST",
    {
      idempotencyKey: randomUUID(),
      allowDuplicate: false,
      candidate: {
        discogsId: null,
        artist,
        title,
        year: null,
        originalYear: 1977,
        genres: ["Rock"],
        label: "",
        catalogNumber: "",
        country: "",
        format: "Vinyl, LP",
        coverUrl: null,
        sourceUrl: null,
      },
      copy: {
        mediaCondition: "VG+",
        sleeveCondition: "VG",
        purchasePrice: "15.00",
        currency: "USD",
        acquiredAt: "2026-10-01",
        notes,
        rating: 4,
        location: "Test shelf",
        intent: "keep",
        pressingConfirmed: false,
      },
    },
    fixture,
  );
  assert.equal(result.response.status, 200);
  return result.result;
}

try {
  assert.equal(
    (await pool.query("SELECT current_user")).rows[0].current_user,
    database,
  );
  assert.equal(
    (await request("/api/wizard", "POST", { message: "What should I buy?" }))
      .response.status,
    401,
  );
  for (const fixture of fixtures) {
    await pool.query(
      "INSERT INTO wax.invitations(token_hash,email,kind,expires_at) VALUES($1,$2,'collector',now()+interval '15 minutes')",
      [fixture.digest, fixture.email],
    );
    assert.equal(
      (
        await request(
          "/api/invitations/accept",
          "POST",
          {
            token: fixture.token,
            email: fixture.email,
            password: fixture.password,
            name: "Wizard discovery verification",
          },
          fixture,
        )
      ).response.status,
      200,
    );
    const signedIn = await request(
      "/api/auth/sign-in/email",
      "POST",
      { email: fixture.email, password: fixture.password },
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
  const bobPrivate = await addManual(
    bob,
    "Private fixture artist",
    "BOB_PRIVATE_RECORD_SENTINEL",
    "BOB_PRIVATE_NOTE_SENTINEL",
  );
  const emptyBefore = await library(alice);
  assert.deepEqual(emptyBefore.records, []);
  const buying = await ask(alice, {
    message:
      "I want to buy my first Michael Jackson vinyl album. Which one should I grab? Give me a clear pick and verify it in the catalog.",
  });
  await verifyMichaelJacksonPicks(buying, alice);
  assert.deepEqual((await library(alice)).records, []);
  assert.ok(!JSON.stringify(buying).includes("BOB_PRIVATE"));
  console.log(
    "PASS: an empty collection receives real Michael Jackson buying recommendations and verified catalog cards without adding records.",
  );

  const copy = await addManual(
    alice,
    "Steely Dan",
    "Aja",
    "Private smoke note; retain unchanged.",
  );
  const before = (await library(alice)).records;
  const correction = await ask(alice, {
    message:
      "nah bro, I mean I want to BUY a Michael Jackson record, not play something I already own. Give me one clear pick and verify it.",
    history: [
      { role: "user", content: "Which Michael Jackson album should I grab?" },
      {
        role: "assistant",
        content:
          "You do not own any Michael Jackson. You could play Aja from your collection.",
      },
    ],
  });
  await verifyMichaelJacksonPicks(correction, alice);
  assert.ok(!JSON.stringify(correction).includes(bobPrivate.id));
  assert.deepEqual((await library(alice)).records, before);
  console.log(
    "PASS: a buying correction follows conversation intent and does not attach unrelated owned records or write to the collection.",
  );

  const play = await ask(alice, {
    message:
      "What record from my own collection should I play tonight? Name it and link my copy.",
  });
  assert.match(play.reply, /Aja/i);
  assert.deepEqual(play.recordIds, [copy.id]);
  assert.equal(play.proposal, null);
  assert.deepEqual((await library(alice)).records, before);
  console.log(
    "PASS: owned listening recommendations still refer to the authenticated collector's actual copy.",
  );

  const credits = await ask(alice, {
    message:
      "For Miles Davis's Kind Of Blue, Discogs release 2825456, who plays piano on Freddie Freeloader, and how does that differ from So What? Retrieve the actual track credits and include the catalog source.",
  });
  assert.match(credits.reply, /Wynton Kelly/i);
  assert.match(credits.reply, /Bill Evans/i);
  assert.deepEqual(credits.recordIds, []);
  assert.equal(credits.proposal, null);
  assert.ok(
    credits.sources?.some(
      (source) =>
        source.id === "release:2825456" &&
        source.url === "https://www.discogs.com/release/2825456",
    ),
  );
  assert.deepEqual((await library(alice)).records, before);
  console.log(
    "PASS: the Wizard retrieves actual track-specific pianist credits and supplies a verified release source without changing the collection.",
  );

  const edit = await ask(alice, {
    message:
      "Please move my copy of Aja to the location 'Wizard smoke shelf'. Propose only that location change for review.",
  });
  assert.ok(edit.proposal?.id);
  assert.equal(edit.proposal.changes.length, 1);
  assert.equal(edit.proposal.changes[0].recordId, copy.id);
  assert.deepEqual(edit.proposal.changes[0].changes, {
    location: "Wizard smoke shelf",
  });
  assert.deepEqual((await library(alice)).records, before);
  const proposalPath = `/api/wizard/proposals/${edit.proposal.id}`;
  assert.equal(
    (await request(proposalPath, "POST", { action: "approve" }, bob)).response
      .status,
    404,
  );
  assert.equal(
    (await request(proposalPath, "POST", { action: "reject" }, alice)).response
      .status,
    200,
  );
  assert.deepEqual((await library(alice)).records, before);
  console.log(
    "PASS: edits remain reviewable proposals, another collector cannot approve them, and rejecting leaves the copy unchanged.",
  );
} finally {
  for (const fixture of fixtures) {
    await pool.query("DELETE FROM wax.invitations WHERE token_hash=$1", [
      fixture.digest,
    ]);
    await pool.query("DELETE FROM wax.users WHERE email=$1", [fixture.email]);
  }
  await pool.end();
  console.log("Temporary Wizard collectors and their private data cleaned up.");
}
