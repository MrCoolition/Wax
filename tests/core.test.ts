import {
  beforeAll,
  beforeEach,
  afterAll,
  describe,
  expect,
  it,
  vi,
} from "vitest";
import { readFile } from "node:fs/promises";
import { PGlite, type PGliteInterface } from "@electric-sql/pglite";
import { drizzle } from "drizzle-orm/node-postgres";
import type { Pool, PoolClient } from "pg";
import * as schema from "../src/server/schema";
import {
  DEFAULT_COPY,
  EMPTY_CANDIDATE,
  type CreateRecordInput,
} from "../src/lib/contracts";

const state = vi.hoisted(() => ({
  db: null as unknown as PGlite,
  pool: null as unknown as Pool,
  requestHeaders: new Headers(),
}));
vi.mock("next/headers", () => ({ headers: async () => state.requestHeaders }));
function facade(database: Pick<PGliteInterface, "query">) {
  return {
    async query(
      query: string | { text: string; rowMode?: "array" },
      parameters?: unknown[],
    ) {
      const result = await database.query(
        typeof query === "string" ? query : query.text,
        parameters,
        {
          rowMode:
            typeof query === "string" ? "object" : (query.rowMode ?? "object"),
        },
      );
      return { ...result, rowCount: result.affectedRows, command: "", oid: 0 };
    },
    release() {},
  } as unknown as PoolClient;
}
vi.mock("../src/server/db", async (original) => {
  const real = await original<typeof import("../src/server/db")>();
  return {
    ...real,
    getPool: () => state.pool,
    getDb: () => drizzle(state.pool, { schema }),
    withTransaction: async <T>(action: (client: PoolClient) => Promise<T>) =>
      state.db.transaction(async (tx) => action(facade(tx))),
  };
});
vi.mock("../src/server/catalog", async (original) => ({
  ...(await original<typeof import("../src/server/catalog")>()),
  fetchRelease: vi.fn(async (id: number) => ({
    ...EMPTY_CANDIDATE,
    discogsId: id,
    artist: "The Meters",
    title: "Rejuvenation",
    year: 1974,
    label: "Reprise",
    catalogNumber: "MS 2200",
    country: "US",
    genres: ["Funk"],
    coverUrl: "https://i.discogs.com/example.jpg",
    sourceUrl: `https://www.discogs.com/release/${id}`,
  })),
}));
import {
  approveChanges,
  createCopy,
  createCrate,
  csvCell,
  deleteCopy,
  getCopy,
  getLibrary,
  logListen,
  shareCrate,
  unshareCrate,
  updateCopy,
  updateCrate,
  exportCollection,
} from "../src/server/collection";
import {
  acceptInvitation,
  createInvitation,
  inspectInvitation,
  revokeInvitation,
  tokenDigest,
} from "../src/server/invitations";
import { createWaxAuth } from "../src/server/auth";
import { handleApi } from "../src/server/http";
import { parseConnectionString } from "../src/server/db";
import { databaseUrl } from "../src/server/database-url";
import { GET as libraryRoute } from "../src/app/api/library/route";
import { GET as copyRoute } from "../src/app/api/records/[id]/route";
import { POST as createRecordRoute } from "../src/app/api/records/route";
import { POST as authRoute } from "../src/app/api/auth/[...all]/route";
import { seedOwnerInvitation } from "../src/server/bootstrap";

const owner = {
  id: "owner",
  name: "Owner",
  email: "owner@example.com",
  isOwner: true,
};
const alice = {
  id: "alice",
  name: "Alice",
  email: "alice@example.com",
  isOwner: false,
};
function record(overrides: Partial<CreateRecordInput> = {}): CreateRecordInput {
  return {
    idempotencyKey: crypto.randomUUID(),
    candidate: {
      ...EMPTY_CANDIDATE,
      artist: "The Meters",
      title: "Rejuvenation",
    },
    copy: { ...DEFAULT_COPY },
    allowDuplicate: false,
    ...overrides,
  };
}
beforeAll(async () => {
  process.env.BETTER_AUTH_SECRET =
    "test-only-auth-secret-not-used-outside-fixtures-123456";
  process.env.BETTER_AUTH_URL = "http://localhost:3000";
  state.db = new PGlite({
    parsers: {
      1184: (value) => value,
      1114: (value) => value,
      1082: (value) => value,
    },
  });
  await state.db.waitReady;
  await state.db.exec(
    await readFile(
      new URL("../migrations/0001_core.sql", import.meta.url),
      "utf8",
    ),
  );
  await state.db.exec(
    await readFile(
      new URL("../migrations/0002_integrations.sql", import.meta.url),
      "utf8",
    ),
  );
  await state.db.exec(
    await readFile(
      new URL("../migrations/0003_catalog_separation.sql", import.meta.url),
      "utf8",
    ),
  );
  state.pool = facade(state.db) as unknown as Pool;
});
beforeEach(async () => {
  state.requestHeaders = new Headers();
  await state.db.exec(
    "TRUNCATE wax.users,wax.releases,wax.registration_limits,wax.catalog_cache,wax.rate_limits CASCADE",
  );
  for (const [id, name] of [
    ["owner", "Owner"],
    ["alice", "Alice"],
    ["bob", "Bob"],
  ]) {
    await state.db.query(
      "INSERT INTO wax.users(id,name,email) VALUES($1,$2,$3)",
      [id, name, `${id}@example.com`],
    );
    await state.db.query("INSERT INTO wax.collectors(id) VALUES($1)", [id]);
  }
  await state.db.query("INSERT INTO wax.app_owner(user_id) VALUES($1)", [
    "owner",
  ]);
});
afterAll(async () => {
  await state.db.close();
});

describe("collection isolation and real SQL behavior", () => {
  it("saves manual records without inventing a confirmed pressing or accepting external images", async () => {
    const input = record({
      candidate: {
        ...EMPTY_CANDIDATE,
        artist: "Artist",
        title: "Title",
        coverUrl: "https://attacker.invalid/image.jpg",
      },
      copy: { ...DEFAULT_COPY, pressingConfirmed: true },
    });
    const copy = await createCopy("alice", input);
    expect(copy.pressingConfirmed).toBe(false);
    expect(copy.coverUrl).toBeNull();
    expect((await getLibrary("alice")).records).toHaveLength(1);
    expect((await getLibrary("bob")).records).toHaveLength(0);
  });
  it("denies cross-collector reads, edits, and deletes", async () => {
    const copy = await createCopy("alice", record());
    await expect(getCopy("bob", copy.id)).rejects.toMatchObject({
      status: 404,
    });
    await expect(
      updateCopy("bob", copy.id, { version: 1, notes: "hijack" }),
    ).rejects.toMatchObject({ status: 404 });
    await expect(deleteCopy("bob", copy.id, 1)).rejects.toMatchObject({
      status: 404,
    });
    expect((await getCopy("alice", copy.id)).notes).toBe("");
  });
  it("deduplicates concurrent retries but permits another deliberate physical copy", async () => {
    const input = record();
    const saved = await Promise.all(
      Array.from({ length: 5 }, () => createCopy("alice", input)),
    );
    expect(new Set(saved.map((v) => v.id)).size).toBe(1);
    await expect(createCopy("alice", record())).rejects.toMatchObject({
      status: 409,
    });
    const second = await createCopy("alice", record({ allowDuplicate: true }));
    expect(second.id).not.toBe(saved[0].id);
    await expect(
      createCopy("alice", {
        ...input,
        copy: { ...DEFAULT_COPY, notes: "different" },
      }),
    ).rejects.toMatchObject({ status: 409 });
    expect((await getLibrary("alice")).records).toHaveLength(2);
  });
  it("shares only a safe projection and revokes access immediately", async () => {
    const copy = await createCopy(
      "alice",
      record({
        copy: {
          ...DEFAULT_COPY,
          notes: "PRIVATE-NOTE",
          purchasePrice: "119.99",
          location: "SECRET-SHELF",
        },
      }),
    );
    const crate = await createCrate("alice", { name: "Funk" });
    await updateCrate("alice", crate.id, { recordIds: [copy.id] });
    await shareCrate("alice", crate.id, "bob@example.com");
    const library = await getLibrary("bob");
    expect(library.sharedCrates[0].records[0].title).toBe("Rejuvenation");
    expect(JSON.stringify(library)).not.toContain("PRIVATE-NOTE");
    expect(JSON.stringify(library)).not.toContain("119.99");
    expect(JSON.stringify(library)).not.toContain("SECRET-SHELF");
    await expect(
      updateCrate("bob", crate.id, { name: "Stolen" }),
    ).rejects.toMatchObject({ status: 404 });
    await unshareCrate("alice", crate.id, "bob");
    expect((await getLibrary("bob")).sharedCrates).toHaveLength(0);
  });
  it("rejects cross-owner nested crate records and photos with atomic rollback", async () => {
    const other = await createCopy("bob", record());
    const crate = await createCrate("alice", { name: "Mine" });
    await expect(
      updateCrate("alice", crate.id, {
        name: "Changed",
        recordIds: [other.id],
      }),
    ).rejects.toMatchObject({ status: 404 });
    expect((await getLibrary("alice")).crates[0].name).toBe("Mine");
    const photo = (
      await state.db.query<{ id: string }>(
        `INSERT INTO wax.photos(owner_id,pathname,content_type,byte_size) VALUES('bob','bob/private.jpg','image/jpeg',100) RETURNING id`,
      )
    ).rows[0];
    await expect(
      createCopy("alice", record({ photoIds: [photo.id] })),
    ).rejects.toMatchObject({ status: 409 });
    expect((await getLibrary("alice")).records).toHaveLength(0);
  });
  it("does not change another copy when editing copy-level metadata", async () => {
    const candidate = {
      ...EMPTY_CANDIDATE,
      discogsId: 123,
      artist: "The Meters",
      title: "Rejuvenation",
    };
    const a = await createCopy("alice", record({ candidate }));
    const b = await createCopy("bob", record({ candidate }));
    await updateCopy("alice", a.id, {
      version: 1,
      candidate: { ...candidate, title: "My transcription" },
    });
    expect((await getCopy("bob", b.id)).title).toBe("Rejuvenation");
  });
  it("keeps lazy artwork resolvers after cache expiry without persisting provider artwork", async () => {
    const candidate = {
      ...EMPTY_CANDIDATE,
      discogsId: 123,
      artist: "The Meters",
      title: "My reviewed title",
    };
    await state.db.query(
      "INSERT INTO wax.catalog_cache(cache_key,payload) VALUES($1,$2)",
      [
        "release:123",
        JSON.stringify({
          ...candidate,
          coverUrl: "https://i.discogs.com/current.jpg",
        }),
      ],
    );
    const copy = await createCopy("alice", record({ candidate }));
    expect(copy.coverUrl).toBe("/api/catalog/123/artwork");
    await state.db.exec(
      "UPDATE wax.catalog_cache SET fetched_at=now()-interval '7 hours'",
    );
    expect((await getCopy("alice", copy.id)).coverUrl).toBe(
      "/api/catalog/123/artwork",
    );
    await state.db.query(
      "UPDATE wax.catalog_cache SET payload=$1,fetched_at=now()",
      [
        JSON.stringify({
          ...candidate,
          title: "Provider correction",
          coverUrl: "https://i.discogs.com/refreshed.jpg",
        }),
      ],
    );
    const refreshed = await getCopy("alice", copy.id);
    expect(refreshed.title).toBe("My reviewed title");
    expect(refreshed.coverUrl).toBe("/api/catalog/123/artwork");
    expect(
      (
        await state.db.query<{ metadata: { coverUrl: null } }>(
          "SELECT metadata FROM wax.copies WHERE id=$1",
          [copy.id],
        )
      ).rows[0].metadata.coverUrl,
    ).toBeNull();
    expect(
      (
        await state.db.query<{ metadata: object }>(
          "SELECT metadata FROM wax.releases WHERE id=$1",
          [copy.releaseId],
        )
      ).rows[0].metadata,
    ).toEqual({});
    // Editing the returned candidate must accept its same-origin image URL.
    const edited = await updateCopy("alice", copy.id, {
      version: refreshed.version,
      candidate: { ...candidate, coverUrl: refreshed.coverUrl },
    });
    expect(edited.coverUrl).toBe("/api/catalog/123/artwork");
  });
  it("uses the same lazy cover in listening history and explicitly shared crates", async () => {
    const copy = await createCopy(
      "alice",
      record({
        candidate: {
          ...EMPTY_CANDIDATE,
          discogsId: 123,
          artist: "The Meters",
          title: "Rejuvenation",
        },
      }),
    );
    await logListen("alice", { recordId: copy.id, notes: "Private listen" });
    const crate = await createCrate("alice", {
      name: "Funk",
      description: "",
      color: "#d85a2a",
    });
    await updateCopy("alice", copy.id, { version: 1, crateIds: [crate.id] });
    expect((await getLibrary("bob")).sharedCrates).toEqual([]);
    await shareCrate("alice", crate.id, "bob@example.com");
    const shared = (await getLibrary("bob")).sharedCrates[0].records[0];
    expect(shared.coverUrl).toBe("/api/catalog/123/artwork");
    expect(shared).not.toHaveProperty("notes");
    expect((await getLibrary("alice")).listens[0].coverUrl).toBe(
      "/api/catalog/123/artwork",
    );
    expect((await getLibrary("bob")).listens).toEqual([]);
    await unshareCrate("alice", crate.id, "bob");
    expect((await getLibrary("bob")).sharedCrates).toEqual([]);
  });
  it("applies approved edits atomically and rejects stale or foreign edits", async () => {
    const first = await createCopy("alice", record());
    const second = await createCopy("alice", record({ allowDuplicate: true }));
    await expect(
      approveChanges("alice", [
        {
          recordId: first.id,
          version: 1,
          changes: { notes: "Should roll back" },
        },
        { recordId: second.id, version: 9, changes: { rating: 5 } },
      ]),
    ).rejects.toMatchObject({ status: 409 });
    expect((await getCopy("alice", first.id)).notes).toBe("");
    await expect(
      approveChanges("bob", [
        { recordId: first.id, version: 1, changes: { rating: 5 } },
      ]),
    ).rejects.toMatchObject({ status: 404 });
    await approveChanges("alice", [
      { recordId: first.id, version: 1, changes: { rating: 5 } },
    ]);
    expect((await getCopy("alice", first.id)).version).toBe(2);
    await expect(
      approveChanges("alice", [
        { recordId: first.id, version: 1, changes: { rating: 1 } },
      ]),
    ).rejects.toMatchObject({ status: 409 });
  });
  it("exports only owned copies and neutralizes spreadsheet formulas", async () => {
    await createCopy(
      "alice",
      record({
        candidate: { ...EMPTY_CANDIDATE, artist: "=1+1", title: "Album" },
      }),
    );
    await createCopy(
      "bob",
      record({
        candidate: {
          ...EMPTY_CANDIDATE,
          artist: "PRIVATE-BOB",
          title: "Album",
        },
      }),
    );
    const csv = await (await exportCollection("alice")).text();
    expect(csv).toContain("'=1+1");
    expect(csv).not.toContain("PRIVATE-BOB");
    expect(csvCell("\t=SUM(A1)")).toBe('"\'\t=SUM(A1)"');
  });
});

describe("invite-only Better Auth", () => {
  it("blocks both public HTTP signup and direct public-instance signup", async () => {
    const auth = createWaxAuth(drizzle(state.pool, { schema }));
    const response = await auth.handler(
      new Request("http://localhost:3000/api/auth/sign-up/email", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Origin: "http://localhost:3000",
        },
        body: JSON.stringify({
          email: "outsider@example.com",
          name: "Outsider",
          password: "Long-password-1234",
        }),
      }),
    );
    expect(response.status).toBe(404);
    await expect(
      auth.api.signUpEmail({
        body: {
          email: "outsider@example.com",
          name: "Outsider",
          password: "Long-password-1234",
        },
      }),
    ).rejects.toMatchObject({ statusCode: 400 });
    expect(
      (
        await state.db.query("SELECT id FROM wax.users WHERE email=$1", [
          "outsider@example.com",
        ])
      ).rows,
    ).toHaveLength(0);
  });
  it("redeems a bound invitation once, creates a private collector, and allows password sign-in", async () => {
    const invite = await createInvitation(owner, "new@example.com");
    const token = new URL(invite.url).hash.split("token=")[1];
    expect(await inspectInvitation(token)).toEqual({
      email: "new@example.com",
      kind: "collector",
    });
    const input = {
      token,
      email: "new@example.com",
      name: "New Collector",
      password: "Long-password-1234",
    };
    const accepted = await acceptInvitation(input);
    expect(accepted.collector.isOwner).toBe(false);
    expect((await getLibrary(accepted.collector.id)).records).toEqual([]);
    await expect(acceptInvitation(input)).rejects.toMatchObject({
      status: 400,
    });
    const auth = createWaxAuth(drizzle(state.pool, { schema }));
    const signin = await auth.api.signInEmail({
      body: { email: input.email, password: input.password },
      asResponse: true,
    });
    expect(signin.status).toBe(200);
    expect(signin.headers.get("set-cookie")).toContain("wax.session_token");
  });
  it("rejects wrong-email, revoked, and expired tokens without creating accounts", async () => {
    const invite = await createInvitation(owner, "new@example.com");
    const token = new URL(invite.url).hash.split("token=")[1];
    await expect(
      acceptInvitation({
        token,
        email: "different@example.com",
        name: "No",
        password: "Long-password-1234",
      }),
    ).rejects.toMatchObject({ status: 400 });
    expect(await inspectInvitation(token)).toMatchObject({
      email: "new@example.com",
    });
    await revokeInvitation(owner, invite.id);
    await expect(inspectInvitation(token)).rejects.toMatchObject({
      status: 404,
    });
    const expired = "x".repeat(43);
    await state.db.query(
      `INSERT INTO wax.invitations(token_hash,email,kind,expires_at) VALUES($1,'expired@example.com','collector',now()-interval '1 minute')`,
      [tokenDigest(expired)],
    );
    await expect(
      acceptInvitation({
        token: expired,
        email: "expired@example.com",
        name: "No",
        password: "Long-password-1234",
      }),
    ).rejects.toMatchObject({ status: 400 });
    await expect(
      createInvitation(alice, "other@example.com"),
    ).rejects.toMatchObject({ status: 403 });
  });
  it("rolls back consumption and the auth account if owner bootstrap already exists", async () => {
    const token = "z".repeat(43);
    await state.db.query(
      `INSERT INTO wax.invitations(token_hash,email,kind,expires_at) VALUES($1,'second-owner@example.com','owner',now()+interval '1 day')`,
      [tokenDigest(token)],
    );
    await expect(
      acceptInvitation({
        token,
        email: "second-owner@example.com",
        name: "Second",
        password: "Long-password-1234",
      }),
    ).rejects.toMatchObject({ code: "23505" });
    expect(
      (
        await state.db.query("SELECT id FROM wax.users WHERE email=$1", [
          "second-owner@example.com",
        ])
      ).rows,
    ).toHaveLength(0);
    expect(await inspectInvitation(token)).toMatchObject({ kind: "owner" });
    expect(
      (await state.db.query("SELECT * FROM wax.app_owner")).rows,
    ).toHaveLength(1);
  });
  it("reissues an expired bootstrap only after token rotation and never replaces an established owner", async () => {
    await state.db.exec("DELETE FROM wax.app_owner");
    const oldToken = "old-bootstrap-token".repeat(3),
      newToken = "new-bootstrap-token".repeat(3);
    await seedOwnerInvitation(state.pool, "initial@example.com", oldToken);
    await state.db.exec(
      "UPDATE wax.invitations SET expires_at=now()-interval '1 day'",
    );
    await seedOwnerInvitation(state.pool, "initial@example.com", oldToken);
    await expect(inspectInvitation(oldToken)).rejects.toMatchObject({
      status: 404,
    });
    await seedOwnerInvitation(state.pool, "initial@example.com", newToken);
    await expect(inspectInvitation(oldToken)).rejects.toMatchObject({
      status: 404,
    });
    expect(await inspectInvitation(newToken)).toEqual({
      email: "initial@example.com",
      kind: "owner",
    });
    const accepted = await acceptInvitation({
      token: newToken,
      email: "initial@example.com",
      name: "Initial",
      password: "Long-password-1234",
    });
    expect(accepted.collector.isOwner).toBe(true);
    await seedOwnerInvitation(
      state.pool,
      "attacker@example.com",
      "replacement-token".repeat(3),
    );
    expect(
      (
        await state.db.query<{ user_id: string }>(
          "SELECT user_id FROM wax.app_owner",
        )
      ).rows[0].user_id,
    ).toBe(accepted.collector.id);
    expect(
      (
        await state.db.query(
          "SELECT id FROM wax.invitations WHERE email='attacker@example.com'",
        )
      ).rows,
    ).toHaveLength(0);
  });
  it("authorizes HTTP collection routes with a real session cookie and stops accepting revoked sessions", async () => {
    const libraryRequest = new Request("http://localhost:3000/api/library");
    expect((await libraryRoute(libraryRequest)).status).toBe(401);
    const invite = await createInvitation(owner, "http@example.com");
    const token = new URL(invite.url).hash.split("token=")[1];
    const accepted = await acceptInvitation({
      token,
      email: "http@example.com",
      name: "HTTP Collector",
      password: "Long-password-1234",
    });
    const auth = createWaxAuth(drizzle(state.pool, { schema }));
    const signin = await auth.handler(
      new Request("http://localhost:3000/api/auth/sign-in/email", {
        method: "POST",
        headers: {
          Origin: "http://localhost:3000",
          "Content-Type": "application/json",
          "x-real-ip": "192.0.2.10",
        },
        body: JSON.stringify({
          email: "http@example.com",
          password: "Long-password-1234",
        }),
      }),
    );
    expect(signin.status).toBe(200);
    state.requestHeaders = new Headers({
      Cookie: signin.headers
        .getSetCookie()
        .map((cookie) => cookie.split(";")[0])
        .join("; "),
    });
    const saved = await createRecordRoute(
      new Request("http://localhost:3000/api/records", {
        method: "POST",
        headers: {
          Origin: "http://localhost:3000",
          "Content-Type": "application/json",
        },
        body: JSON.stringify(record()),
      }),
    );
    expect(saved.status).toBe(200);
    expect(
      (await (await libraryRoute(libraryRequest)).json()).records,
    ).toHaveLength(1);
    const foreign = await createCopy("alice", record());
    const denied = await copyRoute(
      new Request(`http://localhost:3000/api/records/${foreign.id}`),
      { params: Promise.resolve({ id: foreign.id }) },
    );
    expect(denied.status).toBe(404);
    expect(await denied.json()).toEqual({ error: "Record not found." });
    await state.db.query("DELETE FROM wax.sessions WHERE user_id=$1", [
      accepted.collector.id,
    ]);
    expect((await libraryRoute(libraryRequest)).status).toBe(401);
  });
  it("blocks cross-site authentication and rate-limits failed HTTP sign-ins in the database", async () => {
    const signin = (origin: string) =>
      authRoute(
        new Request("http://localhost:3000/api/auth/sign-in/email", {
          method: "POST",
          headers: {
            Origin: origin,
            "Content-Type": "application/json",
            "x-real-ip": "192.0.2.11",
          },
          body: JSON.stringify({
            email: "unknown@example.com",
            password: "Long-password-1234",
          }),
        }),
      );
    expect((await signin("https://evil.example")).status).toBe(403);
    const statuses = [];
    for (let attempt = 0; attempt < 9; attempt++)
      statuses.push((await signin("http://localhost:3000")).status);
    expect(statuses).toContain(401);
    expect(statuses.at(-1)).toBe(429);
    expect(
      (await state.db.query("SELECT id FROM wax.rate_limits")).rows.length,
    ).toBeGreaterThan(0);
  });
});

describe("request and configuration boundaries", () => {
  it("rejects cross-site mutation before executing its action", async () => {
    const action = vi.fn(async () => ({ ok: true }));
    const response = await handleApi(
      new Request("https://wax.example/api/records", {
        method: "POST",
        headers: { Origin: "https://evil.example" },
      }),
      action,
    );
    expect(response.status).toBe(403);
    expect(action).not.toHaveBeenCalled();
  });
  it("accepts a psql-wrapped URI without evaluating command text", () => {
    expect(
      parseConnectionString(
        "psql 'postgresql://example:secret@localhost/wax?sslmode=require' --set ON_ERROR_STOP=on",
      ),
    ).toBe("postgresql://example:secret@localhost/wax?sslmode=require");
    expect(() => parseConnectionString("https://example.com")).toThrow();
  });
  it("refuses production credentials in preview and local processes before connecting", () => {
    const production = "postgresql://neondb_owner:fixture@localhost/neondb";
    const preview = "postgresql://wax_preview:fixture@localhost/wax_preview";
    const development =
      "postgresql://wax_development:fixture@localhost/wax_development";
    expect(
      databaseUrl({ VERCEL_ENV: "production", neon_connect: production }),
    ).toBe(production);
    expect(
      databaseUrl({
        VERCEL_ENV: "preview",
        DATABASE_URL: preview,
        neon_connect: production,
      }),
    ).toBe(preview);
    expect(
      databaseUrl({ DATABASE_URL: development, neon_connect: production }),
    ).toBe(development);
    for (const environment of [
      { VERCEL_ENV: "preview", neon_connect: production },
      { VERCEL_ENV: "preview", DATABASE_URL: production },
      {
        VERCEL_ENV: "preview",
        DATABASE_URL: "postgresql://neondb_owner:fixture@localhost/wax_preview",
      },
      {
        VERCEL_ENV: "preview",
        DATABASE_URL:
          "postgresql://wax_preview:fixture@localhost/wax_preview?user=neondb_owner",
      },
      { DATABASE_URL: production },
      { neon_connect: production },
    ])
      expect(() => databaseUrl(environment)).toThrow();
  });
});
