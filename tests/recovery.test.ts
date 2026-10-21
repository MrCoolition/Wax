import {
  beforeAll,
  beforeEach,
  afterAll,
  describe,
  expect,
  it,
  vi,
} from "vitest";
import { readdir, readFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { PGlite, type PGliteInterface } from "@electric-sql/pglite";
import type { Pool, PoolClient } from "pg";
import { drizzle } from "drizzle-orm/node-postgres";
import { hashPassword, verifyPassword } from "better-auth/crypto";
import * as schema from "../src/server/schema";

const state = vi.hoisted(() => ({
  db: null as unknown as PGlite,
  pool: null as unknown as Pool,
  failSessionDelete: false,
}));
vi.mock("next/headers", () => ({ headers: async () => new Headers() }));
function facade(database: Pick<PGliteInterface, "query">) {
  return {
    async query(
      query: string | { text: string; rowMode?: "array" },
      parameters?: unknown[],
    ) {
      const sql = typeof query === "string" ? query : query.text;
      if (state.failSessionDelete && sql.startsWith("DELETE FROM wax.sessions"))
        throw new Error("Simulated session-store failure");
      const result = await database.query(sql, parameters, {
        rowMode:
          typeof query === "string" ? "object" : (query.rowMode ?? "object"),
      });
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
import {
  seedOwnerRecovery,
  inspectRecovery,
  resetPassword,
} from "../src/server/recovery";
import { createWaxAuth } from "../src/server/auth";
import { POST } from "../src/app/api/recovery/route";
import { POST as inspectRoute } from "../src/app/api/recovery/inspect/route";

const email = "owner@example.com";
const token = "first-owner-recovery-token".repeat(3);
const nextToken = "next-owner-recovery-token".repeat(3);
const oldPassword = "Old-owner-password-123";
const newPassword = "New-owner-password-456";
const digest = (value: string) =>
  createHash("sha256").update(value).digest("hex");
let originalHash: string;
const seed = (value = token, address = email) =>
  state.db.transaction((tx) => seedOwnerRecovery(facade(tx), address, value));

beforeAll(async () => {
  process.env.BETTER_AUTH_SECRET =
    "test-only-auth-secret-not-used-outside-fixtures-123456";
  process.env.BETTER_AUTH_URL = "http://localhost:3000";
  originalHash = await hashPassword(oldPassword);
  state.db = new PGlite({
    parsers: { 1184: (v) => v, 1114: (v) => v, 1082: (v) => v },
  });
  await state.db.waitReady;
  const folder = new URL("../migrations/", import.meta.url);
  for (const name of (await readdir(folder))
    .filter((name) => name.endsWith(".sql"))
    .sort())
    await state.db.exec(await readFile(new URL(name, folder), "utf8"));
  state.pool = facade(state.db) as unknown as Pool;
});
beforeEach(async () => {
  state.failSessionDelete = false;
  await state.db.exec(
    "TRUNCATE wax.users,wax.releases,wax.registration_limits,wax.rate_limits CASCADE",
  );
  for (const id of ["owner", "collector"]) {
    await state.db.query(
      "INSERT INTO wax.users(id,name,email) VALUES($1,$1,$2)",
      [id, `${id}@example.com`],
    );
    await state.db.query("INSERT INTO wax.collectors(id) VALUES($1)", [id]);
    await state.db.query(
      "INSERT INTO wax.accounts(id,account_id,provider_id,user_id,password) VALUES($1,$1,'credential',$1,$2)",
      [id, originalHash],
    );
    await state.db.query(
      "INSERT INTO wax.sessions(id,token,user_id,expires_at) VALUES($1,$1,$1,now()+interval '1 day')",
      [id],
    );
  }
  await state.db.exec("INSERT INTO wax.app_owner(user_id) VALUES('owner')");
});
afterAll(async () => {
  await state.db.close();
});

describe("admin-issued owner password recovery", () => {
  it("issues only for the established owner matching the configured email, and stores a digest", async () => {
    expect(await seed(token, "collector@example.com")).toEqual({
      ownerExists: false,
      recoveryAvailable: false,
    });
    expect(
      (await state.db.query("SELECT * FROM wax.owner_recovery_tokens")).rows,
    ).toHaveLength(0);
    expect(await seed(token, " OWNER@EXAMPLE.COM ")).toEqual({
      ownerExists: true,
      recoveryAvailable: true,
    });
    const row = (
      await state.db.query<{ token_hash: string; hours: number }>(
        "SELECT token_hash,extract(epoch from (expires_at-created_at))/3600 AS hours FROM wax.owner_recovery_tokens",
      )
    ).rows[0];
    expect(row.token_hash).toBe(digest(token));
    expect(Number(row.hours)).toBe(24);
    await state.db.exec("DELETE FROM wax.app_owner");
    expect(await seed(nextToken)).toEqual({
      ownerExists: false,
      recoveryAvailable: false,
    });
    await expect(inspectRecovery(token)).rejects.toMatchObject({ status: 404 });
  });

  it("changes the existing password, revokes owner sessions, preserves records, and allows Better Auth sign-in", async () => {
    const release = (
      await state.db.query<{ id: string }>(
        "INSERT INTO wax.releases(metadata) VALUES('{}') RETURNING id",
      )
    ).rows[0].id;
    await state.db.query(
      "INSERT INTO wax.copies(owner_id,release_id,metadata,notes) VALUES('owner',$1,'{}','Keep this collection')",
      [release],
    );
    const before = (await state.db.query("SELECT * FROM wax.copies")).rows;
    await seed();
    expect(await inspectRecovery(token)).toEqual({ email });
    expect(await resetPassword({ token, password: newPassword })).toEqual({
      ok: true,
    });
    const password = (
      await state.db.query<{ password: string }>(
        "SELECT password FROM wax.accounts WHERE user_id='owner'",
      )
    ).rows[0].password;
    expect(
      await verifyPassword({ hash: password, password: newPassword }),
    ).toBe(true);
    expect(
      await verifyPassword({ hash: password, password: oldPassword }),
    ).toBe(false);
    expect(
      (await state.db.query("SELECT user_id FROM wax.sessions")).rows,
    ).toEqual([{ user_id: "collector" }]);
    expect((await state.db.query("SELECT * FROM wax.copies")).rows).toEqual(
      before,
    );
    expect((await state.db.query("SELECT * FROM wax.app_owner")).rows).toEqual([
      { singleton: true, user_id: "owner" },
    ]);
    await expect(inspectRecovery(token)).rejects.toMatchObject({ status: 404 });
    await expect(
      resetPassword({ token, password: oldPassword }),
    ).rejects.toMatchObject({ status: 404 });
    const auth = createWaxAuth(drizzle(state.pool, { schema }));
    const signedIn = await auth.api.signInEmail({
      body: { email, password: newPassword },
      asResponse: true,
    });
    expect(signedIn.status).toBe(200);
    expect(signedIn.headers.get("set-cookie")).toContain("wax.session_token");
  });

  it("does not extend expired links, and fresh rotation revokes previous links permanently", async () => {
    await seed();
    await state.db.exec(
      "UPDATE wax.owner_recovery_tokens SET expires_at=now()-interval '1 minute'",
    );
    const expiry = (
      await state.db.query("SELECT expires_at FROM wax.owner_recovery_tokens")
    ).rows;
    expect(await seed()).toEqual({
      ownerExists: true,
      recoveryAvailable: false,
    });
    expect(
      (await state.db.query("SELECT expires_at FROM wax.owner_recovery_tokens"))
        .rows,
    ).toEqual(expiry);
    expect(await seed(nextToken)).toEqual({
      ownerExists: true,
      recoveryAvailable: true,
    });
    await expect(inspectRecovery(token)).rejects.toMatchObject({ status: 404 });
    await seed();
    expect(await inspectRecovery(nextToken)).toEqual({ email });
    await expect(inspectRecovery(token)).rejects.toMatchObject({ status: 404 });
  });

  it("does not revive consumed links on redeploy, while a new token can recover the same owner", async () => {
    await seed();
    await resetPassword({ token, password: newPassword });
    expect(await seed()).toEqual({
      ownerExists: true,
      recoveryAvailable: false,
    });
    expect(await seed(nextToken)).toEqual({
      ownerExists: true,
      recoveryAvailable: true,
    });
    expect(await inspectRecovery(nextToken)).toEqual({ email });
  });

  it("rolls back token consumption and password changes when revoking sessions fails", async () => {
    await seed();
    state.failSessionDelete = true;
    await expect(
      resetPassword({ token, password: newPassword }),
    ).rejects.toThrow("Simulated session-store failure");
    state.failSessionDelete = false;
    expect(await inspectRecovery(token)).toEqual({ email });
    expect(
      (
        await state.db.query<{ password: string }>(
          "SELECT password FROM wax.accounts WHERE user_id='owner'",
        )
      ).rows[0].password,
    ).toBe(originalHash);
    expect(
      (await state.db.query("SELECT * FROM wax.sessions WHERE user_id='owner'"))
        .rows,
    ).toHaveLength(1);
  });

  it("rejects missing credential accounts and non-owner recovery tokens without consuming them", async () => {
    await seed();
    await state.db.exec("DELETE FROM wax.accounts WHERE user_id='owner'");
    await expect(
      resetPassword({ token, password: newPassword }),
    ).rejects.toMatchObject({ status: 409 });
    expect(await inspectRecovery(token)).toEqual({ email });
    await state.db.query(
      "INSERT INTO wax.owner_recovery_tokens(token_hash,user_id,expires_at) VALUES($1,'collector',now()+interval '1 day')",
      [digest(nextToken)],
    );
    await expect(inspectRecovery(nextToken)).rejects.toMatchObject({
      status: 404,
    });
    await expect(
      resetPassword({ token: nextToken, password: newPassword }),
    ).rejects.toMatchObject({ status: 404 });
  });

  it("permits only one concurrent redemption and enforces password limits before changing data", async () => {
    await seed();
    for (const password of ["short", "x".repeat(129)])
      await expect(resetPassword({ token, password })).rejects.toHaveProperty(
        "name",
        "ZodError",
      );
    const results = await Promise.allSettled([
      resetPassword({ token, password: newPassword }),
      resetPassword({ token, password: oldPassword }),
    ]);
    expect(
      results.filter((result) => result.status === "fulfilled"),
    ).toHaveLength(1);
    expect(
      results.filter((result) => result.status === "rejected"),
    ).toHaveLength(1);
  });

  it("protects HTTP resets by origin and rate limit, and never consumes links during inspection", async () => {
    await seed();
    for (const origin of [undefined, "https://attacker.example"]) {
      const response = await POST(
        new Request("http://localhost:3000/api/recovery", {
          method: "POST",
          headers: origin ? { origin } : {},
          body: JSON.stringify({ token, password: newPassword }),
        }),
      );
      expect(response.status).toBe(403);
      expect(
        (
          await inspectRoute(
            new Request("http://localhost:3000/api/recovery/inspect", {
              method: "POST",
              headers: origin ? { origin } : {},
              body: JSON.stringify({ token }),
            }),
          )
        ).status,
      ).toBe(403);
    }
    const inspectionRequest = () =>
      new Request("http://localhost:3000/api/recovery/inspect", {
        method: "POST",
        headers: { origin: "http://localhost:3000" },
        body: JSON.stringify({ token }),
      });
    for (let count = 0; count < 10; count++) {
      const response = await inspectRoute(inspectionRequest());
      expect(response.status).toBe(200);
      expect(response.headers.get("cache-control")).toBe("private, no-store");
    }
    expect((await inspectRoute(inspectionRequest())).status).toBe(429);
    expect(await inspectRecovery(token)).toEqual({ email });
    const accepted = await POST(
      new Request("http://localhost:3000/api/recovery", {
        method: "POST",
        headers: {
          origin: "http://localhost:3000",
          "x-real-ip": "recovery-test",
        },
        body: JSON.stringify({ token, password: newPassword }),
      }),
    );
    expect(accepted.status).toBe(200);
    expect(await accepted.json()).toEqual({ ok: true });
  });
});
