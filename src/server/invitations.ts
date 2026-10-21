import { createHash, randomBytes } from "node:crypto";
import { z } from "zod";
import { drizzle } from "drizzle-orm/node-postgres";
import type { Collector } from "@/lib/contracts";
import { getPool, withTransaction } from "./db";
import { authBaseUrl, createWaxAuth, normalizeEmail } from "./auth";
import * as schema from "./schema";
import { HttpError } from "./http";
import { getCollector } from "./collection";

export const tokenDigest = (token: string) =>
  createHash("sha256").update(token).digest("hex");
const tokenSchema = z.string().min(32).max(256);
const asDate = (v: Date | string | null) =>
  v instanceof Date ? v.toISOString() : v;
function ownerOnly(collector: Collector) {
  if (!collector.isOwner)
    throw new HttpError(403, "Only the WAX owner can manage invitations.");
}
export async function listInvitations(collector: Collector) {
  ownerOnly(collector);
  const { rows } = await getPool().query(
    `SELECT id,email,kind,expires_at,consumed_at,revoked_at FROM wax.invitations WHERE kind='collector' ORDER BY created_at DESC LIMIT 200`,
  );
  return {
    invitations: rows.map((r) => ({
      id: r.id,
      email: r.email,
      kind: r.kind,
      expiresAt: asDate(r.expires_at),
      consumedAt: asDate(r.consumed_at),
      revokedAt: asDate(r.revoked_at),
    })),
  };
}
export async function createInvitation(collector: Collector, email: string) {
  ownerOnly(collector);
  const normalized = normalizeEmail(z.string().email().parse(email));
  const token = randomBytes(32).toString("base64url");
  const expiresAt = new Date(Date.now() + 7 * 24 * 60 * 60 * 1000);
  const id = await withTransaction(async (client) => {
    await client.query("SELECT pg_advisory_xact_lock(hashtextextended($1,0))", [
      `invite:${normalized}`,
    ]);
    if (
      (
        await client.query("SELECT id FROM wax.users WHERE lower(email)=$1", [
          normalized,
        ])
      ).rows.length
    )
      throw new HttpError(409, "That collector already has a WAX account.");
    await client.query(
      `UPDATE wax.invitations SET revoked_at=now() WHERE lower(email)=$1 AND kind='collector' AND consumed_at IS NULL AND revoked_at IS NULL`,
      [normalized],
    );
    return (
      await client.query(
        `INSERT INTO wax.invitations(token_hash,email,kind,created_by,expires_at) VALUES($1,$2,'collector',$3,$4) RETURNING id`,
        [tokenDigest(token), normalized, collector.id, expiresAt],
      )
    ).rows[0].id;
  });
  return {
    id,
    email: normalized,
    url: `${authBaseUrl()}/join#token=${encodeURIComponent(token)}`,
    expiresAt: expiresAt.toISOString(),
  };
}
export async function revokeInvitation(collector: Collector, id: string) {
  ownerOnly(collector);
  z.string().uuid().parse(id);
  const { rows } = await getPool().query(
    `UPDATE wax.invitations SET revoked_at=now() WHERE id=$1 AND kind='collector' AND consumed_at IS NULL AND revoked_at IS NULL RETURNING id`,
    [id],
  );
  if (!rows.length) throw new HttpError(404, "Active invitation not found.");
  return { ok: true };
}
export async function inspectInvitation(rawToken: string) {
  const token = tokenSchema.parse(rawToken);
  const { rows } = await getPool().query(
    "SELECT email,kind FROM wax.invitations WHERE token_hash=$1 AND expires_at>now() AND revoked_at IS NULL AND consumed_at IS NULL",
    [tokenDigest(token)],
  );
  if (!rows[0])
    throw new HttpError(
      404,
      "This invitation has expired or has already been used.",
    );
  return { email: rows[0].email, kind: rows[0].kind };
}
export async function limitRegistration(request: Request) {
  // Vercel overwrites x-real-ip; this value is only used for throttling, never identity.
  const ip = request.headers.get("x-real-ip") || "local";
  const key = tokenDigest(`registration:${ip}`);
  const { rows } = await getPool().query(
    `INSERT INTO wax.registration_limits(key,count,window_start) VALUES($1,1,now())
    ON CONFLICT(key) DO UPDATE SET count=CASE WHEN wax.registration_limits.window_start<now()-interval '1 minute' THEN 1 ELSE wax.registration_limits.count+1 END,
    window_start=CASE WHEN wax.registration_limits.window_start<now()-interval '1 minute' THEN now() ELSE wax.registration_limits.window_start END RETURNING count`,
    [key],
  );
  if (rows[0].count > 10)
    throw new HttpError(
      429,
      "Too many attempts. Please wait a minute and try again.",
    );
}
export async function acceptInvitation(raw: unknown) {
  const input = z
    .object({
      token: tokenSchema,
      name: z.string().trim().min(1).max(100),
      email: z.string().email(),
      password: z.string().min(12).max(128),
    })
    .strict()
    .parse(raw);
  const email = normalizeEmail(input.email);
  const userId = await withTransaction(async (client) => {
    const { rows } = await client.query(
      `UPDATE wax.invitations SET consumed_at=now() WHERE token_hash=$1 AND lower(email)=$2 AND consumed_at IS NULL AND revoked_at IS NULL AND expires_at>now() RETURNING id,kind`,
      [tokenDigest(input.token), email],
    );
    const invite = rows[0];
    if (!invite)
      throw new HttpError(
        400,
        "This invitation is invalid, expired, or already used.",
      );
    if (
      (
        await client.query("SELECT id FROM wax.users WHERE lower(email)=$1", [
          email,
        ])
      ).rows.length
    )
      throw new HttpError(409, "This account already exists. Sign in instead.");
    const registration = createWaxAuth(drizzle(client, { schema }), email);
    const result = await registration.api.signUpEmail({
      body: { name: input.name, email, password: input.password },
    });
    const user = (
      await client.query(
        "SELECT id FROM wax.users WHERE id=$1 AND lower(email)=$2",
        [result.user.id, email],
      )
    ).rows[0];
    if (!user) throw new HttpError(409, "This account could not be created.");
    await client.query("INSERT INTO wax.collectors(id) VALUES($1)", [user.id]);
    if (invite.kind === "owner")
      await client.query(
        "INSERT INTO wax.app_owner(singleton,user_id) VALUES(true,$1)",
        [user.id],
      );
    await client.query(
      "UPDATE wax.invitations SET redeemed_user_id=$2 WHERE id=$1",
      [invite.id, user.id],
    );
    return user.id as string;
  });
  return { collector: await getCollector(userId), ok: true };
}
