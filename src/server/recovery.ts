import { createHash } from "node:crypto";
import { hashPassword } from "better-auth/crypto";
import type { PoolClient } from "pg";
import { z } from "zod";
import { getPool, withTransaction } from "./db";
import { HttpError } from "./http";

const tokenSchema = z.string().min(32).max(256);
const tokenDigest = (token: string) =>
  createHash("sha256").update(token).digest("hex");
const unavailable = "This recovery link has expired or has already been used.";

// The migration runner calls this inside its transaction. The lock also
// serializes issuance with redemption, so a rotation cannot race a reset.
export async function seedOwnerRecovery(
  client: Pick<PoolClient, "query">,
  rawEmail: string,
  rawToken: string,
) {
  const email = z.string().trim().email().parse(rawEmail).toLowerCase();
  const hash = tokenDigest(tokenSchema.parse(rawToken));
  await client.query(
    "SELECT pg_advisory_xact_lock(hashtextextended('wax:owner-recovery',0))",
  );
  const owner = (
    await client.query<{ id: string }>(
      `SELECT u.id FROM wax.app_owner o JOIN wax.users u ON u.id=o.user_id
       WHERE lower(u.email)=$1`,
      [email],
    )
  ).rows[0];
  if (!owner) return { ownerExists: false, recoveryAvailable: false };
  const prior = (
    await client.query<{ available: boolean }>(
      `SELECT user_id=$2 AND consumed_at IS NULL AND revoked_at IS NULL AND expires_at>now() AS available
       FROM wax.owner_recovery_tokens WHERE token_hash=$1`,
      [hash, owner.id],
    )
  ).rows[0];
  // A reused token must never extend its expiry or revive a consumed/revoked link.
  if (prior) return { ownerExists: true, recoveryAvailable: prior.available };
  await client.query(
    `UPDATE wax.owner_recovery_tokens SET revoked_at=now()
     WHERE user_id=$1 AND consumed_at IS NULL AND revoked_at IS NULL`,
    [owner.id],
  );
  await client.query(
    `INSERT INTO wax.owner_recovery_tokens(token_hash,user_id,expires_at)
     VALUES($1,$2,now()+interval '24 hours')`,
    [hash, owner.id],
  );
  return { ownerExists: true, recoveryAvailable: true };
}

export async function inspectRecovery(rawToken: string) {
  const hash = tokenDigest(tokenSchema.parse(rawToken));
  const { rows } = await getPool().query<{ email: string }>(
    `SELECT u.email FROM wax.owner_recovery_tokens r
     JOIN wax.app_owner o ON o.user_id=r.user_id JOIN wax.users u ON u.id=r.user_id
     WHERE r.token_hash=$1 AND r.expires_at>now() AND r.consumed_at IS NULL AND r.revoked_at IS NULL`,
    [hash],
  );
  if (!rows[0]) throw new HttpError(404, unavailable);
  return { email: rows[0].email };
}

export async function resetPassword(raw: unknown) {
  const input = z
    .object({ token: tokenSchema, password: z.string().min(12).max(128) })
    .strict()
    .parse(raw);
  // Reject invalid links before spending time hashing, then recheck under lock.
  await inspectRecovery(input.token);
  const password = await hashPassword(input.password);
  await withTransaction(async (client) => {
    await client.query(
      "SELECT pg_advisory_xact_lock(hashtextextended('wax:owner-recovery',0))",
    );
    const { rows } = await client.query<{ user_id: string }>(
      `UPDATE wax.owner_recovery_tokens r SET consumed_at=now() FROM wax.app_owner o
       WHERE r.token_hash=$1 AND r.user_id=o.user_id AND r.expires_at>now()
         AND r.consumed_at IS NULL AND r.revoked_at IS NULL RETURNING r.user_id`,
      [tokenDigest(input.token)],
    );
    if (!rows[0]) throw new HttpError(400, unavailable);
    const userId = rows[0].user_id;
    const updated = await client.query(
      `UPDATE wax.accounts SET password=$2,updated_at=now()
       WHERE user_id=$1 AND account_id=$1 AND provider_id='credential' RETURNING id`,
      [userId, password],
    );
    if (updated.rows.length !== 1)
      throw new HttpError(409, "This account could not be recovered.");
    await client.query("DELETE FROM wax.sessions WHERE user_id=$1", [userId]);
  });
  return { ok: true };
}
