import { createHash } from "node:crypto";
import type { PoolClient } from "pg";
import { z } from "zod";

export async function seedOwnerInvitation(
  client: Pick<PoolClient, "query">,
  rawEmail: string,
  token: string,
) {
  const email = z.string().trim().email().parse(rawEmail).toLowerCase();
  if (token.length < 32)
    throw new Error("Bootstrap token must contain at least 32 characters.");
  const hash = createHash("sha256").update(token).digest("hex");
  await client.query(
    `INSERT INTO wax.invitations(token_hash,email,kind,expires_at)
    SELECT $1,$2,'owner',now()+interval '7 days' WHERE NOT EXISTS(SELECT 1 FROM wax.app_owner)
    ON CONFLICT(kind) WHERE kind='owner' DO UPDATE
      SET token_hash=EXCLUDED.token_hash,email=EXCLUDED.email,expires_at=EXCLUDED.expires_at,revoked_at=NULL
      WHERE wax.invitations.consumed_at IS NULL AND wax.invitations.token_hash<>EXCLUDED.token_hash
        AND NOT EXISTS(SELECT 1 FROM wax.app_owner)`,
    [hash, email],
  );
}
