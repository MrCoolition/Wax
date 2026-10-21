CREATE TABLE wax.owner_recovery_tokens (
  token_hash text PRIMARY KEY CHECK (length(token_hash) = 64),
  user_id text NOT NULL REFERENCES wax.users(id) ON DELETE CASCADE,
  created_at timestamptz NOT NULL DEFAULT now(),
  expires_at timestamptz NOT NULL,
  consumed_at timestamptz,
  revoked_at timestamptz
);
CREATE INDEX owner_recovery_user_idx ON wax.owner_recovery_tokens(user_id);
