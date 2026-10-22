-- Public map data only. No collector IDs or precise device coordinates are stored.
CREATE TABLE wax.shop_discovery_cache (
  cache_key text PRIMARY KEY,
  payload jsonb NOT NULL,
  expires_at timestamptz NOT NULL
);
CREATE INDEX shop_discovery_cache_expiry_idx ON wax.shop_discovery_cache(expires_at);

-- Atomic provider reservations apply across all Vercel function instances.
CREATE TABLE wax.shop_provider_limits (
  provider text PRIMARY KEY CHECK(provider='photon'),
  available_at timestamptz NOT NULL
);
