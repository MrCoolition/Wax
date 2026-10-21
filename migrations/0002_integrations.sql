CREATE TABLE wax.photos (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  owner_id text NOT NULL REFERENCES wax.collectors(id) ON DELETE CASCADE,
  copy_id uuid REFERENCES wax.copies(id) ON DELETE SET NULL,
  pathname text NOT NULL UNIQUE,
  content_type text NOT NULL CHECK(content_type IN ('image/jpeg','image/png','image/webp')),
  byte_size integer NOT NULL CHECK(byte_size BETWEEN 1 AND 3145728),
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX photos_owner_copy_idx ON wax.photos(owner_id,copy_id);
CREATE TABLE wax.wizard_proposals (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  owner_id text NOT NULL REFERENCES wax.collectors(id) ON DELETE CASCADE,
  summary text NOT NULL,
  changes jsonb NOT NULL CHECK(jsonb_typeof(changes)='array'),
  status text NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','approved','rejected')),
  created_at timestamptz NOT NULL DEFAULT now(),
  decided_at timestamptz
);
CREATE INDEX wizard_proposals_owner_idx ON wax.wizard_proposals(owner_id,created_at DESC);
CREATE TABLE wax.catalog_cache (
  cache_key text PRIMARY KEY,
  payload jsonb NOT NULL,
  fetched_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX catalog_cache_fetched_idx ON wax.catalog_cache(fetched_at);
CREATE TABLE wax.integration_limits (
  owner_id text NOT NULL REFERENCES wax.collectors(id) ON DELETE CASCADE,
  operation text NOT NULL,
  window_start timestamptz NOT NULL,
  count integer NOT NULL,
  PRIMARY KEY(owner_id,operation)
);
