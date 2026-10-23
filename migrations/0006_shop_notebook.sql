-- Collector-private shop ratings and staff notes. Never part of discovery caches.
CREATE TABLE wax.shop_notebook (
  owner_id text NOT NULL REFERENCES wax.collectors(id) ON DELETE CASCADE,
  shop_id text NOT NULL CHECK(shop_id ~ '^(node|way|relation)/[1-9][0-9]*$'),
  shop jsonb NOT NULL CHECK(jsonb_typeof(shop) = 'object' AND shop->>'id' = shop_id),
  rating integer CHECK(rating BETWEEN 1 AND 5),
  employees jsonb NOT NULL DEFAULT '[]'::jsonb
    CHECK(jsonb_typeof(employees) = 'array' AND jsonb_array_length(employees) <= 30),
  version integer NOT NULL DEFAULT 1 CHECK(version > 0),
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY(owner_id, shop_id)
);
CREATE INDEX shop_notebook_owner_updated_idx ON wax.shop_notebook(owner_id, updated_at DESC);
