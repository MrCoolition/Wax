-- Release rows identify releases; the expiring catalog_cache owns provider payloads.
-- Reviewed collection text remains on each copy and is never overwritten on refresh.
UPDATE wax.releases SET metadata='{}'::jsonb;
UPDATE wax.copies SET metadata=jsonb_set(metadata,'{coverUrl}','null'::jsonb)
  WHERE metadata->>'coverUrl' IS NOT NULL;
