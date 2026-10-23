-- Verified directory locations can carry the same private ratings and staff notes.
-- Existing OSM identities, ownership, versions and saved data remain unchanged.
ALTER TABLE wax.shop_notebook DROP CONSTRAINT shop_notebook_shop_id_check;
ALTER TABLE wax.shop_notebook ADD CONSTRAINT shop_notebook_shop_id_check
  CHECK (shop_id ~ '^(node|way|relation)/[1-9][0-9]*$'
    OR shop_id ~ '^directory/[a-z0-9]+(-[a-z0-9]+)*$' AND length(shop_id) <= 90);
