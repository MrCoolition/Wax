import { readdir, readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { createHash } from "node:crypto";
import { Pool } from "pg";
import { databaseUrl } from "../src/server/database-url";
import { seedOwnerInvitation } from "../src/server/bootstrap";

const connectionString = databaseUrl();
const pool = new Pool({ connectionString, max: 1 });
const client = await pool.connect();
try {
  // A transaction-scoped lock works on Neon's pooled endpoint as well as direct connections.
  await client.query("BEGIN");
  await client.query(
    "SELECT pg_advisory_xact_lock(hashtextextended('wax:migrations',0))",
  );
  await client.query("CREATE SCHEMA IF NOT EXISTS wax");
  await client.query(
    "CREATE TABLE IF NOT EXISTS wax.schema_migrations(name text PRIMARY KEY, checksum text NOT NULL, applied_at timestamptz NOT NULL DEFAULT now())",
  );
  const directory = resolve("migrations");
  const files = (await readdir(directory))
    .filter((file) => /^\d+.*\.sql$/.test(file))
    .sort();
  for (const name of files) {
    const sql = await readFile(resolve(directory, name), "utf8");
    const checksum = createHash("sha256").update(sql).digest("hex");
    const prior = (
      await client.query(
        "SELECT checksum FROM wax.schema_migrations WHERE name=$1",
        [name],
      )
    ).rows[0];
    if (prior) {
      if (prior.checksum !== checksum)
        throw new Error(`Applied migration was modified: ${name}`);
      continue;
    }
    await client.query(sql);
    await client.query(
      "INSERT INTO wax.schema_migrations(name,checksum) VALUES($1,$2)",
      [name, checksum],
    );
    console.log(`Applied ${name}`);
  }
  const ownerEmail = process.env.WAX_OWNER_EMAIL?.trim().toLowerCase();
  const bootstrap = process.env.WAX_BOOTSTRAP_TOKEN;
  if (ownerEmail && bootstrap) {
    await seedOwnerInvitation(client, ownerEmail, bootstrap);
    console.log("Owner invitation checked.");
  }
  await client.query("COMMIT");
  console.log("WAX migrations complete.");
} catch (error) {
  await client.query("ROLLBACK");
  throw error;
} finally {
  client.release();
  await pool.end();
}
