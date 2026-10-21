import pg from "pg";

const raw = process.env.DATABASE_URL || process.env.neon_connect || "";
const connectionString = raw.match(/postgres(?:ql)?:\/\/[^\s'"`]+/)?.[0];
if (!connectionString) throw new Error("Production connection is missing");
const configuration = process.env.WAX_ENV_PROVISIONING;
if (!configuration) {
  console.log("Nonproduction database provisioning not requested.");
  process.exit(0);
}
const targets = JSON.parse(configuration);
const pool = new pg.Pool({ connectionString, max: 1 });
let stage = "inspect-permissions";
try {
  const permission = await pool.query(
    "SELECT current_user AS role, rolcreatedb, rolcreaterole FROM pg_roles WHERE rolname=current_user",
  );
  console.log(JSON.stringify({ provisioningPermissions: permission.rows[0] }));
  for (const target of targets) {
    if (
      !/^wax_(development|preview)$/.test(target.name) ||
      !/^[a-f0-9]{64}$/.test(target.password)
    )
      throw new Error("Invalid provisioning configuration");
    const role = await pool.query("SELECT 1 FROM pg_roles WHERE rolname = $1", [
      target.name,
    ]);
    stage = `create-role:${target.name}`;
    if (!role.rowCount)
      await pool.query(
        `CREATE ROLE "${target.name}" LOGIN PASSWORD '${target.password}' NOSUPERUSER NOCREATEDB NOCREATEROLE`,
      );
    const database = await pool.query(
      "SELECT pg_get_userbyid(datdba) AS owner FROM pg_database WHERE datname = $1",
      [target.name],
    );
    stage = `create-database:${target.name}`;
    if (!database.rowCount) {
      // Postgres 16+ requires SET membership to assign a new database owner.
      const adminRole = permission.rows[0].role.replaceAll('"', '""');
      await pool.query(
        `GRANT "${target.name}" TO "${adminRole}" WITH SET TRUE`,
      );
      try {
        await pool.query(
          `CREATE DATABASE "${target.name}" OWNER "${target.name}"`,
        );
      } finally {
        await pool.query(
          `GRANT "${target.name}" TO "${adminRole}" WITH SET FALSE`,
        );
      }
    } else if (database.rows[0].owner !== target.name)
      throw new Error("Existing database ownership differs");
    const targetUrl = new URL(connectionString);
    targetUrl.username = target.name;
    targetUrl.password = target.password;
    targetUrl.pathname = `/${target.name}`;
    const targetPool = new pg.Pool({
      connectionString: targetUrl.href,
      max: 1,
    });
    try {
      stage = `connect:${target.name}`;
      await targetPool.query("SELECT 1");
      console.log(
        JSON.stringify({
          environmentDatabase: target.name,
          status: "connected",
          hostname: targetUrl.hostname,
          port: targetUrl.port || "5432",
        }),
      );
    } finally {
      await targetPool.end();
    }
  }
} catch (error) {
  console.error(
    JSON.stringify({
      provisioning: "failed",
      stage,
      errorType: error?.name,
      code: error?.code,
      ...(error?.code === "42501" ? { reason: error.message } : {}),
    }),
  );
  process.exitCode = 1;
} finally {
  await pool.end();
}
