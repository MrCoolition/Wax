import { Pool, type PoolClient } from "pg";
import { drizzle } from "drizzle-orm/node-postgres";
import * as schema from "./schema";
import { databaseUrl } from "./database-url";
export { parseConnectionString } from "./database-url";

let pool: Pool | undefined;
let database: ReturnType<typeof drizzle<typeof schema>> | undefined;
export function getPool(): Pool {
  return (pool ??= new Pool({
    connectionString: databaseUrl(),
    max: 5,
    idleTimeoutMillis: 10_000,
    connectionTimeoutMillis: 10_000,
  }));
}
export function getDb() {
  return (database ??= drizzle(getPool(), { schema }));
}
export async function withTransaction<T>(
  action: (client: PoolClient) => Promise<T>,
): Promise<T> {
  const client = await getPool().connect();
  try {
    await client.query("BEGIN");
    const result = await action(client);
    await client.query("COMMIT");
    return result;
  } catch (error) {
    await client.query("ROLLBACK");
    throw error;
  } finally {
    client.release();
  }
}
