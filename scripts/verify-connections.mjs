import { neon } from "@neondatabase/serverless";
import { databaseUrl } from "../src/server/database-url.ts";
try {
  const sql = neon(databaseUrl());
  const result = await sql`SELECT 1 AS connected`;
  if (result[0]?.connected !== 1) throw new Error("Unexpected response");
  const schemas =
    await sql`SELECT schema_name FROM information_schema.schemata WHERE schema_name IN ('vinyl', 'wax')`;
  console.log(
    JSON.stringify({
      database: "connected",
      existingAppSchemas: schemas.map((x) => x.schema_name),
      openAIConfigured: Boolean(
        process.env.OPENAI_API_KEY || process.env.chaz_gpt,
      ),
      blobConfigured: Boolean(
        process.env.BLOB_READ_WRITE_TOKEN || process.env.BLOB_STORE_ID,
      ),
    }),
  );
} catch (error) {
  console.error(
    JSON.stringify({
      database: "unavailable",
      errorType: error?.name || "Error",
      code: /^[A-Z0-9_]{2,40}$/.test(error?.code || "")
        ? error.code
        : undefined,
    }),
  );
  process.exit(1);
}
