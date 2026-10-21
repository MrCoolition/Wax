import { neon } from "@neondatabase/serverless";
import { databaseUrl } from "@/server/database-url";
export const dynamic = "force-dynamic";
export async function GET() {
  try {
    const sql = neon(databaseUrl());
    await sql`SELECT 1`;
    return Response.json(
      { status: "ok" },
      { headers: { "Cache-Control": "no-store" } },
    );
  } catch {
    return Response.json(
      { status: "unavailable" },
      { status: 503, headers: { "Cache-Control": "no-store" } },
    );
  }
}
