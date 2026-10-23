import { z } from "zod";
import { fetchOriginalYear } from "./catalog";
import { getPool } from "./db";

export const ORIGINAL_YEAR_BATCH_SIZE = 8;
export const originalYearRequestSchema = z.object({}).strict();

export interface OriginalYearBatch {
  updated: number;
  processed: number;
  remaining: number;
  transientFailure: boolean;
}

type LegacyCopy = {
  id: string;
  version: number;
  discogs_id: number | null;
};

/** Called only for the authenticated collector; library GET never calls providers. */
export async function enrichOriginalYears(
  ownerId: string,
): Promise<OriginalYearBatch> {
  const started = Date.now();
  const db = getPool();
  const rows = (
    await db.query<LegacyCopy>(
      `SELECT id,version,(metadata->>'discogsId')::bigint AS discogs_id
     FROM wax.copies WHERE owner_id=$1 AND NOT (metadata ? 'originalYear')
     ORDER BY created_at,id LIMIT $2`,
      [ownerId, ORIGINAL_YEAR_BATCH_SIZE],
    )
  ).rows;
  let updated = 0;
  let processed = 0;
  let transientFailure = false;
  for (const copy of rows) {
    // Keep a bounded request even if a provider is slow. One lookup can take
    // up to two 15-second requests, leaving room within the route's limit.
    if (Date.now() - started > 20_000) break;
    const discogsId = copy.discogs_id === null ? null : Number(copy.discogs_id);
    let originalYear: number | null = null;
    try {
      if (discogsId !== null) {
        const resolution = await fetchOriginalYear(discogsId);
        if (!resolution.resolved) {
          transientFailure = true;
          break;
        }
        originalYear = resolution.originalYear;
      }
    } catch {
      transientFailure = true;
      break;
    }
    processed++;
    const changed = await db.query(
      `UPDATE wax.copies SET metadata=jsonb_set(metadata,'{originalYear}',$5::jsonb),
         version=version+1,updated_at=clock_timestamp()
       WHERE owner_id=$1 AND id=$2 AND version=$3
         AND (metadata->>'discogsId') IS NOT DISTINCT FROM $4::text
         AND NOT (metadata ? 'originalYear') RETURNING id`,
      [
        ownerId,
        copy.id,
        copy.version,
        discogsId === null ? null : String(discogsId),
        JSON.stringify(originalYear),
      ],
    );
    updated += changed.rows.length;
  }
  const remaining = (
    await db.query<{ count: number }>(
      `SELECT count(*)::int AS count FROM wax.copies WHERE owner_id=$1 AND NOT (metadata ? 'originalYear')`,
      [ownerId],
    )
  ).rows[0].count;
  return { updated, processed, remaining, transientFailure };
}
