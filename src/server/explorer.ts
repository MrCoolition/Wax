import { z } from "zod";
import type {
  ExplorerArtist,
  ExplorerDiscography,
  ExplorerRelease,
  ExplorerStart,
} from "@/lib/explorer";
import { getCopy } from "./collection";
import { getPool } from "./db";
import { HttpError } from "./http";
import {
  normalizeExplorerArtist,
  normalizeExplorerDiscography,
  normalizeExplorerRelease,
} from "./explorer-normalize";

export const explorerIdSchema = z
  .string()
  .max(16)
  .regex(/^[1-9]\d*$/)
  .transform(Number)
  .pipe(z.number().int().positive().max(Number.MAX_SAFE_INTEGER));
export const explorerPageSchema = z
  .object({ page: z.coerce.number().int().min(1).max(500).default(1) })
  .strict();
const pending = new Map<string, Promise<unknown>>();

async function provider(path: string): Promise<unknown> {
  try {
    const token = process.env.DISCOGS_TOKEN;
    const response = await fetch(`https://api.discogs.com${path}`, {
      headers: {
        "User-Agent": "WAX/1.0 (+https://github.com/MrCoolition/Wax)",
        Accept: "application/json",
        ...(token ? { Authorization: `Discogs token=${token}` } : {}),
      },
      cache: "no-store",
      redirect: "error",
      signal: AbortSignal.timeout(15_000),
    });
    if (response.status === 429)
      throw new HttpError(
        429,
        "Discogs is taking a breather. Your place in the tree is safe; try again shortly.",
        {
          retryAfter: Math.min(
            300,
            Math.max(1, Number(response.headers.get("retry-after")) || 60),
          ),
        },
      );
    if (response.status === 404)
      throw new HttpError(
        404,
        "This Discogs entry is no longer available. Try another branch.",
      );
    if (!response.ok)
      throw new HttpError(
        503,
        "Discogs could not load this branch. Please try again.",
      );
    const maximum = 2 * 1024 * 1024;
    if (Number(response.headers.get("content-length")) > maximum)
      throw new HttpError(502, "This Discogs entry is too large to explore.");
    const reader = response.body?.getReader();
    if (!reader) throw new HttpError(502, "Discogs returned an empty entry.");
    const chunks: Uint8Array[] = [];
    let bytes = 0;
    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        bytes += value.byteLength;
        if (bytes > maximum) {
          await reader.cancel();
          throw new HttpError(
            502,
            "This Discogs entry is too large to explore.",
          );
        }
        chunks.push(value);
      }
      return JSON.parse(Buffer.concat(chunks, bytes).toString("utf8"));
    } finally {
      reader.releaseLock();
    }
  } catch (error) {
    if (error instanceof HttpError) throw error;
    throw new HttpError(
      503,
      "Discogs did not respond. Your place in the tree is safe; try again.",
    );
  }
}

async function cached<T>(key: string, read: () => Promise<T>): Promise<T> {
  const inFlight = pending.get(key);
  if (inFlight) return inFlight as Promise<T>;
  const task = (async () => {
    await getPool().query(
      "DELETE FROM wax.catalog_cache WHERE fetched_at < now()-interval '6 hours'",
    );
    const found = await getPool().query<{ payload: T }>(
      "SELECT payload FROM wax.catalog_cache WHERE cache_key=$1 AND fetched_at>now()-interval '6 hours'",
      [key],
    );
    if (found.rows[0]) return found.rows[0].payload;
    const payload = await read();
    await getPool().query(
      "INSERT INTO wax.catalog_cache(cache_key,payload) VALUES($1,$2::jsonb) ON CONFLICT(cache_key) DO UPDATE SET payload=excluded.payload,fetched_at=now()",
      [key, JSON.stringify(payload)],
    );
    return payload;
  })();
  pending.set(key, task);
  try {
    return await task;
  } finally {
    pending.delete(key);
  }
}

const checkedId = (id: number) =>
  z.number().int().positive().max(Number.MAX_SAFE_INTEGER).parse(id);
export async function fetchExplorerRelease(
  id: number,
): Promise<ExplorerRelease> {
  checkedId(id);
  return cached(`explorer:v2:release:${id}`, async () =>
    normalizeExplorerRelease(await provider(`/releases/${id}`), id),
  );
}
export async function fetchExplorerArtist(id: number): Promise<ExplorerArtist> {
  checkedId(id);
  return cached(`explorer:v1:artist:${id}`, async () =>
    normalizeExplorerArtist(await provider(`/artists/${id}`), id),
  );
}
export async function fetchExplorerDiscography(
  id: number,
  page = 1,
): Promise<ExplorerDiscography> {
  checkedId(id);
  explorerPageSchema.parse({ page });
  return cached(`explorer:v1:artist-releases:${id}:${page}`, async () =>
    normalizeExplorerDiscography(
      await provider(
        `/artists/${id}/releases?per_page=20&page=${page}&sort=year&sort_order=asc`,
      ),
      id,
      page,
    ),
  );
}
export async function startExplorer(
  ownerId: string,
  copyId: string,
): Promise<ExplorerStart> {
  // Ownership is checked before any private copy metadata or provider lookup.
  const copy = await getCopy(ownerId, copyId);
  return {
    copyId: copy.id,
    artist: copy.artist,
    title: copy.title,
    discogsId: copy.discogsId,
    release: copy.discogsId ? await fetchExplorerRelease(copy.discogsId) : null,
  };
}
