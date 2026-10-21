import { z } from "zod";
import type { CatalogCandidate } from "../lib/contracts";
import { getPool } from "./db";
import { HttpError } from "./http";

const optionalQuery = z.string().trim().max(200).optional();
export const catalogSearchSchema = z.object({
  query: optionalQuery,
  artist: optionalQuery,
  title: optionalQuery,
  catalogNumber: optionalQuery,
  barcode: z
    .string()
    .trim()
    .max(40)
    .regex(/^[0-9 -]*$/)
    .optional(),
  page: z.coerce.number().int().min(1).max(100).optional(),
});
export type CatalogSearch = z.infer<typeof catalogSearchSchema>;
export const releaseQuerySchema = z
  .object({ refresh: z.enum(["true", "false"]).optional() })
  .strict();
type JsonObject = Record<string, unknown>;
const object = (value: unknown): JsonObject =>
  value && typeof value === "object" && !Array.isArray(value)
    ? (value as JsonObject)
    : {};
const text = (value: unknown, max = 500) =>
  typeof value === "string" ? value.trim().slice(0, max) : "";
const array = (value: unknown): unknown[] =>
  Array.isArray(value) ? value : [];
const texts = (value: unknown) =>
  array(value)
    .map((v) => text(v))
    .filter(Boolean)
    .slice(0, 30);
export function safeDiscogsImage(value: unknown): string | null {
  if (typeof value !== "string" || value.length > 4000) return null;
  try {
    const url = new URL(value);
    return url.protocol === "https:" &&
      ["i.discogs.com", "img.discogs.com"].includes(url.hostname) &&
      !url.username &&
      !url.password
      ? url.href
      : null;
  } catch {
    return null;
  }
}
function year(value: unknown): number | null {
  const n =
    typeof value === "number" || typeof value === "string"
      ? Number(value)
      : NaN;
  return Number.isInteger(n) && n >= 1850 && n <= 2200 ? n : null;
}
function id(value: unknown): number {
  const n = Number(value);
  if (!Number.isSafeInteger(n) || n <= 0)
    throw new HttpError(
      502,
      "Discogs returned an incomplete release. Try another candidate or use manual entry.",
    );
  return n;
}
export function normalizeSearchCandidate(value: unknown): CatalogCandidate {
  const raw = object(value);
  const releaseId = id(raw.id);
  const combined = text(raw.title);
  const split = combined.indexOf(" - ");
  return {
    discogsId: releaseId,
    artist: split >= 0 ? combined.slice(0, split) : "",
    title: split >= 0 ? combined.slice(split + 3) : combined,
    year: year(raw.year),
    genres: [...new Set([...texts(raw.genre), ...texts(raw.style)])],
    label: [...new Set(texts(raw.label))].join(" / "),
    catalogNumber: text(raw.catno),
    country: text(raw.country),
    format: texts(raw.format).join(", ") || "Vinyl",
    coverUrl: safeDiscogsImage(raw.cover_image) || safeDiscogsImage(raw.thumb),
    sourceUrl: `https://www.discogs.com/release/${releaseId}`,
  };
}
export function normalizeRelease(value: unknown): CatalogCandidate {
  const raw = object(value);
  const releaseId = id(raw.id);
  const labels = array(raw.labels).map(object);
  const images = array(raw.images).map(object);
  const primary = images.find((image) => image.type === "primary") || images[0];
  const formats = array(raw.formats)
    .map(object)
    .map((format) =>
      [text(format.name), ...texts(format.descriptions), text(format.text)]
        .filter(Boolean)
        .join(", "),
    );
  return {
    discogsId: releaseId,
    artist: array(raw.artists)
      .map(object)
      .map((artist) => text(artist.name).replace(/ \(\d+\)$/, ""))
      .filter(Boolean)
      .join(" & "),
    title: text(raw.title),
    year: year(raw.year),
    genres: [...new Set([...texts(raw.genres), ...texts(raw.styles)])],
    label: [
      ...new Set(labels.map((label) => text(label.name)).filter(Boolean)),
    ].join(" / "),
    catalogNumber: [
      ...new Set(labels.map((label) => text(label.catno)).filter(Boolean)),
    ].join(" / "),
    country: text(raw.country),
    format: formats.join(" / ") || "Vinyl",
    coverUrl:
      safeDiscogsImage(primary?.uri) || safeDiscogsImage(primary?.uri150),
    sourceUrl: `https://www.discogs.com/release/${releaseId}`,
  };
}

async function cached<T>(
  key: string,
  fetchValue: () => Promise<T>,
  refresh = false,
): Promise<T> {
  // Never serve provider data older than six hours. Confirmed copy fields live separately.
  await getPool().query(
    "DELETE FROM wax.catalog_cache WHERE fetched_at < now()-interval '6 hours'",
  );
  if (!refresh) {
    const found = await getPool().query<{ payload: T }>(
      "SELECT payload FROM wax.catalog_cache WHERE cache_key=$1",
      [key],
    );
    if (found.rows[0]) return found.rows[0].payload;
  }
  const payload = await fetchValue();
  await getPool().query(
    `INSERT INTO wax.catalog_cache(cache_key,payload) VALUES($1,$2::jsonb)
    ON CONFLICT(cache_key) DO UPDATE SET payload=excluded.payload,fetched_at=now()`,
    [key, JSON.stringify(payload)],
  );
  return payload;
}

async function discogs(path: string): Promise<unknown> {
  const token = process.env.DISCOGS_TOKEN;
  try {
    const response = await fetch(`https://api.discogs.com${path}`, {
      headers: {
        "User-Agent": "WAX/1.0 (+https://github.com/MrCoolition/Wax)",
        Accept: "application/json",
        ...(token ? { Authorization: `Discogs token=${token}` } : {}),
      },
      signal: AbortSignal.timeout(15_000),
      cache: "no-store",
    });
    if (response.status === 429)
      throw new HttpError(
        429,
        "Discogs is taking a breather. Your entries are saved on this screen; try again shortly.",
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
        "That Discogs release is no longer available. Try another match or manual entry.",
      );
    if (!response.ok)
      throw new HttpError(
        503,
        "Discogs lookup is unavailable right now. You can retry or add this record manually.",
      );
    return await response.json();
  } catch (error) {
    if (error instanceof HttpError) throw error;
    throw new HttpError(
      503,
      "Discogs did not respond. Try again or keep digging with manual entry.",
    );
  }
}

export async function searchCatalog(
  input: CatalogSearch,
): Promise<CatalogCandidate[]> {
  const params = catalogSearchSchema.parse(input);
  if (
    ![
      params.query,
      params.artist,
      params.title,
      params.catalogNumber,
      params.barcode,
    ].some(Boolean)
  )
    return [];
  const query = new URLSearchParams({
    type: "release",
    format: "Vinyl",
    per_page: "20",
    page: String(params.page || 1),
  });
  for (const [key, value] of [
    ["q", params.query],
    ["artist", params.artist],
    ["release_title", params.title],
    ["catno", params.catalogNumber],
    ["barcode", params.barcode?.replace(/[ -]/g, "")],
  ]) {
    if (key && value) query.set(key, value);
  }
  return cached(`search:${query}`, async () => {
    const response = object(await discogs(`/database/search?${query}`));
    return array(response.results).slice(0, 20).map(normalizeSearchCandidate);
  });
}

export async function fetchRelease(
  releaseId: number,
  options: { refresh?: boolean } = {},
): Promise<CatalogCandidate> {
  z.number().int().positive().max(Number.MAX_SAFE_INTEGER).parse(releaseId);
  const { refresh = false } = z
    .object({ refresh: z.boolean().optional() })
    .strict()
    .parse(options);
  return cached(
    `release:${releaseId}`,
    async () => normalizeRelease(await discogs(`/releases/${releaseId}`)),
    refresh,
  );
}
