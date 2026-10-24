import { z } from "zod";
import type { CatalogCandidate } from "../lib/contracts";
import type {
  AlbumSearchResult,
  CatalogAlbum,
  CatalogPage,
  PressingSearchResult,
} from "../lib/catalog-search";
import { getPool } from "./db";
import { HttpError } from "./http";

const optionalQuery = z.string().trim().max(200).optional();
const pageInput = z.coerce.number().int().min(1).max(100).optional();
const pressingFilters = {
  country: optionalQuery,
  label: optionalQuery,
  year: z.coerce.number().int().min(1850).max(2200).optional(),
};
export const catalogSearchSchema = z
  .object({
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
    ...pressingFilters,
    page: pageInput,
  })
  .strict();
export type CatalogSearch = z.infer<typeof catalogSearchSchema>;
export const albumSearchSchema = z
  .object({
    query: optionalQuery,
    artist: optionalQuery,
    title: optionalQuery,
    page: pageInput,
  })
  .strict();
export const albumVersionsSchema = z
  .object({
    ...pressingFilters,
    page: pageInput,
  })
  .strict();
export const catalogIdSchema = z
  .string()
  .regex(/^[1-9]\d{0,15}$/)
  .transform(Number)
  .pipe(z.number().int().positive().max(Number.MAX_SAFE_INTEGER));
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
      !url.port &&
      !url.username &&
      !url.password
      ? url.href
      : null;
  } catch {
    return null;
  }
}
// A same-origin resolver keeps saved records independent of expiring CDN links.
// It is generated from a provider ID, never a user-supplied URL.
export function catalogArtworkPath(releaseId: unknown): string | null {
  return typeof releaseId === "number" &&
    Number.isSafeInteger(releaseId) &&
    releaseId > 0
    ? `/api/catalog/${releaseId}/artwork`
    : null;
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
    originalYear: null,
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
    originalYear: null,
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
      redirect: "error",
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
    const maximum = 2 * 1024 * 1024;
    if (Number(response.headers.get("content-length")) > maximum)
      throw new HttpError(
        502,
        "Discogs returned too much data. Please try again.",
      );
    const reader = response.body?.getReader();
    if (!reader)
      throw new HttpError(502, "Discogs returned an empty response.");
    const chunks: Uint8Array[] = [];
    let length = 0;
    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        length += value.byteLength;
        if (length > maximum) {
          await reader.cancel();
          throw new HttpError(
            502,
            "Discogs returned too much data. Please try again.",
          );
        }
        chunks.push(value);
      }
      return JSON.parse(Buffer.concat(chunks, length).toString("utf8"));
    } finally {
      reader.releaseLock();
    }
  } catch (error) {
    if (error instanceof HttpError) throw error;
    throw new HttpError(
      503,
      "Discogs did not respond. Try again or keep digging with manual entry.",
    );
  }
}

type CachedRelease = {
  candidate: CatalogCandidate;
  masterId: number | null;
  invalidMasterId?: boolean;
};
type ReleaseResult = {
  candidate: CatalogCandidate;
  originalYearResolved: boolean;
};
const pendingReleases = new Map<string, Promise<ReleaseResult>>();
const pendingMasterYears = new Map<string, Promise<number | null>>();

export function normalizeMasterYear(
  value: unknown,
  expectedId: number,
): number | null {
  const master = object(value);
  if (typeof master.id !== "number" || master.id !== expectedId)
    throw new HttpError(
      502,
      "Discogs returned a different master release. Please try again.",
    );
  return year(master.year);
}

async function fetchMasterYear(
  masterId: number,
  refresh: boolean,
): Promise<number | null> {
  const key = `${masterId}:${refresh}`;
  const pending = pendingMasterYears.get(key);
  if (pending) return pending;
  const task = cached(
    `master-year:v1:${masterId}`,
    async () =>
      normalizeMasterYear(await discogs(`/masters/${masterId}`), masterId),
    refresh,
  );
  pendingMasterYears.set(key, task);
  try {
    return await task;
  } finally {
    if (pendingMasterYears.get(key) === task) pendingMasterYears.delete(key);
  }
}

const emptyPagination = (page = 1): CatalogPage => ({
  page,
  pages: 1,
  total: 0,
  hasMore: false,
});
function catalogPagination(value: unknown, expectedPage: number): CatalogPage {
  const raw = object(value);
  const page = Number(raw.page);
  const pages = Number(raw.pages);
  const total = Number(raw.items);
  if (
    page !== expectedPage ||
    !Number.isSafeInteger(pages) ||
    pages < 0 ||
    !Number.isSafeInteger(total) ||
    total < 0
  )
    throw new HttpError(
      502,
      "Discogs returned incomplete search results. Please try again.",
    );
  const boundedPages = Math.max(1, Math.min(100, pages));
  return { page, pages: boundedPages, total, hasMore: page < boundedPages };
}
const pendingDiscovery = new Map<string, Promise<unknown>>();
async function cachedDiscovery<T>(
  key: string,
  fetchValue: () => Promise<T>,
): Promise<T> {
  const existing = pendingDiscovery.get(key);
  if (existing) return existing as Promise<T>;
  const pending = cached(key, fetchValue);
  pendingDiscovery.set(key, pending);
  try {
    return await pending;
  } finally {
    if (pendingDiscovery.get(key) === pending) pendingDiscovery.delete(key);
  }
}
const normalizedWords = (value: string | undefined) =>
  (value || "")
    .normalize("NFKC")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim();

/** This documented album nickname is specific to the Beatles, not every White Album.
 * Canonical source: https://www.discogs.com/master/46402-The-Beatles-The-Beatles
 * Artist's own nickname reference: https://www.thebeatles.com/22-november-1968-white-album-released
 */
export function isBeatlesWhiteAlbum(input: {
  query?: string;
  artist?: string;
  title?: string;
}): boolean {
  const artist = normalizedWords(input.artist);
  const title = normalizedWords(input.title);
  if (artist || title)
    return (
      /^(the )?beatles$/.test(artist) &&
      /^(the )?white album$/.test(title) &&
      (!input.query ||
        /^(the )?beatles (the )?white album$/.test(
          normalizedWords(input.query),
        ))
    );
  return /^(the )?beatles (the )?white album$/.test(
    normalizedWords(input.query),
  );
}
const WHITE_ALBUM_NOTICE =
  "“The White Album” is The Beatles’ 1968 self-titled album. Choose your vinyl pressing next.";

function normalizeAlbumSearch(value: unknown): CatalogAlbum {
  const candidate = normalizeSearchCandidate(value);
  return {
    masterId: candidate.discogsId!,
    artist: candidate.artist,
    title: candidate.title,
    originalYear: candidate.year,
    coverUrl: candidate.coverUrl,
    sourceUrl: `https://www.discogs.com/master/${candidate.discogsId}`,
  };
}
async function fetchCatalogAlbum(masterId: number): Promise<CatalogAlbum> {
  z.number().int().positive().max(Number.MAX_SAFE_INTEGER).parse(masterId);
  return cachedDiscovery(`album:v1:${masterId}`, async () => {
    const raw = object(await discogs(`/masters/${masterId}`));
    const originalYear = normalizeMasterYear(raw, masterId);
    const metadata = normalizeRelease(raw);
    return {
      masterId,
      artist: metadata.artist,
      title: metadata.title,
      originalYear,
      coverUrl: metadata.coverUrl,
      sourceUrl: `https://www.discogs.com/master/${masterId}`,
    };
  });
}

export async function searchAlbums(
  input: z.infer<typeof albumSearchSchema>,
): Promise<AlbumSearchResult> {
  const params = albumSearchSchema.parse(input);
  const page = params.page || 1;
  if (![params.query, params.artist, params.title].some(Boolean))
    return { albums: [], pagination: emptyPagination(page) };
  if (isBeatlesWhiteAlbum(params)) {
    const album = await fetchCatalogAlbum(46402);
    if (
      album.artist !== "The Beatles" ||
      album.title !== "The Beatles" ||
      album.originalYear !== 1968
    )
      throw new HttpError(
        502,
        "The album listing changed. Try searching by artist and title.",
      );
    return {
      albums: page === 1 ? [album] : [],
      pagination: { page, pages: 1, total: 1, hasMore: false },
      notice: WHITE_ALBUM_NOTICE,
    };
  }
  const query = new URLSearchParams({
    type: "master",
    format: "Vinyl",
    per_page: "20",
    page: String(page),
  });
  for (const [key, value] of [
    ["q", params.query],
    ["artist", params.artist],
    ["release_title", params.title],
  ])
    if (key && value) query.set(key, value);
  return cachedDiscovery(`album-search:v1:${query}`, async () => {
    const raw = object(await discogs(`/database/search?${query}`));
    const seen = new Set<number>();
    const albums = array(raw.results)
      .slice(0, 20)
      .filter((value) => object(value).type === "master")
      .map(normalizeAlbumSearch)
      .filter((album) => {
        if (seen.has(album.masterId)) return false;
        seen.add(album.masterId);
        return true;
      });
    return {
      albums,
      pagination: catalogPagination(raw.pagination, page),
      ...(Number(object(raw.pagination).pages) > 100
        ? {
            notice:
              "Showing the first 100 pages. Narrow the artist or title to find more specific matches.",
          }
        : {}),
    };
  });
}

export async function fetchAlbumVersions(
  masterId: number,
  input: z.infer<typeof albumVersionsSchema> = {},
): Promise<PressingSearchResult> {
  z.number().int().positive().max(Number.MAX_SAFE_INTEGER).parse(masterId);
  const params = albumVersionsSchema.parse(input);
  const page = params.page || 1;
  const query = new URLSearchParams({
    format: "Vinyl",
    per_page: "20",
    page: String(page),
  });
  for (const [key, value] of [
    ["country", params.country],
    ["label", params.label],
    ["released", params.year?.toString()],
  ])
    if (key && value) query.set(key, value);
  return cachedDiscovery(`album-versions:v1:${masterId}:${query}`, async () => {
    // Album metadata gives a reliable artist and original year. Never fetch every pressing.
    const album = await fetchCatalogAlbum(masterId);
    const raw = object(await discogs(`/masters/${masterId}/versions?${query}`));
    const seen = new Set<number>();
    const candidates: CatalogCandidate[] = array(raw.versions)
      .slice(0, 20)
      .map(object)
      .filter((version) => texts(version.major_formats).includes("Vinyl"))
      .map((version) => {
        const releaseId = id(version.id);
        return {
          discogsId: releaseId,
          artist: album.artist,
          title: text(version.title) || album.title,
          year: year(version.released),
          originalYear: album.originalYear,
          genres: [],
          label: text(version.label),
          catalogNumber: text(version.catno),
          country: text(version.country),
          format: ["Vinyl", text(version.format)].filter(Boolean).join(", "),
          coverUrl: safeDiscogsImage(version.thumb),
          sourceUrl: `https://www.discogs.com/release/${releaseId}`,
        };
      })
      .filter((candidate) => {
        if (seen.has(candidate.discogsId!)) return false;
        seen.add(candidate.discogsId!);
        return true;
      });
    return {
      candidates,
      pagination: catalogPagination(raw.pagination, page),
      ...(Number(object(raw.pagination).pages) > 100
        ? {
            notice:
              "Showing the first 100 pages. Narrow the country, label or pressing year to find more specific matches.",
          }
        : {}),
    };
  });
}

export async function searchCatalogPage(
  input: CatalogSearch,
): Promise<PressingSearchResult> {
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
    return { candidates: [], pagination: emptyPagination(params.page || 1) };
  if (isBeatlesWhiteAlbum(params) && !params.catalogNumber && !params.barcode) {
    const results = await fetchAlbumVersions(46402, {
      country: params.country,
      label: params.label,
      year: params.year,
      page: params.page,
    });
    return { ...results, notice: WHITE_ALBUM_NOTICE };
  }
  // A catalog number or barcode still belongs in exact release search. Apply
  // the same narrow nickname correction without discarding those identifiers.
  const alias = isBeatlesWhiteAlbum(params);
  const query = new URLSearchParams({
    type: "release",
    format: "Vinyl",
    per_page: "20",
    page: String(params.page || 1),
  });
  for (const [key, value] of [
    ["q", alias ? undefined : params.query],
    ["artist", alias ? "The Beatles" : params.artist],
    ["release_title", alias ? "The Beatles" : params.title],
    ["catno", params.catalogNumber],
    ["barcode", params.barcode?.replace(/[ -]/g, "")],
    ["country", params.country],
    ["label", params.label],
    ["year", params.year?.toString()],
  ]) {
    if (key && value) query.set(key, value);
  }
  return cachedDiscovery(`search:v2:${query}`, async () => {
    const response = object(await discogs(`/database/search?${query}`));
    return {
      candidates: array(response.results)
        .slice(0, 20)
        .map(normalizeSearchCandidate),
      pagination: catalogPagination(response.pagination, params.page || 1),
      ...(alias ? { notice: WHITE_ALBUM_NOTICE } : {}),
      ...(Number(object(response.pagination).pages) > 100
        ? {
            notice:
              "Showing the first 100 pages. Add pressing details to narrow the results.",
          }
        : {}),
    };
  });
}

/** Compatibility wrapper for photo recognition and collection explorer callers. */
export async function searchCatalog(
  input: CatalogSearch,
): Promise<CatalogCandidate[]> {
  return (await searchCatalogPage(input)).candidates;
}

async function releaseResult(
  releaseId: number,
  options: { refresh?: boolean; includeOriginalYear?: boolean } = {},
): Promise<ReleaseResult> {
  z.number().int().positive().max(Number.MAX_SAFE_INTEGER).parse(releaseId);
  const { refresh = false, includeOriginalYear = true } = z
    .object({
      refresh: z.boolean().optional(),
      includeOriginalYear: z.boolean().optional(),
    })
    .strict()
    .parse(options);
  // Duplicate copies often render together. Share an in-flight refresh instead
  // of spending one provider request per visible copy in this instance.
  const key = `${releaseId}:${refresh}:${includeOriginalYear}`;
  const pending = pendingReleases.get(key);
  if (pending) return pending;
  const result = (async () => {
    const stored = await cached<CachedRelease>(
      `release:v2:${releaseId}`,
      async () => {
        const raw = object(await discogs(`/releases/${releaseId}`));
        const release = normalizeRelease(raw);
        if (release.discogsId !== releaseId)
          throw new HttpError(
            502,
            "Discogs returned a different release. Please try again.",
          );
        const masterId =
          typeof raw.master_id === "number" &&
          Number.isSafeInteger(raw.master_id) &&
          raw.master_id > 0
            ? raw.master_id
            : null;
        const invalidMasterId =
          raw.master_id !== undefined &&
          raw.master_id !== null &&
          raw.master_id !== 0 &&
          masterId === null;
        return {
          candidate: release,
          masterId,
          ...(invalidMasterId ? { invalidMasterId: true } : {}),
        };
      },
      refresh,
    );
    let originalYear: number | null = null;
    let originalYearResolved = includeOriginalYear && !stored.invalidMasterId;
    if (includeOriginalYear && stored.masterId !== null) {
      try {
        originalYear = await fetchMasterYear(stored.masterId, refresh);
      } catch {
        originalYearResolved = false;
        // An unavailable master must not block adding a pressing or loading
        // its artwork. Leave this unknown, never infer it from pressing.year.
      }
    }
    return {
      candidate: { ...stored.candidate, originalYear },
      originalYearResolved,
    };
  })();
  pendingReleases.set(key, result);
  try {
    return await result;
  } finally {
    pendingReleases.delete(key);
  }
}

export async function fetchRelease(
  releaseId: number,
  options: { refresh?: boolean; includeOriginalYear?: boolean } = {},
): Promise<CatalogCandidate> {
  return (await releaseResult(releaseId, options)).candidate;
}

/** Distinguishes a genuinely unknown year from a provider outage for safe backfills. */
export async function fetchOriginalYear(releaseId: number): Promise<{
  originalYear: number | null;
  resolved: boolean;
}> {
  const result = await releaseResult(releaseId);
  return {
    originalYear: result.candidate.originalYear ?? null,
    resolved: result.originalYearResolved,
  };
}
