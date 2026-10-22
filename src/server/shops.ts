import { createHash } from "node:crypto";
import { setTimeout as sleep } from "node:timers/promises";
import { z } from "zod";
import {
  approximateShopArea,
  type NearbyShops,
  type RecordShop,
  type ShopArea,
  type ShopPlace,
} from "@/lib/shops";
import { getPool } from "./db";
import { HttpError } from "./http";
import { hasVinylSalesEvidence } from "./shop-vinyl";

export const placesSchema = z
  .object({
    query: z
      .string()
      .trim()
      .min(2)
      .max(120)
      .regex(/^[^\x00-\x1f\x7f]+$/),
  })
  .strict();
export const nearbySchema = z
  .object({
    latitude: z.number().finite().min(-90).max(90),
    longitude: z.number().finite().min(-180).max(180),
    radiusKm: z.union([z.literal(10), z.literal(25), z.literal(50)]),
  })
  .strict();
type JsonObject = Record<string, unknown>;
const object = (value: unknown): JsonObject =>
  value && typeof value === "object" && !Array.isArray(value)
    ? (value as JsonObject)
    : {};
const text = (value: unknown, max = 240): string =>
  typeof value === "string"
    ? value
        .replace(/[\x00-\x1f\x7f]/g, " ")
        .trim()
        .slice(0, max)
    : "";
const unavailable = () =>
  new HttpError(
    503,
    "The map service could not finish this search. Keep this area and try again shortly.",
    { retryAfter: 15 },
  );
const hash = (value: string) =>
  createHash("sha256").update(value).digest("hex");

function endpoint(reverse = false): URL {
  try {
    const value = reverse
      ? process.env.PHOTON_REVERSE_API_URL || "https://photon.komoot.io/reverse"
      : process.env.PHOTON_API_URL || "https://photon.komoot.io/api/";
    const url = new URL(value);
    if (url.protocol !== "https:" || url.username || url.password || url.hash)
      throw new Error();
    return url;
  } catch {
    throw new HttpError(
      503,
      "The map service is not configured. Please try again later.",
    );
  }
}

async function readJson(
  response: Response | Request,
  maxBytes: number,
): Promise<unknown> {
  if (Number(response.headers.get("content-length")) > maxBytes)
    throw unavailable();
  const reader = response.body?.getReader();
  if (!reader) throw unavailable();
  const chunks: Uint8Array[] = [];
  let length = 0;
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      length += value.byteLength;
      if (length > maxBytes) {
        await reader.cancel();
        throw unavailable();
      }
      chunks.push(value);
    }
    return JSON.parse(Buffer.concat(chunks, length).toString("utf8"));
  } finally {
    reader.releaseLock();
  }
}

export async function readShopRequest(request: Request): Promise<unknown> {
  try {
    return await readJson(request, 4096);
  } catch {
    throw new HttpError(400, "Enter a city, ZIP code, or valid search area.");
  }
}

async function reserveProvider(mayWait = true): Promise<void> {
  const result = await getPool().query<{ provider: string }>(
    `
    INSERT INTO wax.shop_provider_limits(provider,available_at)
    VALUES($1,now()+($2::double precision * interval '1 millisecond'))
    ON CONFLICT(provider) DO UPDATE SET available_at=excluded.available_at
    WHERE wax.shop_provider_limits.available_at <= now()
    RETURNING provider`,
    ["photon", 1100],
  );
  if (result.rows.length) return;
  const wait = await getPool().query<{ retry_after_ms: number }>(
    `
    SELECT greatest(1,ceil(extract(epoch FROM available_at-now())*1000))::int AS retry_after_ms
    FROM wax.shop_provider_limits WHERE provider=$1`,
    ["photon"],
  );
  const waitMs = wait.rows[0]?.retry_after_ms || 5000;
  // A place choice immediately after geocoding should not need another click.
  // One bounded wait; concurrent collectors never create an unbounded queue.
  if (mayWait && waitMs <= 1190) {
    await sleep(waitMs + 10);
    return reserveProvider(false);
  }
  throw new HttpError(
    429,
    "Another crate digger is searching. Try again in a moment.",
    { retryAfter: Math.ceil(waitMs / 1000) },
  );
}

function retryAfter(response: Response): number {
  const value = response.headers.get("retry-after");
  const numeric = Number(value);
  const seconds =
    value && Number.isFinite(numeric)
      ? numeric
      : value
        ? Math.ceil((Date.parse(value) - Date.now()) / 1000)
        : 30;
  return Math.min(300, Math.max(1, Number.isFinite(seconds) ? seconds : 30));
}

async function providerJson(url: URL, details = false): Promise<unknown> {
  if (!details) await reserveProvider();
  try {
    const response = await fetch(url, {
      headers: {
        "User-Agent": "WAX/1.0 (+https://wax-coolition.vercel.app)",
        Accept: "application/json",
      },
      signal: AbortSignal.timeout(details ? 8000 : 12_000),
      cache: "no-store",
      redirect: "error",
    });
    if (response.status === 429) {
      const seconds = retryAfter(response);
      await getPool().query(
        `UPDATE wax.shop_provider_limits
        SET available_at=greatest(available_at,now()+make_interval(secs => $2))
        WHERE provider=$1`,
        ["photon", seconds],
      );
      throw new HttpError(
        429,
        "The map service is busy. Your search area is still here; try again shortly.",
        { retryAfter: seconds },
      );
    }
    if (!response.ok) throw unavailable();
    return await readJson(response, details ? 1_000_000 : 500_000);
  } catch (error) {
    if (error instanceof HttpError) throw error;
    throw unavailable();
  }
}

async function cached<T>(
  key: string,
  ttlSeconds: number,
  fetchValue: () => Promise<T>,
): Promise<T> {
  await getPool().query(
    "DELETE FROM wax.shop_discovery_cache WHERE expires_at <= now()",
  );
  const found = await getPool().query<{ payload: T }>(
    "SELECT payload FROM wax.shop_discovery_cache WHERE cache_key=$1 AND expires_at > now()",
    [key],
  );
  if (found.rows[0]) return found.rows[0].payload;
  const payload = await fetchValue();
  await getPool().query(
    `INSERT INTO wax.shop_discovery_cache(cache_key,payload,expires_at)
    VALUES($1,$2::jsonb,now()+make_interval(secs => $3))
    ON CONFLICT(cache_key) DO UPDATE SET payload=excluded.payload,expires_at=excluded.expires_at`,
    [key, JSON.stringify(payload), ttlSeconds],
  );
  return payload;
}

function coordinates(latitude: unknown, longitude: unknown): ShopArea | null {
  return typeof latitude === "number" &&
    Number.isFinite(latitude) &&
    latitude >= -90 &&
    latitude <= 90 &&
    typeof longitude === "number" &&
    Number.isFinite(longitude) &&
    longitude >= -180 &&
    longitude <= 180
    ? { latitude, longitude }
    : null;
}

export function normalizePlaces(value: unknown, zipOnly = false): ShopPlace[] {
  const raw = object(value);
  if (
    raw.type !== "FeatureCollection" ||
    raw.error !== undefined ||
    raw.message !== undefined ||
    !Array.isArray(raw.features)
  )
    throw unavailable();
  const places: ShopPlace[] = [];
  let usableFeatures = 0;
  for (const feature of raw.features.slice(0, 30)) {
    const item = object(feature);
    const geometry = object(item.geometry);
    const properties = object(item.properties);
    const point = Array.isArray(geometry.coordinates)
      ? geometry.coordinates
      : [];
    const area =
      geometry.type === "Point" ? coordinates(point[1], point[0]) : null;
    const label = [
      ...new Set(
        [
          text(properties.name),
          text(properties.postcode),
          text(properties.city),
          text(properties.state),
          text(properties.country),
        ].filter(Boolean),
      ),
    ].join(", ");
    if (item.type !== "Feature" || !area || !label) continue;
    usableFeatures++;
    if (zipOnly && properties.osm_value !== "postcode") continue;
    const approximate = approximateShopArea(area);
    const id = hash(
      `${label}:${approximate.latitude}:${approximate.longitude}`,
    ).slice(0, 24);
    if (!places.some((place) => place.id === id))
      places.push({ id, label, ...approximate });
  }
  if (raw.features.length && !usableFeatures) throw unavailable();
  return places.slice(0, 5);
}

export async function searchShopPlaces(
  input: z.infer<typeof placesSchema>,
): Promise<ShopPlace[]> {
  const { query } = placesSchema.parse(input);
  const normalized = query.replace(/\s+/g, " ");
  const zipOnly = /^\d{5}(-\d{4})?$/.test(normalized);
  const url = endpoint();
  url.searchParams.set("q", normalized);
  url.searchParams.set("limit", zipOnly ? "8" : "5");
  url.searchParams.set("lang", "en");
  if (!zipOnly)
    for (const layer of ["city", "locality", "district"])
      url.searchParams.append("layer", layer);
  return cached(
    `places:${hash(url.href.toLowerCase())}`,
    7 * 24 * 3600,
    async () => normalizePlaces(await providerJson(url), zipOnly),
  );
}

export function distanceKm(first: ShopArea, second: ShopArea): number {
  const radians = (degrees: number) => (degrees * Math.PI) / 180;
  const latitude = radians(second.latitude - first.latitude);
  const longitude = radians(second.longitude - first.longitude);
  const value =
    Math.sin(latitude / 2) ** 2 +
    Math.cos(radians(first.latitude)) *
      Math.cos(radians(second.latitude)) *
      Math.sin(longitude / 2) ** 2;
  return (
    6371.0088 *
    2 *
    Math.atan2(Math.sqrt(value), Math.sqrt(Math.max(0, 1 - value)))
  );
}

function safeWebsite(value: unknown): string | null {
  const candidate = text(value, 2000);
  if (!candidate || /[\s<>]/.test(candidate)) return null;
  try {
    const url = new URL(candidate);
    return ["http:", "https:"].includes(url.protocol) &&
      !url.username &&
      !url.password
      ? url.href
      : null;
  } catch {
    return null;
  }
}

function safePhone(value: unknown): string | null {
  const phone = text(value, 80);
  return phone && /^[+\d][\d\s()+./;#-]{4,79}$/.test(phone) ? phone : null;
}

type OsmType = "node" | "way" | "relation";
export type OsmShopDetails = Record<
  string,
  {
    tags: JsonObject;
    visible: boolean;
    point?: ShopArea;
  }
>;

function osmIdentity(
  tags: JsonObject,
): { type: OsmType; id: number; key: string } | null {
  const type = ({ N: "node", W: "way", R: "relation" } as const)[
    String(tags.osm_type) as "N" | "W" | "R"
  ];
  return type &&
    typeof tags.osm_id === "number" &&
    Number.isSafeInteger(tags.osm_id) &&
    tags.osm_id > 0
    ? { type, id: tags.osm_id, key: `${type}/${tags.osm_id}` }
    : null;
}

function photonFeatures(value: unknown): unknown[] {
  const raw = object(value);
  if (
    raw.type !== "FeatureCollection" ||
    raw.error !== undefined ||
    raw.message !== undefined ||
    !Array.isArray(raw.features)
  )
    throw unavailable();
  return raw.features;
}

async function currentShopDetails(value: unknown): Promise<OsmShopDetails> {
  const grouped: Record<OsmType, Set<number>> = {
    node: new Set(),
    way: new Set(),
    relation: new Set(),
  };
  for (const feature of photonFeatures(value).slice(0, 100)) {
    const identity = osmIdentity(object(object(feature).properties));
    if (identity) grouped[identity.type].add(identity.id);
  }
  const details: OsmShopDetails = {};
  // At most one bounded request per element type. Only public tags and node
  // coordinates are retained; contributor information is discarded.
  for (const type of ["node", "way", "relation"] as const) {
    const ids = [...grouped[type]].sort((a, b) => a - b);
    if (!ids.length) continue;
    const url = new URL(`https://api.openstreetmap.org/api/0.6/${type}s.json`);
    url.searchParams.set(`${type}s`, ids.join(","));
    const batch = await cached<OsmShopDetails>(
      `osm-details:v1:${hash(url.href)}`,
      7 * 24 * 3600,
      async () => {
        const raw = object(await providerJson(url, true));
        if (
          !Array.isArray(raw.elements) ||
          raw.elements.length > 100 ||
          raw.error !== undefined
        )
          throw unavailable();
        const found: OsmShopDetails = {};
        for (const entry of raw.elements) {
          const element = object(entry);
          if (
            element.type !== type ||
            typeof element.id !== "number" ||
            !grouped[type].has(element.id)
          )
            throw unavailable();
          if (
            element.visible !== undefined &&
            typeof element.visible !== "boolean"
          )
            throw unavailable();
          if (
            element.tags !== undefined &&
            (!element.tags ||
              typeof element.tags !== "object" ||
              Array.isArray(element.tags))
          )
            throw unavailable();
          const tags = object(element.tags);
          if (Object.values(tags).some((tag) => typeof tag !== "string"))
            throw unavailable();
          const point =
            type === "node" ? coordinates(element.lat, element.lon) : null;
          if (type === "node" && element.visible !== false && !point)
            throw unavailable();
          found[`${type}/${element.id}`] = {
            tags,
            visible: element.visible !== false,
            ...(point ? { point } : {}),
          };
        }
        return found;
      },
    );
    Object.assign(details, batch);
  }
  return details;
}

export function normalizeShops(
  value: unknown,
  center: ShopArea,
  radiusKm: number,
  details?: OsmShopDetails,
): { shops: RecordShop[]; limited: boolean } {
  const features = photonFeatures(value);
  const shops: RecordShop[] = [];
  let usableFeatures = 0;
  for (const element of features.slice(0, 100)) {
    const item = object(element);
    const properties = object(item.properties);
    const geometry = object(item.geometry);
    const pointValues = Array.isArray(geometry.coordinates)
      ? geometry.coordinates
      : [];
    const photonPoint =
      geometry.type === "Point"
        ? coordinates(pointValues[1], pointValues[0])
        : null;
    if (
      item.type !== "Feature" ||
      !photonPoint ||
      !Object.keys(properties).length
    )
      continue;
    usableFeatures++;
    const identity = osmIdentity(properties);
    if (!identity || properties.osm_key !== "shop") continue;
    const current = details?.[identity.key];
    if (details && (!current || !current.visible)) continue;
    // Current OSM tags are authoritative. Only Photon address components may
    // fill gaps, never its old category/name/product/closure metadata.
    const tags: JsonObject = current
      ? { ...current.tags }
      : { ...properties, shop: properties.osm_value };
    for (const key of ["housenumber", "street", "city", "state", "postcode"])
      tags[key] = tags[`addr:${key}`] || properties[key];
    const point = current?.point || photonPoint;
    const name = text(tags.name) || text(tags["name:en"]);
    if (!name || !hasVinylSalesEvidence(identity.key, name, point, tags))
      continue;
    const distance = distanceKm(center, point);
    if (distance > radiusKm) continue;
    const address = [
      [text(tags.housenumber), text(tags.street)].filter(Boolean).join(" "),
      text(tags.city),
      text(tags.state),
      text(tags.postcode),
    ]
      .filter(Boolean)
      .join(", ");
    const shop: RecordShop = {
      id: identity.key,
      name,
      ...point,
      distanceKm: Math.round(distance * 100) / 100,
      address: address || null,
      website:
        safeWebsite(tags.website) || safeWebsite(tags["contact:website"]),
      phone: safePhone(tags.phone) || safePhone(tags["contact:phone"]),
      openingHours: text(tags.opening_hours, 500) || null,
      osmUrl: `https://www.openstreetmap.org/${identity.key}`,
    };
    const duplicate = shops.find(
      (existing) =>
        existing.id === shop.id ||
        (existing.name.toLocaleLowerCase() === shop.name.toLocaleLowerCase() &&
          distanceKm(existing, shop) <= 0.05),
    );
    if (duplicate) {
      duplicate.address ||= shop.address;
      duplicate.website ||= shop.website;
      duplicate.phone ||= shop.phone;
      duplicate.openingHours ||= shop.openingHours;
    } else shops.push(shop);
  }
  if (features.length && !usableFeatures) throw unavailable();
  shops.sort(
    (a, b) => a.distanceKm - b.distanceKm || a.name.localeCompare(b.name),
  );
  // Public Photon instances may clamp a larger requested limit to 50.
  return {
    shops: shops.slice(0, 50),
    limited: shops.length > 50 || features.length >= 50,
  };
}

export async function findNearbyShops(
  input: z.infer<typeof nearbySchema>,
): Promise<NearbyShops> {
  const parsed = nearbySchema.parse(input);
  const center = approximateShopArea(parsed);
  const url = endpoint(true);
  url.searchParams.set("lat", String(center.latitude));
  url.searchParams.set("lon", String(center.longitude));
  url.searchParams.set("radius", String(parsed.radiusKm));
  url.searchParams.set("limit", "100");
  url.searchParams.set("include", "osm.shop.music,osm.shop.musical_instrument");
  url.searchParams.set("lang", "en");
  const key = `nearby:vinyl-v2:${hash(`${url.href}:${center.latitude}:${center.longitude}:${parsed.radiusKm}`)}`;
  return cached(key, 6 * 3600, async () => {
    const candidates = await providerJson(url);
    const details = await currentShopDetails(candidates);
    const result = normalizeShops(candidates, center, parsed.radiusKm, details);
    return {
      center,
      radiusKm: parsed.radiusKm,
      ...result,
      fetchedAt: new Date().toISOString(),
    };
  });
}
