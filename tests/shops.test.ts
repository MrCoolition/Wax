import { readFile } from "node:fs/promises";
import { PGlite } from "@electric-sql/pglite";
import {
  afterAll,
  beforeAll,
  beforeEach,
  describe,
  expect,
  it,
  vi,
} from "vitest";

const state = vi.hoisted(() => ({
  db: null as PGlite | null,
  fetch: vi.fn(),
  collector: vi.fn(),
}));
vi.mock("../src/server/db", () => ({
  getPool: () => ({
    query: (sql: string, params?: unknown[]) => state.db!.query(sql, params),
  }),
}));
vi.mock("../src/server/auth", () => ({ requireCollector: state.collector }));
import { HttpError } from "../src/server/http";
import { approximateShopArea } from "../src/lib/shops";
import {
  distanceKm,
  findNearbyShops,
  nearbySchema,
  normalizePlaces,
  normalizeShops,
  placesSchema,
  searchShopPlaces,
} from "../src/server/shops";
import { POST as placesRoute } from "../src/app/api/shops/places/route";
import { POST as nearbyRoute } from "../src/app/api/shops/nearby/route";

const center = { latitude: 40, longitude: -75 };
const area = { ...center, radiusKm: 25 as const };
const feature = (
  id = 1,
  longitude = -75,
  properties: Record<string, unknown> = {},
) => ({
  type: "Feature",
  geometry: { type: "Point", coordinates: [longitude, 40] },
  properties: {
    osm_type: "N",
    osm_id: id,
    osm_key: "shop",
    osm_value: "music",
    name: `Record shop ${id}`,
    ...properties,
  },
});
const collection = (features: unknown[] = []) => ({
  type: "FeatureCollection",
  features,
});
const request = (
  path: string,
  body: unknown,
  origin = "https://wax.example.test",
) =>
  new Request(`https://wax.example.test/api/shops/${path}`, {
    method: "POST",
    headers: { Origin: origin, "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
const allowProvider = () =>
  state.db!.exec("DELETE FROM wax.shop_provider_limits");
const cacheCount = async () =>
  (
    await state.db!.query<{ count: number }>(
      "SELECT count(*)::int AS count FROM wax.shop_discovery_cache",
    )
  ).rows[0].count;

describe("nearby record shop discovery", () => {
  beforeAll(async () => {
    state.db = new PGlite();
    for (const file of [
      "0001_core.sql",
      "0002_integrations.sql",
      "0005_shop_discovery.sql",
    ])
      await state.db.exec(
        await readFile(
          new URL(`../migrations/${file}`, import.meta.url),
          "utf8",
        ),
      );
    await state.db.exec(
      "INSERT INTO wax.users(id,name,email) VALUES('alice','Alice','alice@example.test'),('bob','Bob','bob@example.test'); INSERT INTO wax.collectors(id) VALUES('alice'),('bob')",
    );
  });
  beforeEach(async () => {
    vi.clearAllMocks();
    vi.stubGlobal("fetch", state.fetch);
    vi.stubEnv("PHOTON_API_URL", "https://photon.komoot.io/api/");
    vi.stubEnv("PHOTON_REVERSE_API_URL", "https://photon.komoot.io/reverse");
    state.collector.mockResolvedValue({ id: "alice" });
    await state.db!.exec(
      "TRUNCATE wax.shop_discovery_cache,wax.shop_provider_limits,wax.integration_limits",
    );
  });
  afterAll(async () => {
    vi.unstubAllGlobals();
    vi.unstubAllEnvs();
    await state.db?.close();
  });

  it("validates bounded coordinates, radius, and city queries", () => {
    for (const bad of [NaN, Infinity, 91, -91, "40"])
      expect(nearbySchema.safeParse({ ...area, latitude: bad }).success).toBe(
        false,
      );
    expect(nearbySchema.safeParse({ ...area, longitude: 181 }).success).toBe(
      false,
    );
    expect(nearbySchema.safeParse({ ...area, radiusKm: 500 }).success).toBe(
      false,
    );
    expect(nearbySchema.safeParse({ ...area, ownerId: "bob" }).success).toBe(
      false,
    );
    expect(placesSchema.safeParse({ query: "x" }).success).toBe(false);
    expect(placesSchema.safeParse({ query: "New\nYork" }).success).toBe(false);
    expect(placesSchema.parse({ query: " Philadelphia " }).query).toBe(
      "Philadelphia",
    );
    expect(
      approximateShopArea({ latitude: 39.95263, longitude: -75.16523 }),
    ).toEqual({ latitude: 39.95, longitude: -75.17 });
  });

  it("requires a collector and same-origin POST before provider or database calls", async () => {
    const wrongOrigin = await nearbyRoute(
      request("nearby", area, "https://untrusted.test"),
    );
    expect(wrongOrigin.status).toBe(403);
    expect(state.collector).not.toHaveBeenCalled();
    state.collector.mockRejectedValue(
      new HttpError(401, "Sign in to your collection."),
    );
    expect(
      (await placesRoute(request("places", { query: "Philadelphia" }))).status,
    ).toBe(401);
    expect((await nearbyRoute(request("nearby", area))).status).toBe(401);
    expect(state.fetch).not.toHaveBeenCalled();
    expect(await cacheCount()).toBe(0);
  });

  it("rejects oversized or malformed requests without querying a provider", async () => {
    expect(
      (await placesRoute(request("places", { query: "x".repeat(5000) })))
        .status,
    ).toBe(400);
    expect(
      (await nearbyRoute(request("nearby", { ...area, radiusKm: "25" })))
        .status,
    ).toBe(400);
    expect(state.fetch).not.toHaveBeenCalled();
  });

  it("enforces collector request limits even for cached results", async () => {
    state.fetch.mockResolvedValue(Response.json(collection()));
    for (let index = 0; index < 20; index++)
      expect((await nearbyRoute(request("nearby", area))).status).toBe(200);
    const denied = await nearbyRoute(request("nearby", area));
    expect(denied.status).toBe(429);
    expect((await denied.json()).details.retryAfter).toBeGreaterThan(0);
    expect(state.fetch).toHaveBeenCalledTimes(1);
  });

  it("uses approximate coordinates upstream and shares cached public data across collectors", async () => {
    state.fetch.mockResolvedValue(Response.json(collection([feature()])));
    const specific = {
      latitude: 40.00123456,
      longitude: -75.00123456,
      radiusKm: 25,
    };
    const first = await nearbyRoute(request("nearby", specific));
    expect(first.status).toBe(200);
    state.collector.mockResolvedValue({ id: "bob" });
    expect(
      (await nearbyRoute(request("nearby", { ...specific, latitude: 40.002 })))
        .status,
    ).toBe(200);
    expect(state.fetch).toHaveBeenCalledTimes(1);
    const url = new URL(String(state.fetch.mock.calls[0][0]));
    expect(url.searchParams.get("lat")).toBe("40");
    expect(url.searchParams.get("lon")).toBe("-75");
    expect(url.searchParams.get("radius")).toBe("25");
    expect(url.searchParams.get("include")).toBe("osm.shop.music");
    const cache = await state.db!.query(
      "SELECT * FROM wax.shop_discovery_cache",
    );
    expect(JSON.stringify(cache.rows)).not.toContain("40.00123456");
    expect(JSON.stringify(cache.rows)).not.toContain("alice");
    expect(JSON.stringify(cache.rows)).not.toContain("bob");
    expect(first.headers.get("Cache-Control")).toBe("private, no-store");
  });

  it("maps cities and ambiguous ZIP codes without treating street numbers as ZIP areas", async () => {
    const city = feature(1, -75.12345, {
      name: "Philadelphia",
      city: "Philadelphia",
      state: "Pennsylvania",
      country: "United States",
      osm_value: "city",
    });
    const zip = feature(2, -75.16, {
      name: "19103",
      postcode: "19103",
      city: "Philadelphia",
      state: "Pennsylvania",
      country: "United States",
      osm_value: "postcode",
    });
    const house = feature(3, -74, { name: "19103", osm_value: "house" });
    state.fetch.mockResolvedValueOnce(Response.json(collection([city])));
    const result = await searchShopPlaces({ query: "Philadelphia" });
    expect(result[0]).toMatchObject({
      latitude: 40,
      longitude: -75.12,
      label: "Philadelphia, Pennsylvania, United States",
    });
    const url = new URL(String(state.fetch.mock.calls[0][0]));
    expect(url.searchParams.getAll("layer")).toEqual([
      "city",
      "locality",
      "district",
    ]);
    expect(normalizePlaces(collection([zip, house]), true)).toHaveLength(1);
    await allowProvider();
    state.fetch.mockResolvedValueOnce(Response.json(collection([zip, house])));
    expect(await searchShopPlaces({ query: "19103" })).toHaveLength(1);
    expect(
      new URL(String(state.fetch.mock.calls[1][0])).searchParams.has("layer"),
    ).toBe(false);
  });

  it("expires shared data and caches valid empty searches", async () => {
    state.fetch.mockImplementation(async () => Response.json(collection()));
    await findNearbyShops(area);
    await findNearbyShops(area);
    expect(state.fetch).toHaveBeenCalledTimes(1);
    const lifetime = (
      await state.db!.query<{ seconds: number }>(
        "SELECT extract(epoch FROM expires_at-now())::int AS seconds FROM wax.shop_discovery_cache",
      )
    ).rows[0].seconds;
    expect(lifetime).toBeGreaterThan(21590);
    expect(lifetime).toBeLessThanOrEqual(21600);
    await state.db!.exec(
      "UPDATE wax.shop_discovery_cache SET expires_at=now()-interval '1 second'",
    );
    await allowProvider();
    await findNearbyShops(area);
    expect(state.fetch).toHaveBeenCalledTimes(2);
  });

  it("throttles globally across place and nearby routes without associating locations with users", async () => {
    await state.db!.exec(
      "INSERT INTO wax.shop_provider_limits VALUES('photon',now()+interval '20 seconds')",
    );
    await expect(
      searchShopPlaces({ query: "Philadelphia" }),
    ).rejects.toMatchObject({
      status: 429,
      details: { retryAfter: expect.any(Number) },
    });
    await expect(findNearbyShops(area)).rejects.toMatchObject({ status: 429 });
    expect(state.fetch).not.toHaveBeenCalled();
    expect(await cacheCount()).toBe(0);
  });

  it("waits once for a just-completed place search before looking up nearby shops", async () => {
    state.fetch.mockImplementation(async () => Response.json(collection()));
    await searchShopPlaces({ query: "Philadelphia" });
    const start = Date.now();
    await findNearbyShops(area);
    expect(Date.now() - start).toBeGreaterThan(950);
    expect(state.fetch).toHaveBeenCalledTimes(2);
  });

  it("propagates provider throttling and blocks follow-up requests until retry time", async () => {
    state.fetch.mockResolvedValue(
      new Response("busy", { status: 429, headers: { "Retry-After": "45" } }),
    );
    await expect(findNearbyShops(area)).rejects.toMatchObject({
      status: 429,
      details: { retryAfter: 45 },
    });
    await expect(searchShopPlaces({ query: "Boston" })).rejects.toMatchObject({
      status: 429,
    });
    expect(state.fetch).toHaveBeenCalledTimes(1);
    expect(await cacheCount()).toBe(0);
  });

  it("does not cache outages, malformed JSON, or provider error payloads", async () => {
    for (const response of [
      new Response("unavailable", { status: 503 }),
      new Response("{invalid", { status: 200 }),
      Response.json({ ...collection(), error: "query failed" }),
      Response.json(collection([{}])),
      Response.json({ features: [] }),
      new Response("{}", { headers: { "Content-Length": "500001" } }),
    ]) {
      await allowProvider();
      state.fetch.mockResolvedValueOnce(response);
      await expect(findNearbyShops(area)).rejects.toMatchObject({
        status: 503,
      });
      expect(await cacheCount()).toBe(0);
    }
    await allowProvider();
    state.fetch.mockRejectedValueOnce(
      new DOMException("Timed out", "TimeoutError"),
    );
    await expect(findNearbyShops(area)).rejects.toMatchObject({ status: 503 });
  });

  it("rejects unsafe configured provider URLs before any request", async () => {
    vi.stubEnv("PHOTON_REVERSE_API_URL", "http://not-secure.example/api");
    await expect(findNearbyShops(area)).rejects.toMatchObject({ status: 503 });
    vi.stubEnv("PHOTON_API_URL", "https://username:password@example.test/api");
    await expect(searchShopPlaces({ query: "Boston" })).rejects.toMatchObject({
      status: 503,
    });
    expect(state.fetch).not.toHaveBeenCalled();
  });

  it("filters instruments, invalid coordinates, closed locations and out-of-range shops; sorts by distance", () => {
    const invalid = feature(20);
    invalid.geometry.coordinates = [200, 95];
    const result = normalizeShops(
      collection([
        feature(1, -75.1),
        feature(2, -75.01),
        feature(3, -75, { osm_value: "musical_instrument" }),
        feature(4, -75, { disused: "yes" }),
        feature(5, -75, { "abandoned:shop": "music" }),
        feature(6, -75, { opening_hours: "closed" }),
        feature(7, -70),
        invalid,
      ]),
      center,
      25,
    );
    expect(result.shops.map((shop) => shop.id)).toEqual(["node/2", "node/1"]);
    expect(result.shops[0].distanceKm).toBeCloseTo(0.85, 1);
    expect(
      distanceKm({ latitude: 0, longitude: 0 }, { latitude: 0, longitude: 1 }),
    ).toBeCloseTo(111.195, 2);
    expect(
      distanceKm(
        { latitude: 0, longitude: 179.99 },
        { latitude: 0, longitude: -179.99 },
      ),
    ).toBeLessThan(3);
  });

  it("deduplicates nearby node/way entries conservatively and preserves safe contact metadata", () => {
    const result = normalizeShops(
      collection([
        feature(1, -75, {
          name: "Good Records",
          website: "javascript:alert(1)",
          phone: "<script>",
          opening_hours: "Mo-Sa 11:00-18:00",
        }),
        feature(2, -75.0001, {
          osm_type: "W",
          name: "good records",
          website: "https://records.example/",
          phone: "+1 215 555 0100",
          street: "Main Street",
          housenumber: "123",
          city: "Philadelphia",
        }),
        feature(3, -75.1, { name: "Good Records" }),
        feature(4, -75, { osm_type: "javascript", osm_id: "anything" }),
      ]),
      center,
      25,
    );
    expect(result.shops).toHaveLength(2);
    expect(result.shops[0]).toMatchObject({
      id: "node/1",
      website: "https://records.example/",
      phone: "+1 215 555 0100",
      openingHours: "Mo-Sa 11:00-18:00",
      address: "123 Main Street, Philadelphia",
      osmUrl: "https://www.openstreetmap.org/node/1",
    });
    expect(result.shops[1].website).toBeNull();
    expect(result.shops[1].openingHours).toBeNull();
  });

  it("bounds returned results and reports truncation honestly", () => {
    const result = normalizeShops(
      collection(
        Array.from({ length: 70 }, (_, index) =>
          feature(index + 1, -75 + index * 0.0001),
        ),
      ),
      center,
      25,
    );
    expect(result.shops).toHaveLength(50);
    expect(result.limited).toBe(true);
    expect(normalizeShops(collection(), center, 25)).toEqual({
      shops: [],
      limited: false,
    });
  });
});
