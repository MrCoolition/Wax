import { readFile } from "node:fs/promises";
import { createHash } from "node:crypto";
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
  type OsmShopDetails,
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
const osmElement = (
  id = 1,
  tags: Record<string, unknown> = {},
  type = "node",
) => ({
  type,
  id,
  visible: true,
  lat: 40,
  lon: -75,
  tags: { shop: "music", name: `Record shop ${id}`, ...tags },
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
    state.fetch
      .mockResolvedValueOnce(Response.json(collection([feature()])))
      .mockResolvedValueOnce(Response.json({ elements: [osmElement()] }));
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
    expect(state.fetch).toHaveBeenCalledTimes(2);
    const url = new URL(String(state.fetch.mock.calls[0][0]));
    expect(url.searchParams.get("lat")).toBe("40");
    expect(url.searchParams.get("lon")).toBe("-75");
    expect(url.searchParams.get("radius")).toBe("25");
    expect(url.searchParams.get("include")).toBe(
      "osm.shop.music,osm.shop.musical_instrument",
    );
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

  it("keeps Columbus vinyl shops and excludes generic, percussion, instrument, and sheet-music listings", () => {
    const listings: [number, string, number, number, string?][] = [
      [12919585334, "Lost Weekend Records", 40.0226458, -83.0131238],
      [13520935953, "Elizabeth Records", 40.025676, -83.0016702],
      [12651098886, "RPM", 40.0148379, -83.0116131],
      [9044668716, "Used Kids Records", 40.014711, -82.9999399],
      [7899581385, "Magnolia Thunderpussy", 39.9859689, -83.005539],
      [3659314459, "Columbus Percussion", 40.0655782, -83.0180051],
      [6898576649, "Sam Ash Music", 40.0615617, -82.9745807],
      [231227657, "Stanton's Sheet Music", 39.9558304, -82.9942125, "W"],
      [231226739, "German Village Music Haus", 39.956072, -82.9894572, "W"],
    ];
    const candidates = listings.map(
      ([id, name, latitude, longitude, osmType = "N"]) => {
        const item = feature(id, longitude, { name, osm_type: osmType });
        item.geometry.coordinates[1] = latitude;
        return item;
      },
    );
    const details: OsmShopDetails = Object.fromEntries(
      listings.map(([id, name, latitude, longitude, osmType = "N"]) => [
        `${osmType === "N" ? "node" : "way"}/${id}`,
        {
          visible: true,
          tags: { shop: "music", name },
          point: { latitude, longitude },
        },
      ]),
    );
    const result = normalizeShops(
      collection(candidates),
      { latitude: 40.02, longitude: -83.04 },
      25,
      details,
    );
    expect(result.shops.map((shop) => shop.name).sort()).toEqual(
      [
        "Lost Weekend Records",
        "Elizabeth Records",
        "RPM",
        "Used Kids Records",
        "Magnolia Thunderpussy",
      ].sort(),
    );
    details["node/12651098886"].tags.music = "cd;cassette";
    details["node/7899581385"].tags["music:vinyl"] = "no";
    expect(
      normalizeShops(
        collection(candidates),
        { latitude: 40.02, longitude: -83.04 },
        25,
        details,
      )
        .shops.map((shop) => shop.name)
        .sort(),
    ).toEqual(
      ["Lost Weekend Records", "Elizabeth Records", "Used Kids Records"].sort(),
    );
    // Source evidence is tied to this location and identity, never to a name alone.
    expect(
      normalizeShops(
        collection([
          feature(1, -75, { name: "RPM" }),
          feature(7899581385, -75, { name: "Magnolia Thunderpussy" }),
        ]),
        center,
        25,
      ).shops,
    ).toEqual([]);
  });

  it("requires positive vinyl evidence and admits mixed shops only with explicit vinyl-sales tags", () => {
    const result = normalizeShops(
      collection([
        feature(1, -75, { name: "Main Street Music" }),
        feature(2, -75, { name: "Main Street Music", music: "vinyl;cd" }),
        feature(3, -75, {
          name: "Guitar Store",
          osm_value: "musical_instrument",
          "music:vinyl": "yes",
        }),
        feature(4, -75, { name: "Records and Lessons" }),
        feature(5, -75, { name: "Records and Lessons", "sells:vinyl": "yes" }),
        feature(6, -75, { name: "CD Records", music: "cd" }),
        feature(7, -75, { name: "Tape Records", music: "cassette;cd" }),
        feature(8, -75, { name: "Control Vinyl" }),
        feature(9, -75, { name: "Timecode Vinyl" }),
        feature(10, -75, { name: "Vinyl Flooring" }),
        feature(11, -75, {
          name: "Main Street Music",
          description: "Vinyl records available",
          website: "https://vinyl.example/",
        }),
        feature(12, -75, { name: "Record Store Studio", studio: "audio" }),
        feature(13, -75, { name: "Main Street Vinyl" }),
      ]),
      center,
      25,
    );
    expect(result.shops.map((shop) => shop.id).sort()).toEqual(
      ["node/2", "node/3", "node/5", "node/13"].sort(),
    );
  });

  it("lets explicit no-vinyl and closure metadata override every positive signal", () => {
    const falseFlags = [
      "music:vinyl",
      "music:records",
      "sells:vinyl",
      "vinyl",
      "records",
    ];
    const features = falseFlags.map((flag, index) =>
      feature(index + 1, -75, {
        name: "Good Vinyl Records",
        music: "vinyl",
        [flag]: "no",
      }),
    );
    features.push(
      feature(30, -75, {
        name: "Good Vinyl Records",
        "music:vinyl": "yes",
        "disused:shop": "music",
      }),
      feature(31, -75, {
        name: "Good Vinyl Records",
        "music:vinyl": "yes",
        opening_hours: "off",
      }),
      feature(32, -75, {
        name: "Good Vinyl Records",
        "music:vinyl": "yes",
        osm_value: "vacant",
      }),
    );
    expect(normalizeShops(collection(features), center, 25).shops).toEqual([]);
  });

  it("uses authoritative current names, categories, coordinates, and deletion status", () => {
    const details: OsmShopDetails = {
      "node/1": {
        visible: true,
        tags: { shop: "music", name: "Renamed Music" },
      },
      "node/2": {
        visible: true,
        tags: { shop: "musical_instrument", name: "Good Records" },
      },
      "node/3": {
        visible: false,
        tags: { shop: "music", name: "Good Records" },
      },
      "node/4": {
        visible: true,
        tags: { shop: "music", name: "Moved Records" },
        point: { latitude: 45, longitude: -75 },
      },
      "node/5": {
        visible: true,
        tags: { shop: "music", "music:vinyl": "yes" },
      },
      "node/6": {
        visible: true,
        tags: {
          shop: "music",
          name: "Current Records",
          "addr:street": "Fresh Street",
        },
      },
    };
    const result = normalizeShops(
      collection(
        Array.from({ length: 7 }, (_, index) =>
          feature(index + 1, -75, {
            name: "Old Records",
            "music:vinyl": "yes",
            street: "Old Street",
            city: "Philadelphia",
          }),
        ),
      ),
      center,
      25,
      details,
    );
    expect(result.shops).toHaveLength(1);
    expect(result.shops[0]).toMatchObject({
      id: "node/6",
      name: "Current Records",
      address: "Fresh Street, Philadelphia",
    });
  });

  it("filters before deduplicating so rejected listings cannot supply links or contact details", () => {
    const result = normalizeShops(
      collection([
        feature(1, -75, {
          name: "Good Records",
          vinyl: "no",
          website: "https://wrong.example/",
          phone: "+1 555 555 5555",
        }),
        feature(2, -75.0001, { name: "Good Records" }),
      ]),
      center,
      25,
    );
    expect(result.shops).toHaveLength(1);
    expect(result.shops[0]).toMatchObject({
      id: "node/2",
      website: null,
      phone: null,
    });
  });

  it("fetches at most three fixed-host OSM batches and shares seven-day metadata without precise locations", async () => {
    const candidates = collection([
      feature(),
      feature(2, -75, {
        osm_type: "W",
        osm_value: "musical_instrument",
        name: "Mixed Music Shop",
      }),
      feature(3, -75, { osm_type: "R" }),
    ]);
    state.fetch.mockImplementation(async (input: URL) => {
      const url = new URL(input);
      if (url.hostname === "photon.komoot.io") return Response.json(candidates);
      const type = url.pathname.includes("nodes")
        ? "node"
        : url.pathname.includes("ways")
          ? "way"
          : "relation";
      const id = type === "node" ? 1 : type === "way" ? 2 : 3;
      return Response.json({
        elements: [
          {
            ...osmElement(
              id,
              type === "way"
                ? {
                    shop: "musical_instrument",
                    name: "Mixed Music Shop",
                    "music:vinyl": "yes",
                  }
                : {},
              type,
            ),
            user: "mapper-private-name",
            uid: 123,
          },
        ],
      });
    });
    expect((await findNearbyShops(area)).shops).toHaveLength(3);
    expect(state.fetch).toHaveBeenCalledTimes(4);
    for (const [input, options] of state.fetch.mock.calls.slice(1)) {
      const url = new URL(String(input));
      expect(url.origin).toBe("https://api.openstreetmap.org");
      expect(url.pathname).toMatch(
        /^\/api\/0\.6\/(nodes|ways|relations)\.json$/,
      );
      expect([...url.searchParams.values()][0]).toMatch(/^[1-3]$/);
      expect(options).toMatchObject({
        redirect: "error",
        cache: "no-store",
        headers: { Accept: "application/json" },
      });
      expect(options.headers.Authorization).toBeUndefined();
    }
    const cache = await state.db!.query<{ payload: unknown; seconds: number }>(
      "SELECT payload,extract(epoch FROM expires_at-now())::int AS seconds FROM wax.shop_discovery_cache WHERE cache_key LIKE 'osm-details:%'",
    );
    expect(cache.rows).toHaveLength(3);
    expect(
      cache.rows.every((row) => row.seconds > 604790 && row.seconds <= 604800),
    ).toBe(true);
    expect(JSON.stringify(cache.rows)).not.toContain("mapper-private-name");
    await allowProvider();
    await findNearbyShops({ ...area, radiusKm: 10 });
    expect(state.fetch).toHaveBeenCalledTimes(5); // New Photon radius, same cached OSM elements.
  });

  it("does not turn unavailable or malformed current details into cached empty results", async () => {
    const invalid = [
      new Response("unavailable", { status: 503 }),
      Response.json({ elements: [{ ...osmElement(), tags: "invalid" }] }),
      Response.json({ elements: [{ ...osmElement(), tags: [] }] }),
      Response.json({ elements: [{ ...osmElement(), lat: 91 }] }),
      Response.json({ elements: [{ ...osmElement(), id: 999 }] }),
      Response.json({ error: "unavailable" }),
      new Response("{}", { headers: { "Content-Length": "1000001" } }),
    ];
    for (const response of invalid) {
      await allowProvider();
      state.fetch
        .mockResolvedValueOnce(Response.json(collection([feature()])))
        .mockResolvedValueOnce(response);
      await expect(findNearbyShops(area)).rejects.toMatchObject({
        status: 503,
      });
      expect(await cacheCount()).toBe(0);
    }
    await allowProvider();
    state.fetch
      .mockResolvedValueOnce(Response.json(collection([feature()])))
      .mockRejectedValueOnce(new DOMException("Timed out", "TimeoutError"));
    await expect(findNearbyShops(area)).rejects.toMatchObject({ status: 503 });
    expect(await cacheCount()).toBe(0);
  });

  it("excludes absent and deleted OSM elements rather than trusting stale Photon listings", async () => {
    state.fetch
      .mockResolvedValueOnce(Response.json(collection([feature(), feature(2)])))
      .mockResolvedValueOnce(
        Response.json({ elements: [{ type: "node", id: 1, visible: false }] }),
      );
    expect((await findNearbyShops(area)).shops).toEqual([]);
    expect(state.fetch).toHaveBeenCalledTimes(2);
  });

  it("ignores pre-filter nearby cache entries immediately", async () => {
    const url = new URL("https://photon.komoot.io/reverse");
    for (const [key, value] of Object.entries({
      lat: "40",
      lon: "-75",
      radius: "25",
      limit: "100",
      include: "osm.shop.music",
      lang: "en",
    }))
      url.searchParams.set(key, value);
    const oldKey = `nearby:${createHash("sha256").update(`${url.href}:40:-75:25`).digest("hex")}`;
    await state.db!.query(
      "INSERT INTO wax.shop_discovery_cache VALUES($1,$2::jsonb,now()+interval '6 hours')",
      [oldKey, JSON.stringify({ shops: [{ name: "Sam Ash Music" }] })],
    );
    state.fetch
      .mockResolvedValueOnce(
        Response.json(collection([feature(1, -75, { name: "Sam Ash Music" })])),
      )
      .mockResolvedValueOnce(
        Response.json({ elements: [osmElement(1, { name: "Sam Ash Music" })] }),
      );
    expect((await findNearbyShops(area)).shops).toEqual([]);
    expect(state.fetch).toHaveBeenCalledTimes(2);
    const keys = await state.db!.query<{ cache_key: string }>(
      "SELECT cache_key FROM wax.shop_discovery_cache",
    );
    expect(
      keys.rows.some(({ cache_key }) =>
        cache_key.startsWith("nearby:vinyl-v2:"),
      ),
    ).toBe(true);
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
