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
const state = vi.hoisted(() => ({ db: null as PGlite | null, fetch: vi.fn() }));
vi.mock("../src/server/db", () => ({
  getPool: () => ({
    query: (sql: string, params?: unknown[]) => state.db!.query(sql, params),
  }),
}));
import {
  fetchRelease,
  normalizeRelease,
  releaseQuerySchema,
  fetchOriginalYear,
  normalizeMasterYear,
  searchCatalog,
} from "../src/server/catalog";

describe("Discogs release freshness", () => {
  beforeAll(async () => {
    state.db = new PGlite();
    await state.db.exec(
      "CREATE SCHEMA wax; CREATE TABLE wax.catalog_cache(cache_key text PRIMARY KEY,payload jsonb NOT NULL,fetched_at timestamptz NOT NULL DEFAULT now());",
    );
  });
  beforeEach(async () => {
    await state.db!.exec("TRUNCATE wax.catalog_cache");
    state.fetch.mockReset();
    vi.stubGlobal("fetch", state.fetch);
  });
  afterAll(async () => {
    vi.unstubAllGlobals();
    await state.db?.close();
  });
  const fresh = normalizeRelease({
    id: 42,
    title: "Cached title",
    artists: [{ name: "Artist" }],
  });
  async function seed(age = "1 hour") {
    await state.db!.query(
      "INSERT INTO wax.catalog_cache(cache_key,payload,fetched_at) VALUES($1,$2::jsonb,now()-$3::interval)",
      [
        "release:v2:42",
        JSON.stringify({ candidate: fresh, masterId: null }),
        age,
      ],
    );
  }
  function upstream() {
    state.fetch.mockResolvedValue(
      Response.json({
        id: 42,
        title: "Updated title",
        artists: [{ name: "Artist" }],
      }),
    );
  }

  it("uses the fresh six-hour cache during normal reads", async () => {
    await seed();
    expect(await fetchRelease(42)).toEqual(fresh);
    expect(state.fetch).not.toHaveBeenCalled();
  });
  it("explicit refresh bypasses a fresh cache and persists the fetched release", async () => {
    await seed();
    upstream();
    expect((await fetchRelease(42, { refresh: true })).title).toBe(
      "Updated title",
    );
    expect(state.fetch).toHaveBeenCalledOnce();
    expect((await fetchRelease(42)).title).toBe("Updated title");
    expect(state.fetch).toHaveBeenCalledOnce();
  });
  it("refetches expired data without an explicit refresh", async () => {
    await seed("6 hours 1 minute");
    upstream();
    expect((await fetchRelease(42)).title).toBe("Updated title");
    expect(state.fetch).toHaveBeenCalledOnce();
  });
  it("fails a refresh clearly instead of silently returning the old result", async () => {
    await seed();
    state.fetch.mockResolvedValue(new Response("", { status: 429 }));
    await expect(fetchRelease(42, { refresh: true })).rejects.toMatchObject({
      status: 429,
    });
    expect((await fetchRelease(42)).title).toBe("Cached title");
  });
  it("accepts only explicit true/false query values", () => {
    expect(releaseQuerySchema.parse({ refresh: "true" })).toEqual({
      refresh: "true",
    });
    expect(releaseQuerySchema.parse({ refresh: "false" })).toEqual({
      refresh: "false",
    });
    expect(releaseQuerySchema.parse({})).toEqual({});
    expect(() => releaseQuerySchema.parse({ refresh: "yes" })).toThrow();
    expect(() => releaseQuerySchema.parse({ refresh: true })).toThrow();
    expect(() =>
      releaseQuerySchema.parse({ url: "https://example.test" }),
    ).toThrow();
  });

  it("uses the validated master year for original release and retains the pressing year", async () => {
    state.fetch.mockImplementation(async (url: string) =>
      Response.json(
        url.includes("/masters/")
          ? { id: 100, year: 1966, main_release: 900 }
          : {
              id: 42,
              title: "Pet Sounds",
              artists: [{ name: "The Beach Boys" }],
              year: 2024,
              master_id: 100,
            },
      ),
    );
    expect(await fetchRelease(42)).toMatchObject({
      year: 2024,
      originalYear: 1966,
    });
    expect(state.fetch.mock.calls.map(([url]) => url)).toEqual([
      "https://api.discogs.com/releases/42",
      "https://api.discogs.com/masters/100",
    ]);
    expect(await fetchOriginalYear(42)).toEqual({
      originalYear: 1966,
      resolved: true,
    });
    expect(state.fetch).toHaveBeenCalledTimes(2);
    await state.db!.exec(
      "UPDATE wax.catalog_cache SET fetched_at=now()-interval '7 hours' WHERE cache_key LIKE 'master-year:%'",
    );
    await fetchRelease(42);
    expect(state.fetch).toHaveBeenCalledTimes(3);
  });

  it("never treats a pressing year or missing master metadata as an original album year", async () => {
    state.fetch.mockResolvedValue(
      Response.json({
        id: 42,
        title: "Only a pressing",
        year: 1981,
        master_id: 0,
      }),
    );
    expect(await fetchOriginalYear(42)).toEqual({
      originalYear: null,
      resolved: true,
    });
    expect((await fetchRelease(42)).year).toBe(1981);
    expect(state.fetch).toHaveBeenCalledOnce();
    expect(normalizeMasterYear({ id: 100, year: 0 }, 100)).toBeNull();
    expect(() => normalizeMasterYear({ id: 101, year: 1971 }, 100)).toThrow();
  });

  it("leaves failed master enrichment retryable without breaking artwork or catalog selection", async () => {
    state.fetch
      .mockResolvedValueOnce(
        Response.json({ id: 42, title: "Reissue", year: 2022, master_id: 100 }),
      )
      .mockResolvedValueOnce(
        new Response("busy", { status: 429, headers: { "Retry-After": "30" } }),
      );
    expect(await fetchOriginalYear(42)).toEqual({
      originalYear: null,
      resolved: false,
    });
    expect(
      await fetchRelease(42, { includeOriginalYear: false }),
    ).toMatchObject({ year: 2022, originalYear: null });
    expect(state.fetch).toHaveBeenCalledTimes(2);
    state.fetch.mockResolvedValueOnce(Response.json({ id: 100, year: 1972 }));
    expect(await fetchOriginalYear(42)).toEqual({
      originalYear: 1972,
      resolved: true,
    });
    expect(state.fetch).toHaveBeenCalledTimes(3);
  });

  it("coalesces shared master lookups and keeps search free from per-candidate master requests", async () => {
    state.fetch.mockImplementation(async (url: string) => {
      if (url.includes("/database/search"))
        return Response.json({
          results: [
            { id: 42, title: "Artist - Album", year: 2024, master_id: 100 },
          ],
        });
      if (url.includes("/masters/")) {
        await new Promise((resolve) => setTimeout(resolve, 20));
        return Response.json({ id: 100, year: 1971 });
      }
      return Response.json({
        id: Number(url.split("/").at(-1)),
        title: "Album",
        year: 2024,
        master_id: 100,
      });
    });
    expect((await searchCatalog({ title: "Album" }))[0]).toMatchObject({
      year: 2024,
      originalYear: null,
    });
    expect(state.fetch).toHaveBeenCalledOnce();
    const releases = await Promise.all([fetchRelease(42), fetchRelease(43)]);
    expect(releases.map((release) => release.originalYear)).toEqual([
      1971, 1971,
    ]);
    expect(
      state.fetch.mock.calls.filter(([url]) => url.includes("/masters/")),
    ).toHaveLength(1);
  });

  it("does not turn malformed provider master IDs into a permanently resolved unknown", async () => {
    state.fetch.mockResolvedValue(
      Response.json({
        id: 42,
        title: "Album",
        year: 2024,
        master_id: "https://wrong.example/master",
      }),
    );
    expect(await fetchOriginalYear(42)).toEqual({
      originalYear: null,
      resolved: false,
    });
    expect(
      await fetchRelease(42, { includeOriginalYear: false }),
    ).toMatchObject({ year: 2024, originalYear: null });
    expect(state.fetch).toHaveBeenCalledOnce();
  });
});
