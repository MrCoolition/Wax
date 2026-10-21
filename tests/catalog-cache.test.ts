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
      ["release:42", JSON.stringify(fresh), age],
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
});
