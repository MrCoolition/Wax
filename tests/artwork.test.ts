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
import { GET } from "../src/app/api/catalog/[id]/artwork/route";
import { catalogArtworkPath, normalizeRelease } from "../src/server/catalog";
import { HttpError } from "../src/server/http";

const invoke = (id = "42", query = "") =>
  GET(
    new Request(
      `https://wax.example.test/api/catalog/${encodeURIComponent(id)}/artwork${query}`,
    ),
    { params: Promise.resolve({ id }) },
  );
const upstreamRelease = (cover = "https://i.discogs.com/fresh.jpg") => ({
  id: 42,
  title: "Rejuvenation",
  artists: [{ name: "The Meters" }],
  images: cover ? [{ type: "primary", uri: cover }] : [],
});
async function cache(
  age = "1 hour",
  cover = "https://i.discogs.com/cached.jpg",
) {
  await state.db!.query(
    "INSERT INTO wax.catalog_cache(cache_key,payload,fetched_at) VALUES('release:42',$1::jsonb,now()-$2::interval)",
    [JSON.stringify(normalizeRelease(upstreamRelease(cover))), age],
  );
}
function expectPrivate(response: Response) {
  expect(response.headers.get("cache-control")).toBe("private, no-store");
  expect(response.headers.get("vary")).toBe("Cookie");
  expect(response.headers.get("referrer-policy")).toBe("no-referrer");
}

describe("lazy authenticated record artwork", () => {
  beforeAll(async () => {
    state.db = new PGlite();
    for (const file of ["0001_core.sql", "0002_integrations.sql"])
      await state.db.exec(
        await readFile(
          new URL(`../migrations/${file}`, import.meta.url),
          "utf8",
        ),
      );
    await state.db.exec(
      "INSERT INTO wax.users(id,name,email) VALUES('alice','Alice','alice@example.test'); INSERT INTO wax.collectors(id) VALUES('alice')",
    );
  });
  beforeEach(async () => {
    vi.clearAllMocks();
    vi.stubGlobal("fetch", state.fetch);
    state.collector.mockResolvedValue({ id: "alice" });
    await state.db!.exec("TRUNCATE wax.catalog_cache,wax.integration_limits");
  });
  afterAll(async () => {
    vi.unstubAllGlobals();
    await state.db?.close();
  });

  it("refreshes an expired release on image demand and redirects to new artwork", async () => {
    await cache("7 hours", "https://i.discogs.com/expired.jpg");
    state.fetch.mockResolvedValue(Response.json(upstreamRelease()));
    const response = await invoke();
    expect(response.status).toBe(302);
    expect(response.headers.get("location")).toBe(
      "https://i.discogs.com/fresh.jpg",
    );
    expectPrivate(response);
    expect(state.fetch).toHaveBeenCalledOnce();
    expect(state.fetch.mock.calls[0][0]).toBe(
      "https://api.discogs.com/releases/42",
    );
    expect(state.fetch.mock.calls[0][1]).toMatchObject({
      cache: "no-store",
      redirect: "error",
    });
    const refreshed = (
      await state.db!.query<{ payload: { coverUrl: string }; fresh: boolean }>(
        "SELECT payload,fetched_at>now()-interval '1 minute' AS fresh FROM wax.catalog_cache WHERE cache_key='release:42'",
      )
    ).rows[0];
    expect(refreshed.payload.coverUrl).toBe("https://i.discogs.com/fresh.jpg");
    expect(refreshed.fresh).toBe(true);
  });

  it("reuses fresh catalog metadata without another provider request", async () => {
    await cache();
    const response = await invoke();
    expect(response.status).toBe(302);
    expect(response.headers.get("location")).toBe(
      "https://i.discogs.com/cached.jpg",
    );
    expect(state.fetch).not.toHaveBeenCalled();
    expectPrivate(response);
  });

  it("coalesces simultaneous cover lookups for duplicate copies", async () => {
    state.fetch.mockImplementation(async () => {
      await new Promise((resolve) => setTimeout(resolve, 20));
      return Response.json(upstreamRelease());
    });
    const results = await Promise.all([invoke(), invoke(), invoke()]);
    expect(results.map((response) => response.status)).toEqual([302, 302, 302]);
    expect(state.fetch).toHaveBeenCalledOnce();
  });

  it("requires authentication before reading catalog data", async () => {
    state.collector.mockRejectedValue(new HttpError(401, "Please sign in."));
    const response = await invoke();
    expect(response.status).toBe(401);
    expectPrivate(response);
    expect(state.fetch).not.toHaveBeenCalled();
  });

  it("validates fixed release IDs and never requests a caller-provided URL", async () => {
    for (const id of [
      "0",
      "-1",
      "1e2",
      "42.1",
      "https://evil.test",
      "9007199254740992",
    ])
      expect((await invoke(id)).status).toBe(400);
    expect(state.fetch).not.toHaveBeenCalled();
    state.fetch.mockResolvedValue(Response.json(upstreamRelease()));
    expect(
      (await invoke("42", "?url=https://evil.test/image&retry=1")).status,
    ).toBe(302);
    expect(state.fetch.mock.calls[0][0]).toBe(
      "https://api.discogs.com/releases/42",
    );
    expect(catalogArtworkPath(null)).toBeNull();
    expect(catalogArtworkPath("42")).toBeNull();
    expect(catalogArtworkPath(NaN)).toBeNull();
    expect(catalogArtworkPath(42)).toBe("/api/catalog/42/artwork");
  });

  it("rejects missing or unsafe provider artwork", async () => {
    for (const cover of [
      "",
      "https://evil.test/image.jpg",
      "https://user:pass@i.discogs.com/x",
      "https://i.discogs.com:8080/x",
      "javascript:alert(1)",
    ]) {
      await state.db!.exec("TRUNCATE wax.catalog_cache");
      state.fetch.mockResolvedValue(Response.json(upstreamRelease(cover)));
      const response = await invoke();
      expect(response.status).toBe(404);
      expect(response.headers.get("location")).toBeNull();
      expectPrivate(response);
    }
  });

  it("does not serve expired artwork during an outage and permits a later retry", async () => {
    await cache("7 hours");
    state.fetch.mockResolvedValue(
      new Response("", { status: 429, headers: { "Retry-After": "30" } }),
    );
    const throttled = await invoke();
    expect(throttled.status).toBe(429);
    expect(throttled.headers.get("location")).toBeNull();
    expectPrivate(throttled);
    state.fetch.mockResolvedValue(Response.json(upstreamRelease()));
    const recovered = await invoke("42", "?retry=1");
    expect(recovered.status).toBe(302);
    expect(recovered.headers.get("location")).toBe(
      "https://i.discogs.com/fresh.jpg",
    );
  });

  it("bounds streamed provider responses and rejects a different release", async () => {
    state.fetch.mockResolvedValue(
      new Response("x".repeat(2 * 1024 * 1024 + 1)),
    );
    expect((await invoke()).status).toBe(502);
    state.fetch.mockResolvedValue(
      Response.json({ ...upstreamRelease(), id: 43 }),
    );
    expect((await invoke()).status).toBe(502);
    expect(
      (await state.db!.query("SELECT * FROM wax.catalog_cache")).rows,
    ).toEqual([]);
  });

  it("rate limits an authenticated collector without a provider request", async () => {
    await state.db!.query(
      "INSERT INTO wax.integration_limits(owner_id,operation,window_start,count) VALUES('alice','catalog-artwork',now(),240)",
    );
    const response = await invoke();
    expect(response.status).toBe(429);
    expectPrivate(response);
    expect(state.fetch).not.toHaveBeenCalled();
  });
});
