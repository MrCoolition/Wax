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
  auth: vi.fn(),
  limit: vi.fn(),
}));
vi.mock("../src/server/db", () => ({
  getPool: () => ({
    query: (sql: string, params?: unknown[]) => state.db!.query(sql, params),
  }),
}));
vi.mock("../src/server/auth", () => ({ requireCollector: state.auth }));
vi.mock("../src/server/integration-common", () => ({
  consumeIntegrationLimit: state.limit,
}));
import {
  searchAlbums,
  searchCatalog,
  searchCatalogPage,
  fetchAlbumVersions,
  isBeatlesWhiteAlbum,
} from "../src/server/catalog";
import { GET as albumsGet } from "../src/app/api/catalog/albums/route";
import { GET as versionsGet } from "../src/app/api/catalog/albums/[id]/versions/route";
import { GET as releasesGet } from "../src/app/api/catalog/route";
import { HttpError } from "../src/server/http";

const master = {
  id: 46402,
  title: "The Beatles",
  artists: [{ name: "The Beatles" }],
  year: 1968,
  images: [{ type: "primary", uri: "https://i.discogs.com/album.jpg" }],
};
const version = {
  id: 42,
  title: "The Beatles",
  released: "2018",
  country: "US",
  label: "Apple Records",
  catno: "SWBO 101",
  major_formats: ["Vinyl"],
  format: "LP, Album, Reissue",
  thumb: "https://i.discogs.com/copy.jpg",
};
const pagination = (page = 1, items = 42) => ({
  page,
  pages: Math.ceil(items / 20),
  items,
});
const request = (path: string) =>
  new Request(`https://wax.example.test/api/catalog${path}`);
const versionsRequest = (id: string, query = "") =>
  versionsGet(request(`/albums/${id}/versions${query}`), {
    params: Promise.resolve({ id }),
  });

describe("album discovery and pressing filters", () => {
  beforeAll(async () => {
    state.db = new PGlite();
    await state.db.exec(
      "CREATE SCHEMA wax; CREATE TABLE wax.catalog_cache(cache_key text PRIMARY KEY,payload jsonb NOT NULL,fetched_at timestamptz NOT NULL DEFAULT now());",
    );
  });
  beforeEach(async () => {
    vi.clearAllMocks();
    state.fetch.mockReset();
    state.auth.mockResolvedValue({ id: "alice" });
    state.limit.mockResolvedValue(undefined);
    vi.stubGlobal("fetch", state.fetch);
    await state.db!.exec("TRUNCATE wax.catalog_cache");
  });
  afterAll(async () => {
    vi.unstubAllGlobals();
    await state.db?.close();
  });

  it("resolves only the explicit Beatles nickname to its validated album, not a pressing", async () => {
    state.fetch.mockImplementation(async () => Response.json(master));
    for (const query of [
      "beatles white album",
      "Beatles, The White Album",
      "The Beatles - The White Album",
    ])
      expect(isBeatlesWhiteAlbum({ query })).toBe(true);
    expect(
      isBeatlesWhiteAlbum({ artist: "The Beatles", title: "White Album" }),
    ).toBe(true);
    for (const input of [
      { query: "White Album" },
      { artist: "Weezer", title: "White Album" },
      { query: "Beatles White Album demos" },
    ])
      expect(isBeatlesWhiteAlbum(input)).toBe(false);
    const result = await searchAlbums({ query: "Beatles, The White Album" });
    expect(result.albums).toEqual([
      {
        masterId: 46402,
        artist: "The Beatles",
        title: "The Beatles",
        originalYear: 1968,
        coverUrl: "https://i.discogs.com/album.jpg",
        sourceUrl: "https://www.discogs.com/master/46402",
      },
    ]);
    expect(result.notice).toContain("1968 self-titled");
    expect(result.albums[0]).not.toHaveProperty("discogsId");
    await searchAlbums({ artist: "The Beatles", title: "White Album" });
    expect(state.fetch).toHaveBeenCalledOnce();
    expect(state.fetch.mock.calls[0][0]).toBe(
      "https://api.discogs.com/masters/46402",
    );
  });

  it("passes generic artist/title queries through with bounded, provider-independent pagination", async () => {
    state.fetch.mockImplementation(async () =>
      Response.json({
        pagination: { ...pagination(2), urls: { next: "https://evil.test/" } },
        results: [
          {
            id: 16921,
            type: "master",
            title: "Steely Dan - Aja",
            year: "1977",
          },
          { id: 16921, type: "master", title: "Steely Dan - Aja" },
          { id: 88, type: "release", title: "Not an album" },
        ],
      }),
    );
    const result = await searchAlbums({
      artist: "Steely Dan",
      title: "Aja",
      page: 2,
    });
    expect(result.albums).toHaveLength(1);
    expect(result.albums[0]).toMatchObject({
      masterId: 16921,
      originalYear: 1977,
    });
    expect(result.pagination).toEqual({
      page: 2,
      pages: 3,
      total: 42,
      hasMore: true,
    });
    const url = new URL(state.fetch.mock.calls[0][0]);
    expect(Object.fromEntries(url.searchParams)).toEqual({
      type: "master",
      format: "Vinyl",
      per_page: "20",
      page: "2",
      artist: "Steely Dan",
      release_title: "Aja",
    });
    expect(state.fetch).toHaveBeenCalledOnce();
  });

  it("fetches only one master plus one versions page, retaining pressing and album years separately", async () => {
    state.fetch.mockImplementation(async (url: string) =>
      Response.json(
        url.includes("/versions?")
          ? {
              pagination: pagination(2),
              versions: [
                version,
                { ...version, id: 43, major_formats: ["CD"] },
                version,
              ],
            }
          : master,
      ),
    );
    const result = await fetchAlbumVersions(46402, {
      country: "US",
      label: "Apple Records",
      year: 2018,
      page: 2,
    });
    expect(result.candidates).toHaveLength(1);
    expect(result.candidates[0]).toMatchObject({
      discogsId: 42,
      artist: "The Beatles",
      originalYear: 1968,
      year: 2018,
      format: "Vinyl, LP, Album, Reissue",
      catalogNumber: "SWBO 101",
    });
    expect(result.candidates[0]).not.toHaveProperty("pressingConfirmed");
    expect(state.fetch).toHaveBeenCalledTimes(2);
    const url = new URL(state.fetch.mock.calls[1][0]);
    expect(Object.fromEntries(url.searchParams)).toEqual({
      format: "Vinyl",
      per_page: "20",
      page: "2",
      country: "US",
      label: "Apple Records",
      released: "2018",
    });
    await fetchAlbumVersions(46402, {
      country: "US",
      label: "Apple Records",
      year: 2018,
      page: 2,
    });
    expect(state.fetch).toHaveBeenCalledTimes(2);
  });

  it("preserves exact barcode/catalog fields and explicit country/label/year in release searches", async () => {
    state.fetch.mockImplementation(async () =>
      Response.json({
        pagination: pagination(1, 1),
        results: [
          {
            id: 42,
            title: "Artist - Album",
            year: "2024",
            format: ["Vinyl", "LP"],
          },
        ],
      }),
    );
    const input = {
      artist: "Artist",
      title: "Album",
      barcode: "0012 3456-7890",
      catalogNumber: "ABC-123",
      country: "UK",
      label: "A&B",
      year: 2024,
    };
    const result = await searchCatalogPage(input);
    expect(result.pagination).toEqual({
      page: 1,
      pages: 1,
      total: 1,
      hasMore: false,
    });
    expect(await searchCatalog(input)).toEqual(result.candidates);
    expect(state.fetch).toHaveBeenCalledOnce();
    expect(
      Object.fromEntries(new URL(state.fetch.mock.calls[0][0]).searchParams),
    ).toEqual({
      type: "release",
      format: "Vinyl",
      per_page: "20",
      page: "1",
      artist: "Artist",
      release_title: "Album",
      barcode: "001234567890",
      catno: "ABC-123",
      country: "UK",
      label: "A&B",
      year: "2024",
    });
  });

  it("coalesces identical requests and refreshes six-hour-expired discovery data", async () => {
    state.fetch.mockImplementation(async () => {
      await new Promise((resolve) => setTimeout(resolve, 10));
      return Response.json({
        pagination: pagination(1, 1),
        results: [{ id: 1, type: "master", title: "Artist - Album" }],
      });
    });
    await Promise.all([
      searchAlbums({ query: "Artist Album" }),
      searchAlbums({ query: "Artist Album" }),
    ]);
    expect(state.fetch).toHaveBeenCalledOnce();
    await state.db!.exec(
      "UPDATE wax.catalog_cache SET fetched_at=now()-interval '7 hours'",
    );
    await searchAlbums({ query: "Artist Album" });
    expect(state.fetch).toHaveBeenCalledTimes(2);
  });

  it("keeps exact identifiers when resolving the Beatles nickname in pressing search", async () => {
    state.fetch.mockResolvedValue(
      Response.json({
        pagination: pagination(1, 1),
        results: [{ id: 42, title: "The Beatles - The Beatles", year: "1968" }],
      }),
    );
    await searchCatalogPage({
      artist: "The Beatles",
      title: "White Album",
      catalogNumber: "SWBO 101",
      country: "US",
    });
    const query = new URL(state.fetch.mock.calls[0][0]).searchParams;
    expect(query.get("catno")).toBe("SWBO 101");
    expect(query.get("country")).toBe("US");
    expect(query.get("release_title")).toBe("The Beatles");
    expect(query.get("artist")).toBe("The Beatles");
    expect(query.get("type")).toBe("release");
  });

  it("bounds results and deep pagination honestly without following provider URLs", async () => {
    state.fetch.mockImplementation(async () =>
      Response.json({
        pagination: pagination(100, 5000),
        results: Array.from({ length: 30 }, (_, i) => ({
          id: i + 1,
          type: "master",
          title: `Artist - Album ${i}`,
        })),
      }),
    );
    const result = await searchAlbums({ query: "Artist", page: 100 });
    expect(result.albums).toHaveLength(20);
    expect(result.pagination).toEqual({
      page: 100,
      pages: 100,
      total: 5000,
      hasMore: false,
    });
    expect(result.notice).toContain("100 pages");
    await expect(
      searchAlbums({ query: "Artist", page: 101 }),
    ).rejects.toThrow();
    expect(state.fetch).toHaveBeenCalledOnce();
  });

  it("requires authentication and strict bounded route inputs before lookup", async () => {
    state.auth.mockRejectedValueOnce(new HttpError(401, "Please sign in."));
    expect((await albumsGet(request("/albums?query=Aja"))).status).toBe(401);
    for (const id of ["0", "-1", "1.5", "9007199254740992", "https:"])
      expect((await versionsRequest(id)).status).toBe(400);
    for (const query of [
      "?page=101",
      "?year=1700",
      "?catalogNumber=unsupported",
      "?url=https://evil.test/",
    ])
      expect((await versionsRequest("46402", query)).status).toBe(400);
    expect((await releasesGet(request("?barcode=not-a-barcode"))).status).toBe(
      400,
    );
    expect(state.fetch).not.toHaveBeenCalled();
    expect(state.limit).not.toHaveBeenCalled();
  });

  it("returns private errors for throttling, unexpected master IDs, oversize data and outages", async () => {
    state.fetch.mockResolvedValueOnce(
      new Response("busy", { status: 429, headers: { "Retry-After": "15" } }),
    );
    let response = await albumsGet(request("/albums?query=Aja"));
    expect(response.status).toBe(429);
    expect(await response.json()).toMatchObject({
      details: { retryAfter: 15 },
    });
    expect(response.headers.get("cache-control")).toContain("no-store");
    state.fetch.mockResolvedValueOnce(Response.json({ ...master, id: 46403 }));
    expect((await versionsRequest("46402")).status).toBe(502);
    state.fetch.mockResolvedValueOnce(
      new Response("{}", {
        headers: { "Content-Length": String(2 * 1024 * 1024 + 1) },
      }),
    );
    expect((await albumsGet(request("/albums?query=Aja"))).status).toBe(502);
    state.fetch.mockRejectedValueOnce(
      new DOMException("Timeout", "TimeoutError"),
    );
    expect((await albumsGet(request("/albums?query=Aja"))).status).toBe(503);
    expect(state.limit).toHaveBeenCalledWith("alice", "catalog", 120, 60);
  });

  it("shares public catalog cache but authenticates every collector and never returns private library data", async () => {
    state.fetch.mockResolvedValue(Response.json(master));
    const alice = await albumsGet(
      request("/albums?query=beatles%20white%20album"),
    );
    expect(alice.headers.get("cache-control")).toBe("private, no-store");
    state.auth.mockResolvedValue({ id: "bob" });
    const bob = await albumsGet(
      request("/albums?query=beatles%20white%20album"),
    );
    expect(await bob.json()).toEqual(await alice.json());
    expect(state.fetch).toHaveBeenCalledOnce();
    expect(state.limit).toHaveBeenLastCalledWith("bob", "catalog", 120, 60);
  });
});
