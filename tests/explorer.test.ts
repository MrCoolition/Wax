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
import {
  normalizeExplorerArtist,
  normalizeExplorerDiscography,
  normalizeExplorerRelease,
  plainDiscogsProfile,
  resolveTrackReference,
  splitCreditRoles,
} from "../src/server/explorer-normalize";
import {
  fetchExplorerArtist,
  fetchExplorerDiscography,
  fetchExplorerRelease,
} from "../src/server/explorer";
import { GET as releaseRoute } from "../src/app/api/explore/releases/[id]/route";
import { GET as artistRoute } from "../src/app/api/explore/artists/[id]/route";
import { GET as albumsRoute } from "../src/app/api/explore/artists/[id]/releases/route";
import { GET as startRoute } from "../src/app/api/records/[id]/explore/route";

const release = {
  id: 2825456,
  title: "Kind Of Blue",
  year: 1959,
  artists: [{ id: 23755, name: "Miles Davis", role: "" }],
  labels: [{ name: "Columbia", catno: "CS 8163" }],
  images: [{ type: "primary", uri: "https://i.discogs.com/cover.jpg" }],
  tracklist: [
    { position: "A1", title: "So What", type_: "track", duration: "9:22" },
    { position: "A2", title: "Freddie Freeloader", type_: "track" },
    { position: "A3", title: "Blue In Green", type_: "track" },
    { position: "B1", title: "All Blues", type_: "track" },
    { position: "B2", title: "Flamenco Sketches", type_: "track" },
  ],
  extraartists: [
    { id: 252310, name: "Bill Evans", role: "Piano", tracks: "A1, A3, B1, B2" },
    { id: 252308, name: "Wynton Kelly", role: "Piano", tracks: "A2" },
    { id: 23755, name: "Miles Davis", role: "Trumpet", tracks: "" },
    { id: 252311, name: "Jimmy Cobb", role: "Drums", tracks: "" },
  ],
};
const artist = {
  id: 252310,
  name: "Bill Evans",
  realname: "William John Evans",
  profile: "Jazz pianist.",
  groups: [{ id: 1046897, name: "Miles Davis Sextet", active: false }],
  members: [],
  aliases: [{ id: 999, name: "Example alias" }],
  images: [{ uri: "https://i.discogs.com/artist.jpg", type: "primary" }],
};
const discography = {
  pagination: { page: 1, pages: 2, items: 22 },
  releases: [
    {
      id: 222,
      type: "master",
      main_release: 333,
      title: "Another record",
      artist: "Bill Evans",
      role: "Main",
      year: 1961,
    },
    {
      id: 222,
      type: "release",
      title: "Same numeric ID, different entity",
      artist: "Other artist",
      role: "Piano",
      year: 1962,
    },
    {
      id: 444,
      type: "master",
      title: "Missing main pressing",
      artist: "Band",
      role: "Appearance",
    },
  ],
};
const aliceCopy = "10000000-0000-4000-8000-000000000001";
const manualCopy = "10000000-0000-4000-8000-000000000002";
const bobCopy = "10000000-0000-4000-8000-000000000003";
const context = (id: string | number) => ({
  params: Promise.resolve({ id: String(id) }),
});
const request = (path = "releases/2825456") =>
  new Request(`https://wax.example.test/api/explore/${path}`);

describe("grounded release credits and artist relationships", () => {
  it("distinguishes Bill Evans from Wynton Kelly and never spreads blank-scope credits across songs", () => {
    const result = normalizeExplorerRelease(release, release.id);
    expect(result.tracks[0].credits.map((credit) => credit.name)).toEqual([
      "Bill Evans",
    ]);
    expect(result.tracks[1].credits.map((credit) => credit.name)).toEqual([
      "Wynton Kelly",
    ]);
    expect(result.releaseCredits.map((credit) => credit.name)).toEqual([
      "Miles Davis",
      "Jimmy Cobb",
    ]);
    expect(
      result.releaseCredits.every((credit) => credit.trackReference === null),
    ).toBe(true);
    expect(result.artists).toEqual([{ id: 23755, name: "Miles Davis" }]);
    expect(
      result.tracks[0].credits.every((credit) => credit.name !== "Miles Davis"),
    ).toBe(true);
  });

  it("resolves exact positions and numeric ranges by release order around headings", () => {
    const result = normalizeExplorerRelease(
      {
        ...release,
        tracklist: [
          { title: "Side one", position: "", type_: "heading" },
          { title: "One", position: "1" },
          { title: "Two", position: "2" },
          { title: "Side two", position: "", type_: "heading" },
          { title: "Ten", position: "10" },
        ],
        extraartists: [
          { id: 1, name: "Player", role: "Guitar", tracks: "1 to 10" },
        ],
      },
      release.id,
    );
    expect(
      result.tracks
        .filter((track) => track.credits.length)
        .map((track) => track.position),
    ).toEqual(["1", "2", "10"]);
    expect(resolveTrackReference("10 to 1", result.tracks)).toBeNull();
    expect(
      resolveTrackReference("1, 10", result.tracks)?.map(
        (track) => track.position,
      ),
    ).toEqual(["1", "10"]);
  });

  it("retains ambiguous, partial, and unknown references without assigning invented tracks", () => {
    const extraartists = [
      { id: 1, name: "One", role: "Producer", tracks: "A1" },
      { id: 2, name: "Two", role: "Piano", tracks: "A2, ZZ" },
      { id: 3, name: "Three", role: "Bass", tracks: "A1 to A2" },
    ];
    const result = normalizeExplorerRelease(
      {
        ...release,
        extraartists,
        tracklist: [
          { title: "One", position: "A1" },
          { title: "Also one", position: "A1" },
          { title: "Two", position: "A2" },
        ],
      },
      release.id,
    );
    expect(result.tracks.every((track) => track.credits.length === 0)).toBe(
      true,
    );
    expect(
      result.unresolvedCredits.map((credit) => credit.trackReference),
    ).toEqual(["A1", "A2, ZZ", "A1 to A2"]);
    expect(new Set(result.tracks.map((track) => track.id)).size).toBe(3);
  });

  it("preserves nested tracks, direct track credits, nonlinked artists, and bracketed roles", () => {
    const result = normalizeExplorerRelease(
      {
        ...release,
        extraartists: [],
        tracklist: [
          {
            title: "Suite",
            position: "A",
            type_: "index",
            sub_tracks: [
              {
                title: "Part one",
                position: "A1",
                extraartists: [
                  {
                    id: 0,
                    name: "Unlinked writer",
                    role: "Written By, Engineer [Recording, Mixing], Guitar",
                    tracks: "",
                  },
                ],
              },
              {
                title: "Part two",
                position: "A2",
                artists: [{ id: 15, name: "Guest" }],
              },
            ],
          },
        ],
      },
      release.id,
    );
    expect(result.tracks.map((track) => track.id)).toEqual([
      "2825456:0",
      "2825456:0.0",
      "2825456:0.1",
    ]);
    expect(
      result.tracks[1].credits.map((credit) => [
        credit.role,
        credit.category,
        credit.artistId,
      ]),
    ).toEqual([
      ["Written By", "writing", null],
      ["Engineer [Recording, Mixing]", "production", null],
      ["Guitar", "performance", null],
    ]);
    expect(result.tracks[2].credits[0]).toMatchObject({
      artistId: 15,
      role: "Track artist",
      category: "other",
    });
    expect(
      splitCreditRoles("Producer, Engineer [Recording, Mixing], Arranged By"),
    ).toEqual(["Producer", "Engineer [Recording, Mixing]", "Arranged By"]);
  });

  it("keeps contradictory track-local references unresolved", () => {
    const result = normalizeExplorerRelease(
      {
        ...release,
        extraartists: [],
        tracklist: [
          {
            title: "One",
            position: "A1",
            extraartists: [
              { id: 2, name: "Player", role: "Bass", tracks: "A2" },
            ],
          },
          { title: "Two", position: "A2" },
        ],
      },
      release.id,
    );
    expect(result.unresolvedCredits[0]).toMatchObject({
      name: "Player",
      trackReference: "A2",
    });
    expect(result.tracks.every((track) => !track.credits.length)).toBe(true);
  });

  it("deduplicates the same scoped credit listed in both release and track fields", () => {
    const result = normalizeExplorerRelease(
      {
        ...release,
        tracklist: [
          {
            title: "So What",
            position: "A1",
            extraartists: [
              { id: 252310, name: "Bill Evans", role: "Piano", tracks: "" },
            ],
          },
        ],
        extraartists: [
          { id: 252310, name: "Bill Evans", role: "Piano", tracks: "A1" },
        ],
      },
      release.id,
    );
    expect(result.tracks[0].credits).toHaveLength(1);
    expect(result.tracks[0].credits[0].trackReference).toBe("A1");
  });

  it("bounds normalization and rejects malformed identity", () => {
    expect(() => normalizeExplorerRelease(release, 1)).toThrow(HttpError);
    expect(() =>
      normalizeExplorerRelease(
        { ...release, tracklist: undefined },
        release.id,
      ),
    ).toThrow(HttpError);
    expect(() =>
      normalizeExplorerRelease(
        {
          ...release,
          tracklist: Array.from({ length: 301 }, () => ({ title: "Track" })),
        },
        release.id,
      ),
    ).toThrow(HttpError);
    expect(() =>
      normalizeExplorerRelease(
        { ...release, extraartists: Array.from({ length: 1001 }, () => ({})) },
        release.id,
      ),
    ).toThrow(HttpError);
    expect(() =>
      normalizeExplorerRelease(
        {
          ...release,
          extraartists: [
            { id: 1, name: "Player", role: "Piano", tracks: "A".repeat(2001) },
          ],
        },
        release.id,
      ),
    ).toThrow(HttpError);
  });

  it("keeps artist membership status nullable and rejects unsafe images and URLs", () => {
    const result = normalizeExplorerArtist(
      {
        ...artist,
        uri: "javascript:alert(1)",
        images: [{ uri: "https://evil.test/portrait" }],
      },
      artist.id,
    );
    expect(result.imageUrl).toBeNull();
    expect(result.sourceUrl).toBe("https://www.discogs.com/artist/252310");
    expect(result.groups[0].active).toBe(false);
    expect(result.aliases[0].active).toBeNull();
    expect(() => normalizeExplorerArtist(artist, 1)).toThrow(HttpError);
  });

  it("makes common profile markup readable without creating links or HTML", () => {
    const profile =
      "[b]Jazz pianist[/b]\nPlayed with [a=Miles Davis] and [a252308]. [url=https://evil.test]Read more[/url] [additional notes]";
    expect(plainDiscogsProfile(profile)).toBe(
      "Jazz pianist\nPlayed with Miles Davis and Discogs artist #252308. Read more [additional notes]",
    );
    expect(
      normalizeExplorerArtist({ ...artist, profile }, artist.id).profile,
    ).not.toContain("https://evil.test");
    expect(plainDiscogsProfile("x".repeat(20_000))).toHaveLength(6000);
    expect(plainDiscogsProfile(null)).toBe("");
  });

  it("separates master and release IDs and preserves actual discography roles", () => {
    const result = normalizeExplorerDiscography(discography, artist.id, 1);
    expect(result.releases).toHaveLength(3);
    expect(result.releases[0]).toMatchObject({
      id: 222,
      type: "master",
      mainReleaseId: 333,
      sourceUrl: "https://www.discogs.com/master/222",
    });
    expect(result.releases[1]).toMatchObject({
      id: 222,
      type: "release",
      mainReleaseId: null,
      role: "Piano",
    });
    expect(result.releases[2].mainReleaseId).toBeNull();
    expect(() =>
      normalizeExplorerDiscography(discography, artist.id, 2),
    ).toThrow(HttpError);
  });
});

describe("authenticated lazy Artist Tree API", () => {
  beforeAll(async () => {
    state.db = new PGlite();
    for (const file of [
      "0001_core.sql",
      "0002_integrations.sql",
      "0003_catalog_separation.sql",
    ])
      await state.db.exec(
        await readFile(
          new URL(`../migrations/${file}`, import.meta.url),
          "utf8",
        ),
      );
    await state.db.exec(
      "INSERT INTO wax.users(id,name,email) VALUES('alice','Alice','alice@example.test'),('bob','Bob','bob@example.test'); INSERT INTO wax.collectors(id) VALUES('alice'),('bob'); INSERT INTO wax.releases(id,discogs_id,metadata) VALUES('20000000-0000-4000-8000-000000000001',2825456,'{}')",
    );
  });
  beforeEach(async () => {
    vi.clearAllMocks();
    vi.stubGlobal("fetch", state.fetch);
    state.collector.mockResolvedValue({ id: "alice" });
    await state.db!.exec(
      "TRUNCATE wax.copies,wax.catalog_cache,wax.integration_limits CASCADE",
    );
    for (const [id, owner, discogsId] of [
      [aliceCopy, "alice", release.id],
      [manualCopy, "alice", null],
      [bobCopy, "bob", release.id],
    ]) {
      await state.db!.query(
        "INSERT INTO wax.copies(id,owner_id,release_id,metadata,notes) VALUES($1,$2,'20000000-0000-4000-8000-000000000001',$3::jsonb,'PRIVATE NOTES')",
        [
          id,
          owner,
          JSON.stringify({
            discogsId,
            artist: "My reviewed artist",
            title: "My reviewed album",
          }),
        ],
      );
    }
  });
  afterAll(async () => {
    vi.unstubAllGlobals();
    await state.db?.close();
  });

  it("requires an invited collector on every explorer endpoint", async () => {
    state.collector.mockRejectedValue(new HttpError(401, "Please sign in."));
    const responses = await Promise.all([
      releaseRoute(request(), context(release.id)),
      artistRoute(request(), context(artist.id)),
      albumsRoute(request(), context(artist.id)),
      startRoute(request(), context(aliceCopy)),
    ]);
    expect(responses.map((response) => response.status)).toEqual([
      401, 401, 401, 401,
    ]);
    for (const response of responses)
      expect(response.headers.get("cache-control")).toBe("private, no-store");
    expect(state.fetch).not.toHaveBeenCalled();
  });

  it("protects private copy context and returns manual matching context without guessing", async () => {
    expect((await startRoute(request(), context(bobCopy))).status).toBe(404);
    expect(state.fetch).not.toHaveBeenCalled();
    const manual = await startRoute(request(), context(manualCopy));
    expect(manual.status).toBe(200);
    expect(await manual.json()).toEqual({
      copyId: manualCopy,
      artist: "My reviewed artist",
      title: "My reviewed album",
      discogsId: null,
      release: null,
    });
    expect(state.fetch).not.toHaveBeenCalled();
    state.fetch.mockResolvedValue(Response.json(release));
    const owned = await startRoute(request(), context(aliceCopy));
    const body = await owned.json();
    expect(owned.status).toBe(200);
    expect(body.release.id).toBe(release.id);
    expect(JSON.stringify(body)).not.toContain("PRIVATE NOTES");
    expect(state.fetch).toHaveBeenCalledOnce();
  });

  it("loads one requested branch and reuses only fresh normalized public cache", async () => {
    state.fetch.mockResolvedValue(Response.json(release));
    const first = await fetchExplorerRelease(release.id);
    expect(await fetchExplorerRelease(release.id)).toEqual(first);
    expect(state.fetch).toHaveBeenCalledOnce();
    const cache = (
      await state.db!.query<{ cache_key: string; payload: unknown }>(
        "SELECT cache_key,payload FROM wax.catalog_cache",
      )
    ).rows;
    expect(cache[0].cache_key).toBe("explorer:v2:release:2825456");
    expect(JSON.stringify(cache)).not.toContain("PRIVATE NOTES");
    expect(state.fetch.mock.calls[0][0]).toBe(
      "https://api.discogs.com/releases/2825456",
    );
    expect(state.fetch.mock.calls[0][1]).toMatchObject({
      redirect: "error",
      cache: "no-store",
    });
    await state.db!.exec(
      "UPDATE wax.catalog_cache SET fetched_at=now()-interval '7 hours'",
    );
    state.fetch.mockResolvedValue(
      Response.json({ ...release, title: "Updated title" }),
    );
    expect((await fetchExplorerRelease(release.id)).title).toBe(
      "Updated title",
    );
    expect(state.fetch).toHaveBeenCalledTimes(2);
  });

  it("coalesces a shared in-flight branch without fetching linked people", async () => {
    state.fetch.mockImplementation(async () => {
      await new Promise((resolve) => setTimeout(resolve, 20));
      return Response.json(release);
    });
    const results = await Promise.all([
      fetchExplorerRelease(release.id),
      fetchExplorerRelease(release.id),
    ]);
    expect(results[0]).toEqual(results[1]);
    expect(state.fetch).toHaveBeenCalledOnce();
  });

  it("loads artists and explicitly paginated discography without automatic expansion", async () => {
    state.fetch
      .mockResolvedValueOnce(Response.json(artist))
      .mockResolvedValueOnce(Response.json(discography));
    expect((await fetchExplorerArtist(artist.id)).name).toBe("Bill Evans");
    expect(state.fetch).toHaveBeenCalledOnce();
    expect(
      (await fetchExplorerDiscography(artist.id)).releases[0].mainReleaseId,
    ).toBe(333);
    expect(state.fetch.mock.calls[1][0]).toBe(
      "https://api.discogs.com/artists/252310/releases?per_page=20&page=1&sort=year&sort_order=asc",
    );
    expect(state.fetch).toHaveBeenCalledTimes(2);
  });

  it("validates IDs, page bounds, and unsupported query parameters", async () => {
    for (const id of [
      "0",
      "-1",
      "1e2",
      "9007199254740992",
      "https://evil.test",
    ])
      expect((await releaseRoute(request(), context(id))).status).toBe(400);
    for (const query of [
      "page=0",
      "page=501",
      "page=1.5",
      "page=NaN",
      "page=1&url=https://evil.test",
    ])
      expect(
        (
          await albumsRoute(
            request(`artists/252310/releases?${query}`),
            context(artist.id),
          )
        ).status,
      ).toBe(400);
    expect(state.fetch).not.toHaveBeenCalled();
  });

  it("keeps errors private, rejects wrong provider IDs, and permits recovery", async () => {
    state.fetch.mockResolvedValueOnce(
      new Response("", { status: 429, headers: { "Retry-After": "22" } }),
    );
    const limited = await releaseRoute(request(), context(release.id));
    expect(limited.status).toBe(429);
    expect(limited.headers.get("cache-control")).toBe("private, no-store");
    expect((await limited.json()).details.retryAfter).toBe(22);
    state.fetch.mockResolvedValueOnce(Response.json({ ...release, id: 999 }));
    expect((await releaseRoute(request(), context(release.id))).status).toBe(
      502,
    );
    state.fetch.mockResolvedValueOnce(Response.json(release));
    expect((await releaseRoute(request(), context(release.id))).status).toBe(
      200,
    );
    expect(state.fetch).toHaveBeenCalledTimes(3);
  });

  it("rejects oversized upstream JSON and never serves stale data on provider outage", async () => {
    state.fetch.mockResolvedValueOnce(
      new Response("x".repeat(2 * 1024 * 1024 + 1)),
    );
    expect((await releaseRoute(request(), context(release.id))).status).toBe(
      502,
    );
    state.fetch.mockResolvedValueOnce(Response.json(release));
    await fetchExplorerRelease(release.id);
    await state.db!.exec(
      "UPDATE wax.catalog_cache SET fetched_at=now()-interval '7 hours'",
    );
    state.fetch.mockRejectedValueOnce(new Error("network"));
    expect((await releaseRoute(request(), context(release.id))).status).toBe(
      503,
    );
    expect(
      (await state.db!.query("SELECT * FROM wax.catalog_cache")).rows,
    ).toEqual([]);
  });

  it("enforces the per-collector request budget before calling Discogs", async () => {
    await state.db!.query(
      "INSERT INTO wax.integration_limits(owner_id,operation,window_start,count) VALUES('alice','explorer',now(),120)",
    );
    const response = await artistRoute(request(), context(artist.id));
    expect(response.status).toBe(429);
    expect(state.fetch).not.toHaveBeenCalled();
  });
});
