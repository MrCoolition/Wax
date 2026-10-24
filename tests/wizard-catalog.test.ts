import { beforeEach, describe, expect, it, vi } from "vitest";
const state = vi.hoisted(() => ({
  albums: vi.fn(),
  pressings: vi.fn(),
  release: vi.fn(),
  artist: vi.fn(),
  discography: vi.fn(),
}));
vi.mock("../src/server/catalog", async (original) => ({
  ...(await original<typeof import("../src/server/catalog")>()),
  searchAlbums: state.albums,
  searchCatalogPage: state.pressings,
}));
vi.mock("../src/server/explorer", () => ({
  fetchExplorerRelease: state.release,
  fetchExplorerArtist: state.artist,
  fetchExplorerDiscography: state.discography,
}));
import {
  createWizardCatalog,
  compactReleaseEvidence,
} from "../src/server/wizard-catalog";
import { normalizeExplorerRelease } from "../src/server/explorer-normalize";

const album = {
  masterId: 8883,
  artist: "Michael Jackson",
  title: "Thriller",
  originalYear: 1982,
  coverUrl: null,
  sourceUrl: "https://www.discogs.com/master/8883",
};
const release = normalizeExplorerRelease(
  {
    id: 2825456,
    title: "Kind Of Blue",
    year: 2010,
    artists: [{ id: 23755, name: "Miles Davis" }],
    labels: [{ name: "Columbia", catno: "CS 8163" }],
    country: "US",
    formats: [{ name: "Vinyl", descriptions: ["LP"] }],
    tracklist: [
      { position: "A1", title: "So What", type_: "track" },
      { position: "A2", title: "Freddie Freeloader", type_: "track" },
    ],
    extraartists: [
      { id: 252310, name: "Bill Evans", role: "Piano", tracks: "A1" },
      { id: 252308, name: "Wynton Kelly", role: "Piano", tracks: "A2" },
      { id: 23755, name: "Miles Davis", role: "Trumpet", tracks: "" },
      { id: 555, name: "Unknown guest", role: "Vocals", tracks: "Z9" },
    ],
    identifiers: [
      {
        type: "Matrix / Runout",
        value: "XSM-47326-1A",
        description: "Side A etched",
      },
      { type: "Barcode", value: "001234567890" },
    ],
    companies: [
      {
        id: 123,
        name: "Example plant",
        entity_type_name: "Pressed By",
        catno: "ABC",
      },
    ],
    notes:
      "[b]Catalog pressing notes.[/b] [url=https://evil.test]Plain display text[/url]",
  },
  2825456,
  "2026-10-24T12:00:00Z",
);
const artist = {
  id: 252308,
  name: "Wynton Kelly",
  realName: "Wynton Charles Kelly",
  profile: "Pianist.",
  sourceUrl: "https://evil.test/forged",
  imageUrl: null,
  groups: [{ id: 12345, name: "Miles Davis Sextet", active: false }],
  members: [],
  aliases: [],
  fetchedAt: "2026-10-24T12:00:00Z",
};
type Executable = { execute?: unknown };
async function call(
  tool: Executable,
  input: unknown,
  abortSignal?: AbortSignal,
) {
  return (
    tool.execute as (
      input: unknown,
      options: {
        toolCallId: string;
        messages: never[];
        abortSignal?: AbortSignal;
      },
    ) => Promise<unknown>
  )(input, { toolCallId: "test", messages: [], abortSignal });
}

describe("bounded public vinyl knowledge tools", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    state.albums.mockResolvedValue({
      albums: [album],
      pagination: { page: 1, pages: 1, total: 1, hasMore: false },
    });
    state.pressings.mockResolvedValue({
      candidates: [
        {
          discogsId: 2825456,
          artist: "Miles Davis",
          title: "Kind Of Blue",
          year: 2010,
          originalYear: 1959,
          country: "US",
          label: "Columbia",
          catalogNumber: "CS 8163",
          format: "Vinyl, LP",
          sourceUrl: "https://evil.test/",
        },
      ],
      pagination: { total: 1 },
    });
    state.release.mockResolvedValue(release);
    state.artist.mockResolvedValue(artist);
    state.discography.mockResolvedValue({
      artistId: 252308,
      page: 1,
      pages: 2,
      total: 25,
      fetchedAt: artist.fetchedAt,
      releases: [
        {
          id: 11,
          type: "master",
          mainReleaseId: 22,
          title: "Kelly Blue",
          artist: "Wynton Kelly",
          year: 1959,
          role: "Main",
          sourceUrl: "https://www.discogs.com/master/11",
        },
      ],
    });
  });

  it("reads exact track credits without spreading album-wide or ambiguous credits onto a song", async () => {
    const context = createWizardCatalog();
    const result = await call(context.tools.readReleaseCredits, {
      releaseId: 2825456,
      trackTitle: "Freddie Freeloader",
    });
    expect(result).toMatchObject({
      sourceId: "release:2825456",
      pressingYear: 2010,
      trackMatchCount: 1,
      tracks: [
        {
          title: "Freddie Freeloader",
          credits: [{ name: "Wynton Kelly", role: "Piano" }],
        },
      ],
      releaseCredits: [{ name: "Miles Davis", role: "Trumpet" }],
      unresolvedCredits: [{ name: "Unknown guest", trackReference: "Z9" }],
    });
    expect(JSON.stringify(result)).not.toContain("Bill Evans");
    expect(context.sources.get("release:2825456")).toEqual({
      id: "release:2825456",
      kind: "release",
      title: "Miles Davis — Kind Of Blue",
      url: "https://www.discogs.com/release/2825456",
    });
    expect(compactReleaseEvidence(release, "No such song")).toMatchObject({
      tracks: [],
      trackMatchCount: 0,
    });
  });

  it("preserves provider runout/barcode text and company roles without creating mastering claims", async () => {
    const context = createWizardCatalog();
    const result = await call(context.tools.readReleaseCredits, {
      releaseId: 2825456,
      trackTitle: null,
    });
    expect(result).toMatchObject({
      identifiers: [
        {
          type: "Matrix / Runout",
          value: "XSM-47326-1A",
          description: "Side A etched",
        },
        { type: "Barcode", value: "001234567890", description: "" },
      ],
      companies: [
        {
          id: 123,
          name: "Example plant",
          role: "Pressed By",
          catalogNumber: "ABC",
        },
      ],
      notes: "Catalog pressing notes. Plain display text",
    });
    expect(JSON.stringify(result)).not.toContain("evil.test");
    expect(JSON.stringify(result)).not.toContain("mastered by");
  });

  it("queries explicit pressing details and returns separate album/pressing years with vetted source URLs", async () => {
    const context = createWizardCatalog();
    const result = await call(context.tools.findVinylPressings, {
      artist: "Miles Davis",
      title: "Kind Of Blue",
      catalogNumber: "CS 8163",
      country: "US",
      year: 2010,
    });
    expect(state.pressings).toHaveBeenCalledWith({
      artist: "Miles Davis",
      title: "Kind Of Blue",
      catalogNumber: "CS 8163",
      country: "US",
      year: 2010,
    });
    expect(result).toMatchObject({
      candidates: [
        {
          releaseId: 2825456,
          pressingYear: 2010,
          originalYear: 1959,
          sourceId: "release:2825456",
        },
      ],
    });
    expect(context.sources.get("release:2825456")?.url).toBe(
      "https://www.discogs.com/release/2825456",
    );
    expect(JSON.stringify(result)).not.toContain("evil.test");
  });

  it("retains real artist connections and distinguishes a master from its representative release", async () => {
    const context = createWizardCatalog();
    expect(
      await call(context.tools.readArtistConnections, { artistId: 252308 }),
    ).toMatchObject({
      name: "Wynton Kelly",
      groups: [{ id: 12345, name: "Miles Davis Sextet", active: false }],
      sourceId: "artist:252308",
    });
    expect(context.sources.get("artist:252308")?.url).toBe(
      "https://www.discogs.com/artist/252308",
    );
    expect(
      await call(context.tools.readArtistDiscography, {
        artistId: 252308,
        page: 1,
      }),
    ).toMatchObject({
      hasMore: true,
      releases: [{ id: 11, type: "master", mainReleaseId: 22, role: "Main" }],
    });
    expect(state.discography).toHaveBeenCalledWith(252308, 1);
    expect(JSON.stringify([...context.sources.values()])).not.toContain(
      "evil.test",
    );
  });

  it("enforces one six-call budget across concurrent tools and never shares request source maps", async () => {
    const context = createWizardCatalog();
    const work = Array.from({ length: 8 }, (_, index) =>
      index % 2
        ? call(context.tools.readArtistConnections, { artistId: 252308 })
        : call(context.tools.searchAlbums, {
            artist: "Michael Jackson",
            title: "Thriller",
          }),
    );
    const results = await Promise.all(work);
    expect(
      state.albums.mock.calls.length + state.artist.mock.calls.length,
    ).toBe(6);
    expect(results.slice(6)).toEqual([
      expect.objectContaining({ available: false }),
      expect.objectContaining({ available: false }),
    ]);
    expect(context.catalog.get(8883)).toEqual(album);
    expect(context.sources.has("master:8883")).toBe(true);
    const second = createWizardCatalog();
    expect(second.catalog.size).toBe(0);
    expect(second.sources.size).toBe(0);
  });

  it("invalid IDs, unsupported fields, aborted requests and provider failure create no invented evidence", async () => {
    const context = createWizardCatalog();
    for (const input of [
      { releaseId: -1, trackTitle: null },
      { releaseId: 9007199254740992, trackTitle: null },
      { releaseId: 42, trackTitle: null, ownerId: "bob" },
    ])
      expect(await call(context.tools.readReleaseCredits, input)).toMatchObject(
        { available: false },
      );
    expect(state.release).not.toHaveBeenCalled();
    const controller = new AbortController();
    controller.abort();
    expect(
      await call(
        context.tools.readArtistConnections,
        { artistId: 252308 },
        controller.signal,
      ),
    ).toMatchObject({ available: false });
    expect(state.artist).not.toHaveBeenCalled();
    state.release.mockRejectedValueOnce(new Error("Provider down"));
    expect(
      await call(context.tools.readReleaseCredits, {
        releaseId: 42,
        trackTitle: null,
      }),
    ).toMatchObject({ available: false });
    expect(context.sources.size).toBe(0);
    expect(context.catalog.size).toBe(0);
  });

  it("bounds model evidence and album suggestions and leaves missing metadata unknown", async () => {
    const context = createWizardCatalog();
    state.albums.mockResolvedValue({
      albums: Array.from({ length: 20 }, (_, index) => ({
        ...album,
        masterId: index + 1,
      })),
    });
    const result = await call(context.tools.searchAlbums, {
      artist: "Artist",
      title: "Album",
    });
    expect((result as { albums: unknown[] }).albums).toHaveLength(6);
    expect(context.catalog.size).toBe(6);
    const empty = normalizeExplorerRelease(
      {
        id: 1,
        title: "Unknown credits",
        artists: [{ name: "Artist" }],
        tracklist: [],
      },
      1,
    );
    expect(compactReleaseEvidence(empty, null)).toMatchObject({
      tracks: [],
      releaseCredits: [],
      unresolvedCredits: [],
      identifiers: [],
      companies: [],
      notes: "",
    });
    const huge = compactReleaseEvidence(
      {
        ...release,
        tracks: Array.from({ length: 90 }, () => release.tracks[0]),
        identifiers: Array.from({ length: 90 }, () => release.identifiers![0]),
      },
      null,
    );
    expect(huge.tracks).toHaveLength(40);
    expect(huge.identifiers).toHaveLength(40);
    expect(huge.tracksTruncated).toBe(true);
  });
});
