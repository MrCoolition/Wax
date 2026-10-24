import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  DEFAULT_COPY,
  EMPTY_CANDIDATE,
  type RecordCopy,
} from "../src/lib/contracts";
const mocks = vi.hoisted(() => ({
  generate: vi.fn(),
  search: vi.fn(),
  library: vi.fn(),
  query: vi.fn(),
}));
vi.mock("ai", async (original) => ({
  ...(await original<typeof import("ai")>()),
  generateText: mocks.generate,
}));
vi.mock("../src/server/catalog", () => ({ searchAlbums: mocks.search }));
vi.mock("../src/server/collection", () => ({
  getLibrary: mocks.library,
  approveChanges: vi.fn(),
}));
vi.mock("../src/server/db", () => ({
  getPool: () => ({ query: mocks.query }),
  withTransaction: vi.fn(),
}));
vi.mock("../src/server/integration-common", () => ({
  getAiModel: () => "test",
  consumeIntegrationLimit: vi.fn(),
  aiUnavailable: (error: unknown) => {
    throw error;
  },
}));
import {
  askWizard,
  relevantRecordIds,
  verifiedRecommendations,
} from "../src/server/wizard";

const owned = "23bca1b7-2dea-40ee-9949-4b323e9247a5";
const record: RecordCopy = {
  ...EMPTY_CANDIDATE,
  ...DEFAULT_COPY,
  id: owned,
  releaseId: owned,
  version: 1,
  artist: "Steely Dan",
  title: "Aja",
  addedAt: "2026-01-01",
  updatedAt: "2026-01-01",
  crateIds: [],
  photoIds: [],
  spinCount: 0,
  lastPlayedAt: null,
};
const album = {
  masterId: 8883,
  artist: "Michael Jackson",
  title: "Thriller",
  originalYear: 1982,
  coverUrl: null,
  sourceUrl: "https://www.discogs.com/master/8883",
};
type Options = {
  instructions: string;
  messages: unknown[];
  tools: {
    searchAlbums: {
      execute: (input: {
        artist: string;
        title: string;
      }) => Promise<{
        albums: (typeof album)[];
        available?: boolean;
        notice: string | null;
      }>;
    };
  };
};

describe("Wizard discovery without weakening private collection boundaries", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.library.mockResolvedValue({ records: [record], crates: [] });
    mocks.search.mockResolvedValue({
      albums: [album],
      pagination: { page: 1, pages: 1, total: 1, hasMore: false },
    });
  });
  it("uses read-only catalog results to answer a buying follow-up without unrelated owned cards", async () => {
    mocks.generate.mockImplementation(async (options: Options) => {
      expect(options.instructions).toContain("NOT restricted to owned records");
      expect(options.messages.at(-1)).toEqual({
        role: "user",
        content: "nah bro, I mean I want to buy it",
      });
      expect(JSON.stringify(options.messages)).toContain("Michael Jackson");
      const found = await options.tools.searchAlbums.execute({
        artist: "Michael Jackson",
        title: "Thriller",
      });
      expect(found.albums[0].masterId).toBe(8883);
      return {
        output: {
          reply:
            "Start with Thriller for its mix of dance-floor grooves and pop hooks.",
          recordIds: [owned],
          recommendations: [
            {
              masterId: 8883,
              reason: "A mix of dance-floor grooves and pop hooks.",
            },
          ],
          sourceIds: ["master:8883", "master:999999", "master:8883"],
          proposal: null,
        },
      };
    });
    const result = await askWizard("alice", {
      message: "nah bro, I mean I want to buy it",
      history: [
        {
          role: "user",
          content: "Which Michael Jackson record should I grab?",
        },
      ],
    });
    expect(result.recommendations).toEqual([
      { ...album, reason: "A mix of dance-floor grooves and pop hooks." },
    ]);
    expect(result.recordIds).toEqual([]);
    expect(result.sources).toEqual([
      {
        id: "master:8883",
        kind: "master",
        title: "Michael Jackson — Thriller",
        url: album.sourceUrl,
      },
    ]);
    expect(result.proposal).toBeNull();
    expect(mocks.query).not.toHaveBeenCalled();
    expect(mocks.search).toHaveBeenCalledWith({
      artist: "Michael Jackson",
      title: "Thriller",
    });
  });
  it("helps an empty collection discover music", async () => {
    mocks.library.mockResolvedValue({ records: [], crates: [] });
    mocks.generate.mockResolvedValue({
      output: {
        reply: "Start with a record you love front to back.",
        recordIds: [],
        recommendations: [],
        proposal: null,
      },
    });
    const result = await askWizard("new-collector", {
      message: "Where should I start?",
    });
    expect(mocks.generate).toHaveBeenCalledOnce();
    expect(result.reply).toContain("front to back");
    expect(mocks.query).not.toHaveBeenCalled();
  });
  it("keeps answering during catalog outages without fabricating verified cards", async () => {
    mocks.search.mockRejectedValue(new Error("throttled"));
    mocks.generate.mockImplementation(async (options: Options) => {
      const found = await options.tools.searchAlbums.execute({
        artist: "Michael Jackson",
        title: "Thriller",
      });
      expect(found.available).toBe(false);
      expect(found.notice).toContain("unavailable");
      return {
        output: {
          reply: "Thriller is a strong musical starting point.",
          recordIds: [],
          recommendations: [
            { masterId: 8883, reason: "Invented match must not become a card" },
          ],
          proposal: null,
        },
      };
    });
    const result = await askWizard("alice", {
      message: "Recommend a Michael Jackson album.",
    });
    expect(result.recommendations).toEqual([]);
    expect(result.reply).toContain("Thriller");
  });
  it("bounds catalog traffic and cannot make model IDs into collection links", async () => {
    mocks.generate.mockImplementation(async (options: Options) => {
      for (let i = 0; i < 8; i++)
        await options.tools.searchAlbums.execute({
          artist: "Michael Jackson",
          title: "Thriller",
        });
      return {
        output: {
          reply: "Play this.",
          recordIds: ["791a265b-1987-422f-aa08-3960fb5d3825"],
          recommendations: [],
          proposal: null,
        },
      };
    });
    await expect(
      askWizard("alice", { message: "Give me a record." }),
    ).rejects.toMatchObject({ status: 502 });
    expect(mocks.search).toHaveBeenCalledTimes(6);
    expect(mocks.query).not.toHaveBeenCalled();
  });
  it("deduplicates catalog cards and only links named or proposed owned copies", () => {
    const picks = [
      { masterId: 8883, reason: "First" },
      { masterId: 8883, reason: "Duplicate" },
      { masterId: 123, reason: "Unknown" },
    ];
    expect(verifiedRecommendations(picks, new Map([[8883, album]]))).toEqual([
      { ...album, reason: "First" },
    ]);
    expect(
      relevantRecordIds("Try Aja tonight.", [owned], [record], null),
    ).toEqual([owned]);
    expect(
      relevantRecordIds("A new direction.", [owned], [record], null),
    ).toEqual([]);
    expect(
      relevantRecordIds("Review this move.", [owned], [record], {
        changes: [
          { recordId: owned, version: 1, changes: { location: "Shelf A" } },
        ],
      }),
    ).toEqual([owned]);
  });
});
