import { tool } from "ai";
import { z } from "zod";
import type { CatalogAlbum } from "../lib/catalog-search";
import type { ExplorerCredit, ExplorerRelease } from "../lib/explorer";
import type { WizardCatalogSource } from "../lib/wizard";
import { searchAlbums, searchCatalogPage } from "./catalog";
import {
  fetchExplorerArtist,
  fetchExplorerDiscography,
  fetchExplorerRelease,
} from "./explorer";

export const WIZARD_CATALOG_CALL_LIMIT = 6;
const publicText = z.string().trim().min(1).max(200);
const providerId = z.number().int().positive().max(Number.MAX_SAFE_INTEGER);
const albumInput = z.object({ artist: publicText, title: publicText }).strict();
const pressingInput = z
  .object({
    artist: publicText.nullable(),
    title: publicText.nullable(),
    catalogNumber: publicText.nullable(),
    country: publicText.nullable(),
    year: z.number().int().min(1850).max(2200).nullable(),
  })
  .strict();
const releaseInput = z
  .object({ releaseId: providerId, trackTitle: publicText.nullable() })
  .strict();
const artistInput = z.object({ artistId: providerId }).strict();
const discographyInput = artistInput
  .extend({ page: z.number().int().min(1).max(5) })
  .strict();
const unavailable = {
  available: false as const,
  notice:
    "Catalog evidence is unavailable. You may give clearly framed general music advice, but do not invent retrieved credits, identifiers, sources, prices or stock.",
};
const normalize = (value: string) =>
  value
    .normalize("NFKC")
    .toLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, " ")
    .trim();
const briefCredit = (credit: ExplorerCredit) => ({
  artistId: credit.artistId,
  name: credit.name,
  role: credit.role,
  category: credit.category,
  trackReference: credit.trackReference,
});

/** Keep album, track and unresolved credit scopes distinct in evidence sent to the model. */
export function compactReleaseEvidence(
  release: ExplorerRelease,
  trackTitle: string | null,
) {
  const requested = trackTitle ? normalize(trackTitle) : null;
  const matches = requested
    ? release.tracks.filter(
        (track) =>
          track.type === "track" && normalize(track.title) === requested,
      )
    : release.tracks;
  const tracks = matches.slice(0, 40);
  let remainingCredits = 160;
  const takeCredits = (credits: ExplorerCredit[], limit: number) => {
    const taken = credits.slice(0, Math.min(limit, remainingCredits));
    remainingCredits -= taken.length;
    return taken.map(briefCredit);
  };
  const trackEvidence = tracks.map((track) => ({
    position: track.position,
    title: track.title,
    duration: track.duration,
    type: track.type,
    credits: takeCredits(track.credits, 30),
    creditsTruncated: track.credits.length > 30 || remainingCredits === 0,
  }));
  return {
    releaseId: release.id,
    title: release.title,
    artists: release.artists.slice(0, 20),
    pressingYear: release.year,
    label: release.label,
    catalogNumber: release.catalogNumber,
    country: release.country,
    format: release.format,
    fetchedAt: release.fetchedAt,
    tracks: trackEvidence,
    trackMatchCount: requested ? matches.length : null,
    tracksTruncated: matches.length > tracks.length,
    releaseCredits: takeCredits(release.releaseCredits, 60),
    unresolvedCredits: takeCredits(release.unresolvedCredits, 30),
    identifiers: (release.identifiers || []).slice(0, 40),
    companies: (release.companies || []).slice(0, 40),
    notes: (release.notes || "").slice(0, 3000),
    notice:
      "Track credits apply only to their stated track. Release credits are album-level and must not be assigned automatically to every song. Unresolved credits have ambiguous scope. Missing credits or identifiers mean unknown, not absent. This is one catalog edition; notes and identifiers do not independently establish sound quality.",
  };
}

/** Creates isolated, read-only public-catalog tools for one authenticated Wizard answer. */
export function createWizardCatalog(options: { signal?: AbortSignal } = {}) {
  const catalog = new Map<number, CatalogAlbum>();
  const sources = new Map<string, WizardCatalogSource>();
  let calls = 0;
  const source = (
    kind: WizardCatalogSource["kind"],
    id: number,
    title: string,
  ) => {
    providerId.parse(id);
    const key = `${kind}:${id}`;
    const item = {
      id: key,
      kind,
      title: title.slice(0, 300),
      url: `https://www.discogs.com/${kind}/${id}`,
    };
    sources.set(key, item);
    return item;
  };
  async function run<T>(
    action: () => Promise<T>,
    abortSignal?: AbortSignal,
  ): Promise<T | typeof unavailable> {
    if (options.signal?.aborted || abortSignal?.aborted) return unavailable;
    if (calls >= WIZARD_CATALOG_CALL_LIMIT)
      return {
        available: false,
        notice:
          "Catalog lookup budget reached. Answer only from evidence already returned, or give clearly framed general music advice.",
      };
    calls++;
    try {
      return await action();
    } catch {
      return unavailable;
    }
  }
  const tools = {
    searchAlbums: tool({
      description:
        "Find real album groups by public artist and title for buying or music discussion. These are albums, not confirmed pressings or stock. Return masterIds in recommendations. Never send private notes or collection details as search terms.",
      inputSchema: albumInput,
      execute: async (raw, execution) =>
        run(async () => {
          const input = albumInput.parse(raw);
          const found = await searchAlbums(input);
          const albums = found.albums.slice(0, 6);
          for (const album of albums) {
            catalog.set(album.masterId, album);
            source(
              "master",
              album.masterId,
              `${album.artist} — ${album.title}`,
            );
          }
          return {
            albums,
            sourceIds: albums.map((album) => `master:${album.masterId}`),
            notice: found.notice ?? null,
          };
        }, execution?.abortSignal),
    }),
    findVinylPressings: tool({
      description:
        "Search real vinyl pressings by artist/title and optional catalog number, country and pressing year. Use null for unknown fields. Select a returned releaseId for readReleaseCredits to verify musicians, mastering, companies or runouts. Never claim these search candidates confirm someone's copy, current stock, price or pressing quality.",
      inputSchema: pressingInput,
      execute: async (raw, execution) =>
        run(async () => {
          const input = pressingInput.parse(raw);
          if (!input.artist && !input.title && !input.catalogNumber)
            return {
              candidates: [],
              notice: "Supply an artist, album title or catalog number.",
            };
          const found = await searchCatalogPage({
            artist: input.artist ?? undefined,
            title: input.title ?? undefined,
            catalogNumber: input.catalogNumber ?? undefined,
            country: input.country ?? undefined,
            year: input.year ?? undefined,
          });
          const candidates = found.candidates
            .slice(0, 8)
            .flatMap((candidate) => {
              if (!candidate.discogsId) return [];
              const item = source(
                "release",
                candidate.discogsId,
                `${candidate.artist} — ${candidate.title}`,
              );
              return [
                {
                  releaseId: candidate.discogsId,
                  artist: candidate.artist,
                  title: candidate.title,
                  pressingYear: candidate.year,
                  originalYear: candidate.originalYear ?? null,
                  label: candidate.label,
                  catalogNumber: candidate.catalogNumber,
                  country: candidate.country,
                  format: candidate.format,
                  sourceId: item.id,
                },
              ];
            });
          return {
            candidates,
            totalMatches: found.pagination.total,
            notice:
              found.notice ??
              "Search summaries identify candidates; fetch the release before asserting detailed credits or mastering.",
          };
        }, execution?.abortSignal),
    }),
    readReleaseCredits: tool({
      description:
        "Read a real Discogs release's musicians, instruments, writers, production/mastering credits, tracklist, companies and pressing identifiers. Use a returned releaseId or the user's explicit Discogs release ID. trackTitle can name an exact song (null for whole album). Keep track-specific, album-level and ambiguous credits distinct; never infer all tracks from a blank scope.",
      inputSchema: releaseInput,
      execute: async (raw, execution) =>
        run(async () => {
          const { releaseId, trackTitle } = releaseInput.parse(raw);
          const release = await fetchExplorerRelease(releaseId);
          const item = source(
            "release",
            release.id,
            `${release.artists.map((artist) => artist.name).join(" & ")} — ${release.title}`,
          );
          return {
            ...compactReleaseEvidence(release, trackTitle),
            sourceId: item.id,
          };
        }, execution?.abortSignal),
    }),
    readArtistConnections: tool({
      description:
        "Read a credited musician/producer's Discogs profile, group memberships, members and aliases. Use a returned artistId. These are catalog relationships, not proof of particular session or track participation; verify that in release credits.",
      inputSchema: artistInput,
      execute: async (raw, execution) =>
        run(async () => {
          const { artistId } = artistInput.parse(raw);
          const artist = await fetchExplorerArtist(artistId);
          const item = source("artist", artist.id, artist.name);
          return {
            artistId: artist.id,
            name: artist.name,
            realName: artist.realName,
            profile: artist.profile.slice(0, 3000),
            groups: artist.groups.slice(0, 25),
            members: artist.members.slice(0, 25),
            aliases: artist.aliases.slice(0, 20),
            relationshipsTruncated:
              artist.groups.length > 25 ||
              artist.members.length > 25 ||
              artist.aliases.length > 20,
            sourceId: item.id,
            fetchedAt: artist.fetchedAt,
          };
        }, execution?.abortSignal),
    }),
    readArtistDiscography: tool({
      description:
        "Read a page of a musician's related records and their catalog-listed role. Use returned artistId, page1-5. Master entries are album groups; mainReleaseId is a representative edition for deeper credits, not necessarily the earliest/best pressing. A role such as Main or Appearance does not prove an instrument or specific track.",
      inputSchema: discographyInput,
      execute: async (raw, execution) =>
        run(async () => {
          const { artistId, page } = discographyInput.parse(raw);
          const found = await fetchExplorerDiscography(artistId, page);
          const item = source(
            "artist",
            artistId,
            sources.get(`artist:${artistId}`)?.title ??
              `Discogs artist #${artistId} — discography`,
          );
          return {
            artistId,
            page: found.page,
            totalMatches: found.total,
            hasMore: page < Math.min(5, found.pages),
            releases: found.releases.slice(0, 20).map((release) => ({
              id: release.id,
              type: release.type,
              mainReleaseId: release.mainReleaseId,
              title: release.title,
              artist: release.artist,
              year: release.year,
              role: release.role,
            })),
            sourceId: item.id,
            fetchedAt: found.fetchedAt,
          };
        }, execution?.abortSignal),
    }),
  };
  return { tools, catalog, sources };
}
