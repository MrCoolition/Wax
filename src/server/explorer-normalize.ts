import type {
  CreditCategory,
  ExplorerArtist,
  ExplorerCredit,
  ExplorerDiscography,
  ExplorerPerson,
  ExplorerRelation,
  ExplorerRelease,
  ExplorerTrack,
} from "@/lib/explorer";
import { normalizeRelease, safeDiscogsImage } from "./catalog";
import { HttpError } from "./http";

type JsonObject = Record<string, unknown>;
const object = (value: unknown): JsonObject =>
  value && typeof value === "object" && !Array.isArray(value)
    ? (value as JsonObject)
    : {};
const text = (value: unknown, maximum = 500): string =>
  typeof value === "string"
    ? value
        .replace(/[\x00-\x1f\x7f]/g, " ")
        .trim()
        .slice(0, maximum)
    : "";
const list = (value: unknown, maximum: number): unknown[] => {
  if (value === undefined || value === null) return [];
  if (!Array.isArray(value) || value.length > maximum) throw incomplete();
  return value;
};
const positiveId = (value: unknown): number | null =>
  typeof value === "number" && Number.isSafeInteger(value) && value > 0
    ? value
    : null;
const incomplete = () =>
  new HttpError(
    502,
    "Discogs returned incomplete or unusually large credits. Please try another edition or retry.",
  );
const releaseYear = (value: unknown): number | null => {
  const year =
    typeof value === "number" || typeof value === "string"
      ? Number(value)
      : NaN;
  return Number.isInteger(year) && year >= 1850 && year <= 2200 ? year : null;
};
function checkIdentity(raw: JsonObject, expectedId: number) {
  if (positiveId(raw.id) !== expectedId) throw incomplete();
}

export function splitCreditRoles(value: string): string[] {
  const roles: string[] = [];
  let bracketDepth = 0;
  let start = 0;
  for (let index = 0; index < value.length; index++) {
    if (value[index] === "[") bracketDepth++;
    else if (value[index] === "]") bracketDepth = Math.max(0, bracketDepth - 1);
    else if (value[index] === "," && bracketDepth === 0) {
      const role = value.slice(start, index).trim();
      if (role) roles.push(role);
      start = index + 1;
    }
  }
  const last = value.slice(start).trim();
  if (last) roles.push(last);
  if (roles.length > 20) throw incomplete();
  return [...new Set(roles)];
}

export function creditCategory(role: string): CreditCategory {
  const core = role.replace(/\[[\s\S]*$/, "").trim();
  if (
    /^(?:written[- ]by|compos(?:ed|er|ition)|lyrics?|words|music by|songwrit|arrang|orchestrat|adapted|translated)/i.test(
      core,
    )
  )
    return "writing";
  if (
    /produc|engineer|mix(?:ed|ing)?|master(?:ed|ing)?|recorded by|lacquer|cut(?:ting)? by|programmed by|programming|sound design/i.test(
      core,
    )
  )
    return "production";
  if (
    /^(?:performer|accompanied by|conductor|orchestra|choir|chorus|soloist|leader|band|dj|scratches|turntables|vocals?|singer|rap|spoken|beatbox|claps?|handclaps?|whistl)/i.test(
      core,
    ) ||
    /\b(?:guitar|bass|drums?|percussion|piano|organ|keyboards?|synth(?:esizer)?|saxophone|trumpet|trombone|tuba|horn|clarinet|flute|oboe|bassoon|violin|viola|cello|strings|harp|harmonica|accordion|vibraphone|marimba|xylophone|congas?|bongos?|tabla|sitar|banjo|mandolin|ukulele|fiddle|recorder|kazoo|euphonium|clavinet|mellotron|celesta)\b/i.test(
      core,
    )
  )
    return "performance";
  return "other";
}

function people(value: unknown): ExplorerPerson[] {
  return list(value, 100)
    .map(object)
    .map((person) => ({
      id: positiveId(person.id),
      name: text(person.name, 200),
    }))
    .filter((person) => person.name);
}

/** Exact position matching; ranges follow release order, never lexical order. */
export function resolveTrackReference(
  reference: string,
  tracks: ExplorerTrack[],
): ExplorerTrack[] | null {
  const byPosition = new Map<string, ExplorerTrack[]>();
  for (const track of tracks) {
    if (!track.position) continue;
    byPosition.set(track.position, [
      ...(byPosition.get(track.position) ?? []),
      track,
    ]);
  }
  const unique = (position: string) => {
    const matches = byPosition.get(position);
    return matches?.length === 1 && matches[0].type === "track"
      ? matches[0]
      : null;
  };
  const resolved = new Map<string, ExplorerTrack>();
  const chunks = reference.split(",").map((part) => part.trim());
  if (!chunks.length || chunks.length > 300 || chunks.some((part) => !part))
    return null;
  for (const chunk of chunks) {
    const exact = unique(chunk);
    if (exact) {
      resolved.set(exact.id, exact);
      continue;
    }
    const endpoints = chunk.split(/\s+to\s+/i);
    if (endpoints.length !== 2) return null;
    const first = unique(endpoints[0].trim());
    const last = unique(endpoints[1].trim());
    if (!first || !last) return null;
    const start = tracks.indexOf(first);
    const end = tracks.indexOf(last);
    if (start > end) return null;
    const range = tracks
      .slice(start, end + 1)
      .filter((track) => track.type === "track");
    // A duplicate or missing position inside a range is also ambiguous.
    if (
      range.some((track) => !track.position || unique(track.position) !== track)
    )
      return null;
    range.forEach((track) => resolved.set(track.id, track));
  }
  return [...resolved.values()];
}

export function normalizeExplorerRelease(
  value: unknown,
  expectedId: number,
  fetchedAt = new Date().toISOString(),
): ExplorerRelease {
  const raw = object(value);
  checkIdentity(raw, expectedId);
  if (!text(raw.title) || !Array.isArray(raw.tracklist)) throw incomplete();
  const metadata = normalizeRelease(raw);
  const tracks: ExplorerTrack[] = [];
  const trackSources = new Map<string, JsonObject>();
  function flatten(value: unknown, parent: string, depth: number) {
    if (depth > 5) throw incomplete();
    for (const [index, item] of list(value, 300).entries()) {
      if (tracks.length >= 300) throw incomplete();
      const source = object(item);
      const id = `${expectedId}:${parent}${index}`;
      const title = text(source.title);
      if (!title) throw incomplete();
      // Never shorten a reference into a different valid track position.
      if (typeof source.position === "string" && source.position.length > 80)
        throw incomplete();
      const track: ExplorerTrack = {
        id,
        position: text(source.position, 80),
        title,
        duration: text(source.duration, 40),
        type: source.type_ === "heading" ? "heading" : "track",
        credits: [],
      };
      tracks.push(track);
      trackSources.set(id, source);
      if (source.sub_tracks !== undefined)
        flatten(source.sub_tracks, `${parent}${index}.`, depth + 1);
    }
  }
  flatten(raw.tracklist, "", 0);
  const releaseCredits: ExplorerCredit[] = [];
  const unresolvedCredits: ExplorerCredit[] = [];
  let assignments = 0;
  const add = (target: ExplorerCredit[], credit: ExplorerCredit) => {
    if (
      target.some(
        (existing) =>
          existing.artistId === credit.artistId &&
          existing.name === credit.name &&
          existing.role === credit.role &&
          (target !== unresolvedCredits ||
            existing.trackReference === credit.trackReference),
      )
    )
      return;
    if (++assignments > 4000) throw incomplete();
    target.push(credit);
  };
  function credits(
    value: unknown,
    prefix: string,
    directTrack?: ExplorerTrack,
  ) {
    for (const [index, item] of list(value, 1000).entries()) {
      const source = object(item);
      const name = text(source.name, 200);
      const roleText = text(source.role);
      if (!name || !roleText) continue;
      if (typeof source.tracks === "string" && source.tracks.length > 2000)
        throw incomplete();
      const trackReference = text(source.tracks, 2000) || null;
      const roles = splitCreditRoles(roleText);
      const destinations = trackReference
        ? resolveTrackReference(trackReference, tracks)
        : null;
      for (const [roleIndex, role] of roles.entries()) {
        const credit: ExplorerCredit = {
          key: `${prefix}:${index}:${roleIndex}`,
          artistId: positiveId(source.id),
          name,
          role,
          category: creditCategory(role),
          trackReference,
        };
        if (directTrack?.type === "track") {
          // Tracklist credits explicitly belong to that track. Contradictory
          // explicit references remain unresolved instead of broadening scope.
          if (
            trackReference &&
            (!destinations ||
              !destinations.some((track) => track.id === directTrack.id))
          )
            add(unresolvedCredits, credit);
          else add(directTrack.credits, credit);
        } else if (directTrack) {
          add(unresolvedCredits, {
            ...credit,
            trackReference:
              trackReference ?? (directTrack.position || directTrack.title),
          });
        } else if (!trackReference) add(releaseCredits, credit);
        else if (!destinations) add(unresolvedCredits, credit);
        else destinations.forEach((track) => add(track.credits, credit));
      }
    }
  }
  credits(raw.extraartists, "release");
  for (const track of tracks) {
    const source = trackSources.get(track.id)!;
    credits(source.extraartists, track.id, track);
    if (track.type === "track") {
      for (const [index, person] of people(source.artists).entries())
        add(track.credits, {
          key: `${track.id}:artist:${index}`,
          artistId: person.id,
          name: person.name,
          role: "Track artist",
          category: "other",
          trackReference: track.position || null,
        });
    }
  }
  return {
    id: expectedId,
    title: metadata.title,
    artists: people(raw.artists),
    year: metadata.year,
    label: metadata.label,
    catalogNumber: metadata.catalogNumber,
    country: metadata.country,
    format: metadata.format,
    coverUrl: metadata.coverUrl,
    sourceUrl: `https://www.discogs.com/release/${expectedId}`,
    fetchedAt,
    tracks,
    releaseCredits,
    unresolvedCredits,
  };
}

function relations(value: unknown): ExplorerRelation[] {
  const found = new Map<number, ExplorerRelation>();
  for (const item of list(value, 500)) {
    const raw = object(item);
    const id = positiveId(raw.id);
    const name = text(raw.name, 200);
    if (id && name && !found.has(id))
      found.set(id, {
        id,
        name,
        active: typeof raw.active === "boolean" ? raw.active : null,
      });
  }
  return [...found.values()];
}

export function plainDiscogsProfile(value: unknown): string {
  if (typeof value !== "string") return "";
  const entityNames: Record<string, string> = {
    a: "artist",
    l: "label",
    r: "release",
    m: "master",
  };
  return (
    value
      .slice(0, 12_000)
      .replace(/\r\n?/g, "\n")
      .replace(/[\x00-\x09\x0b-\x1f\x7f]/g, " ")
      // Strip display wrappers into plain text; never render provider HTML or
      // construct links from profile markup. Numeric references remain explicit.
      .replace(/\[url(?:=[^\]]*)?\]([\s\S]*?)\[\/url\]/gi, "$1")
      .replace(/\[\/?(?:b|i|u)\]/gi, "")
      .replace(/\[br\s*\/?\]/gi, "\n")
      .replace(
        /\[(a|l|r|m)(?:=([^\]]+)|(\d+))\]/gi,
        (
          _match,
          kind: string,
          named: string | undefined,
          numeric: string | undefined,
        ) => {
          const reference = (named ?? numeric ?? "").trim();
          return /^\d+$/.test(reference)
            ? `Discogs ${entityNames[kind.toLowerCase()]} #${reference}`
            : reference;
        },
      )
      .replace(/\n{3,}/g, "\n\n")
      .trim()
      .slice(0, 6000)
  );
}

export function normalizeExplorerArtist(
  value: unknown,
  expectedId: number,
  fetchedAt = new Date().toISOString(),
): ExplorerArtist {
  const raw = object(value);
  checkIdentity(raw, expectedId);
  const name = text(raw.name, 200);
  if (!name) throw incomplete();
  const images = list(raw.images, 100).map(object);
  const primary = images.find((image) => image.type === "primary") ?? images[0];
  return {
    id: expectedId,
    name,
    realName: text(raw.realname, 200) || null,
    profile: plainDiscogsProfile(raw.profile),
    imageUrl:
      safeDiscogsImage(primary?.uri) || safeDiscogsImage(primary?.uri150),
    sourceUrl: `https://www.discogs.com/artist/${expectedId}`,
    groups: relations(raw.groups),
    members: relations(raw.members),
    aliases: relations(raw.aliases),
    fetchedAt,
  };
}

export function normalizeExplorerDiscography(
  value: unknown,
  artistId: number,
  page: number,
  fetchedAt = new Date().toISOString(),
): ExplorerDiscography {
  const raw = object(value);
  const pagination = object(raw.pagination);
  if (
    pagination.page !== page ||
    !Number.isSafeInteger(pagination.pages) ||
    Number(pagination.pages) < 0 ||
    !Number.isSafeInteger(pagination.items) ||
    Number(pagination.items) < 0 ||
    !Array.isArray(raw.releases)
  )
    throw incomplete();
  const seen = new Set<string>();
  const releases: ExplorerDiscography["releases"] = [];
  for (const value of list(raw.releases, 20)) {
    const album = object(value);
    const id = positiveId(album.id);
    const type = album.type;
    const title = text(album.title);
    if (!id || !title || (type !== "release" && type !== "master"))
      throw incomplete();
    const key = `${type}:${id}`;
    if (seen.has(key)) continue;
    seen.add(key);
    releases.push({
      id,
      type,
      mainReleaseId: type === "master" ? positiveId(album.main_release) : null,
      title,
      artist: text(album.artist, 300),
      year: releaseYear(album.year),
      thumbUrl: safeDiscogsImage(album.thumb),
      role: text(album.role, 100),
      sourceUrl: `https://www.discogs.com/${type}/${id}`,
    });
  }
  return {
    artistId,
    page,
    pages: Number(pagination.pages),
    total: Number(pagination.items),
    releases,
    fetchedAt,
  };
}
