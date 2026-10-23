import type { ShopArea } from "@/lib/shops";

type Tags = Record<string, unknown>;
const value = (input: unknown) =>
  typeof input === "string" ? input.trim().toLocaleLowerCase() : "";
const tokens = (input: unknown) =>
  value(input)
    .split(/[;,|]/)
    .map((part) => part.trim());
const vinylFlags = [
  "music:vinyl",
  "music:records",
  "sells:vinyl",
  "sells:records",
  "vinyl",
  "records",
];
const yes = new Set(["yes", "only", "new", "used", "second_hand"]);
const no = new Set(["no", "false", "0"]);

// Names without a record/vinyl signal need independently checked sales evidence.
// These are individual locations, not chain or name allowlists. A renamed/moved
// business, a closed listing, or explicit no-vinyl metadata cannot use this evidence.
const checkedVinylSellers = [
  {
    id: "node/12651098886",
    name: "rpm",
    latitude: 40.0148379,
    longitude: -83.0116131,
    checkedOn: "2026-10-22",
    sourceUrl:
      "https://www.coleminerecords.com/blogs/news/indie-retail-online-store-directory",
  },
  {
    id: "node/7899581385",
    name: "magnolia thunderpussy",
    latitude: 39.9859689,
    longitude: -83.005539,
    checkedOn: "2026-10-22",
    sourceUrl: "https://thunderpussy.com/Genre/694768",
  },
] as const;

export function hasVinylSalesEvidence(
  id: string,
  name: string,
  point: ShopArea,
  tags: Tags,
): boolean {
  if (
    Object.entries(tags).some(
      ([key, entry]) =>
        /^(disused|abandoned|demolished|removed|razed|closed)(:|$)/.test(key) &&
        value(entry) !== "" &&
        !no.has(value(entry)),
    ) ||
    ["closed", "off"].includes(value(tags.opening_hours)) ||
    value(tags.access) === "no" ||
    tokens(tags.shop).some((shop) =>
      ["no", "vacant", "closed", "disused", "abandoned"].includes(shop),
    ) ||
    vinylFlags.some((key) => no.has(value(tags[key])))
  )
    return false;

  // Explicit product tags can admit mixed instrument/music shops, but generic
  // descriptions, website paths, and DJ control/timecode vinyl are not evidence.
  const music = tokens(tags.music);
  const explicit =
    music.some((kind) =>
      ["vinyl", "records", "vinyl_records"].includes(kind),
    ) || vinylFlags.some((key) => yes.has(value(tags[key])));
  const checked = checkedVinylSellers.some(
    (seller) =>
      seller.id === id &&
      seller.name === value(name) &&
      Math.abs(seller.latitude - point.latitude) < 0.001 &&
      Math.abs(seller.longitude - point.longitude) < 0.001,
  );
  const shops = tokens(tags.shop);
  // Product tags also describe record sections inside book, second-hand, and
  // other mixed retailers. A shop category alone never proves vinyl sales.
  if (explicit && shops.some(Boolean)) return true;
  if (!shops.some((shop) => ["music", "musical_instrument"].includes(shop)))
    return false;
  if (
    music.length > 0 &&
    music.every((kind) =>
      ["cd", "cds", "cassette", "cassettes", "digital", "download"].includes(
        kind,
      ),
    )
  )
    return false;
  if (checked) return true;

  const instrumentSignal =
    /\b(?:percussion|guitars?|drums?|pianos?|instruments?|lessons?|sheet\s+music|music\s+(?:school|lessons)|recording\s+studios?|timecode|control\s*vinyl|accessories|flooring|wraps?|signs)\b/i;
  if (
    !shops.includes("music") ||
    shops.includes("musical_instrument") ||
    music.some((kind) =>
      ["sheet", "sheet_music", "instruments", "musical_instruments"].includes(
        kind,
      ),
    ) ||
    instrumentSignal.test(name) ||
    value(tags.studio) === "audio" ||
    ["music_school", "studio"].includes(value(tags.amenity))
  )
    return false;

  return /\b(?:records|record\s+(?:shop|store|exchange)|vinyl)\b/i.test(name);
}
