import { describe, expect, it } from "vitest";
import {
  catalogSearchSchema,
  normalizeRelease,
  normalizeSearchCandidate,
  safeDiscogsImage,
} from "../src/server/catalog";

describe("Discogs candidate normalization", () => {
  it("preserves leading-zero barcodes and bounds search input", () => {
    expect(
      catalogSearchSchema.parse({ barcode: "0012345678905" }).barcode,
    ).toBe("0012345678905");
    expect(() =>
      catalogSearchSchema.parse({ barcode: "https://evil.test" }),
    ).toThrow();
    expect(() =>
      catalogSearchSchema.parse({ query: "a".repeat(201) }),
    ).toThrow();
  });
  it("maps exact pressing fields without inventing missing artwork or year", () => {
    const release = normalizeRelease({
      id: 2825456,
      title: "Kind Of Blue",
      artists: [{ name: "Miles Davis (2)" }],
      labels: [{ name: "Columbia", catno: "88697680571" }],
      formats: [{ name: "Vinyl", descriptions: ["LP", "Reissue"] }],
      country: "US",
      year: 0,
    });
    expect(release).toMatchObject({
      discogsId: 2825456,
      artist: "Miles Davis",
      title: "Kind Of Blue",
      label: "Columbia",
      catalogNumber: "88697680571",
      year: null,
      coverUrl: null,
      format: "Vinyl, LP, Reissue",
      sourceUrl: "https://www.discogs.com/release/2825456",
    });
  });
  it("separates the first artist/title delimiter and keeps later punctuation", () => {
    expect(
      normalizeSearchCandidate({
        id: 1,
        title: "Artist - Title - Side B",
        year: "1973",
        format: ["Vinyl", "LP"],
      }),
    ).toMatchObject({ artist: "Artist", title: "Title - Side B", year: 1973 });
  });
  it("never turns provider data into arbitrary external image URLs", () => {
    expect(safeDiscogsImage("https://i.discogs.com/art.jpg")).toBe(
      "https://i.discogs.com/art.jpg",
    );
    for (const image of [
      "http://i.discogs.com/x",
      "https://i.discogs.com.evil.test/x",
      "javascript:alert(1)",
      "https://secret@i.discogs.com/x",
      "https://evil.test/art.jpg",
    ])
      expect(safeDiscogsImage(image)).toBeNull();
  });
});
