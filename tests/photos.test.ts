import { describe, expect, it, vi } from "vitest";
vi.mock("../src/server/db", () => ({ getPool: vi.fn() }));
import { inspectPhoto, MAX_PHOTO_BYTES } from "../src/server/photos";

describe("private photo validation", () => {
  it("accepts supported bytes with the matching MIME type", () => {
    expect(
      inspectPhoto(Uint8Array.from([255, 216, 255, 0]), "image/jpeg").extension,
    ).toBe("jpg");
    expect(
      inspectPhoto(
        Uint8Array.from([137, 80, 78, 71, 13, 10, 26, 10]),
        "image/png",
      ).extension,
    ).toBe("png");
    expect(
      inspectPhoto(
        Uint8Array.from([82, 73, 70, 70, 0, 0, 0, 0, 87, 69, 66, 80]),
        "image/webp",
      ).extension,
    ).toBe("webp");
  });
  it("rejects disguised HTML, SVG, mismatched MIME, empty and oversized files", () => {
    expect(() =>
      inspectPhoto(
        new TextEncoder().encode('<svg onload="alert(1)"></svg>'),
        "image/png",
      ),
    ).toThrow();
    expect(() =>
      inspectPhoto(Uint8Array.from([255, 216, 255]), "image/png"),
    ).toThrow();
    expect(() => inspectPhoto(new Uint8Array(), "image/jpeg")).toThrow();
    expect(() =>
      inspectPhoto(new Uint8Array(MAX_PHOTO_BYTES + 1), "image/jpeg"),
    ).toThrow();
  });
});
