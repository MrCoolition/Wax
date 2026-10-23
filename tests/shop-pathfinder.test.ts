import { describe, expect, it } from "vitest";
import {
  digRouteUrl,
  moveDigStop,
  newShopsInExpandedArea,
} from "../src/lib/shop-pathfinder";
import type { NearbyShops, RecordShop } from "../src/lib/shops";

const center = { latitude: 40.02345, longitude: -83.04123 };
const stops = [
  { latitude: 40.1, longitude: -83.1 },
  { latitude: 40.2, longitude: -83.2 },
  { latitude: 40.3, longitude: -83.3 },
  { latitude: 40.4, longitude: -83.4 },
];

describe("dig routes", () => {
  it("preserves the chosen stop order and only shares an approximate origin", () => {
    const url = new URL(digRouteUrl(center, stops)!);
    expect(url.origin + url.pathname).toBe("https://www.google.com/maps/dir/");
    expect(url.searchParams.get("origin")).toBe("40.02,-83.04");
    expect(url.searchParams.get("waypoints")).toBe(
      "40.1,-83.1|40.2,-83.2|40.3,-83.3",
    );
    expect(url.searchParams.get("destination")).toBe("40.4,-83.4");
    expect(url.searchParams.get("api")).toBe("1");
    expect(url.searchParams.get("travelmode")).toBe("driving");
  });
  it("supports a single destination and refuses invalid or oversized routes", () => {
    expect(
      new URL(digRouteUrl(center, stops.slice(0, 1))!).searchParams.has(
        "waypoints",
      ),
    ).toBe(false);
    expect(digRouteUrl(center, [])).toBeNull();
    expect(digRouteUrl(center, [...stops, stops[0]])).toBeNull();
    expect(digRouteUrl(center, [{ latitude: NaN, longitude: 0 }])).toBeNull();
    expect(digRouteUrl({ latitude: 91, longitude: 0 }, stops)).toBeNull();
  });
  it("reorders without mutating or losing stops and respects both ends", () => {
    const original = ["A", "B", "C"];
    expect(moveDigStop(original, 1, -1)).toEqual(["B", "A", "C"]);
    expect(moveDigStop(original, 1, 1)).toEqual(["A", "C", "B"]);
    expect(moveDigStop(original, 0, -1)).toEqual(original);
    expect(moveDigStop(original, 2, 1)).toEqual(original);
    expect(moveDigStop(original, -1, 1)).toEqual(original);
    expect(original).toEqual(["A", "B", "C"]);
  });
});

describe("expanded search feedback", () => {
  const result: NearbyShops = {
    center,
    radiusKm: 10,
    fetchedAt: "2026-10-23T12:00:00Z",
    limited: false,
    shops: [{ id: "node/1" } as RecordShop],
  };
  it("counts newly found shops and reports honest zero when the same shops return", () => {
    expect(
      newShopsInExpandedArea(result, {
        ...result,
        radiusKm: 50,
        shops: [
          ...result.shops,
          { id: "directory/farther-records" } as RecordShop,
        ],
      }),
    ).toBe(1);
    expect(newShopsInExpandedArea(result, { ...result, radiusKm: 25 })).toBe(0);
  });
  it("does not compare different places, contractions, or an initial search", () => {
    expect(
      newShopsInExpandedArea(result, {
        ...result,
        center: stops[0],
        radiusKm: 50,
      }),
    ).toBeNull();
    expect(newShopsInExpandedArea(result, result)).toBeNull();
    expect(newShopsInExpandedArea(null, result)).toBeNull();
  });
});
