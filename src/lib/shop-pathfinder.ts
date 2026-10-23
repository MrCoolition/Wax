import { approximateShopArea, type NearbyShops, type ShopArea } from "./shops";

// Three intermediate waypoints plus a destination works on mobile Maps URLs.
export const MAX_DIG_STOPS = 4;

function validPoint(point: ShopArea): boolean {
  return (
    Number.isFinite(point.latitude) &&
    Math.abs(point.latitude) <= 90 &&
    Number.isFinite(point.longitude) &&
    Math.abs(point.longitude) <= 180
  );
}

export function digRouteUrl(
  center: ShopArea,
  stops: readonly ShopArea[],
): string | null {
  if (
    !validPoint(center) ||
    !stops.length ||
    stops.length > MAX_DIG_STOPS ||
    !stops.every(validPoint)
  )
    return null;
  const point = (area: ShopArea) => `${area.latitude},${area.longitude}`;
  const url = new URL("https://www.google.com/maps/dir/");
  url.searchParams.set("api", "1");
  url.searchParams.set("origin", point(approximateShopArea(center)));
  url.searchParams.set("destination", point(stops[stops.length - 1]));
  url.searchParams.set("travelmode", "driving");
  if (stops.length > 1)
    url.searchParams.set("waypoints", stops.slice(0, -1).map(point).join("|"));
  return url.href;
}

export function moveDigStop<T>(
  stops: readonly T[],
  index: number,
  direction: -1 | 1,
): T[] {
  const next = [...stops];
  const target = index + direction;
  if (
    !Number.isInteger(index) ||
    index < 0 ||
    index >= next.length ||
    target < 0 ||
    target >= next.length
  )
    return next;
  [next[index], next[target]] = [next[target], next[index]];
  return next;
}

/** Only compare successful concentric expansions, never a different city. */
export function newShopsInExpandedArea(
  previous: NearbyShops | null,
  next: NearbyShops,
): number | null {
  if (
    !previous ||
    next.radiusKm <= previous.radiusKm ||
    next.center.latitude !== previous.center.latitude ||
    next.center.longitude !== previous.center.longitude
  )
    return null;
  const prior = new Set(previous.shops.map((shop) => shop.id));
  return next.shops.filter((shop) => !prior.has(shop.id)).length;
}
