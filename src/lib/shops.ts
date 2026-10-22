export const SHOP_RADII_KM = [10, 25, 50] as const;
export type ShopRadiusKm = (typeof SHOP_RADII_KM)[number];

export interface ShopArea {
  latitude: number;
  longitude: number;
}

export interface ShopPlace extends ShopArea {
  id: string;
  label: string;
}

export interface RecordShop extends ShopArea {
  id: string;
  name: string;
  distanceKm: number;
  address: string | null;
  website: string | null;
  phone: string | null;
  openingHours: string | null;
  osmUrl: string;
}

export interface NearbyShops {
  center: ShopArea;
  radiusKm: ShopRadiusKm;
  shops: RecordShop[];
  fetchedAt: string;
  limited: boolean;
}

/** Approximate area only: precise device coordinates never leave the browser. */
export function approximateShopArea(area: ShopArea): ShopArea {
  return {
    latitude: Math.round(area.latitude * 100) / 100,
    longitude: Math.round(area.longitude * 100) / 100,
  };
}
