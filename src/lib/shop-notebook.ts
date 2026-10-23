export const SHOP_AVATARS = [
  { id: "groove", label: "Groove" },
  { id: "sunshine", label: "Sunshine" },
  { id: "midnight", label: "Midnight" },
  { id: "cherry", label: "Cherry" },
  { id: "mint", label: "Mint" },
  { id: "cosmic", label: "Cosmic" },
] as const;

export type ShopAvatarId = (typeof SHOP_AVATARS)[number]["id"];

export interface ShopSnapshot {
  id: string;
  name: string;
  latitude: number;
  longitude: number;
  address: string | null;
  website: string | null;
}

export interface ShopEmployee {
  id: string;
  name: string;
  role: string;
  avatar: ShopAvatarId;
}

export interface SavedShop {
  shop: ShopSnapshot;
  rating: number | null;
  employees: ShopEmployee[];
  version: number;
  updatedAt: string;
}

export interface SaveShopInput {
  shop: ShopSnapshot;
  rating: number | null;
  employees: ShopEmployee[];
  version: number | null;
}

export interface ShopNotebook {
  shops: SavedShop[];
}
