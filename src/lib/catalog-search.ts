import type { CatalogCandidate } from "./contracts";

/** An album grouping, never an exact pressing or a saveable copy. */
export interface CatalogAlbum {
  masterId: number;
  artist: string;
  title: string;
  originalYear: number | null;
  coverUrl: string | null;
  sourceUrl: string;
}

export interface CatalogPage {
  page: number;
  pages: number;
  total: number;
  hasMore: boolean;
}

export interface AlbumSearchResult {
  albums: CatalogAlbum[];
  pagination: CatalogPage;
  notice?: string;
}

export interface PressingSearchResult {
  candidates: CatalogCandidate[];
  pagination: CatalogPage;
  notice?: string;
}
