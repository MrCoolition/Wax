export type CreditCategory = "performance" | "production" | "writing" | "other";
export interface ExplorerPerson {
  id: number | null;
  name: string;
}
export interface ExplorerCredit {
  key: string;
  artistId: number | null;
  name: string;
  role: string;
  category: CreditCategory;
  /** Original Discogs track reference; null means no specific tracks stated. */
  trackReference: string | null;
}
export interface ExplorerTrack {
  id: string;
  position: string;
  title: string;
  duration: string;
  type: "track" | "heading";
  credits: ExplorerCredit[];
}
export interface ExplorerRelease {
  id: number;
  title: string;
  artists: ExplorerPerson[];
  year: number | null;
  label: string;
  catalogNumber: string;
  country: string;
  format: string;
  coverUrl: string | null;
  sourceUrl: string;
  fetchedAt: string;
  tracks: ExplorerTrack[];
  releaseCredits: ExplorerCredit[];
  unresolvedCredits: ExplorerCredit[];
  /** Provider-reported pressing identifiers, never inferred from artwork. */
  identifiers?: { type: string; value: string; description: string }[];
  companies?: {
    id: number | null;
    name: string;
    role: string;
    catalogNumber: string;
  }[];
  notes?: string;
}
export interface ExplorerRelation {
  id: number;
  name: string;
  active: boolean | null;
}
export interface ExplorerArtist {
  id: number;
  name: string;
  realName: string | null;
  profile: string;
  imageUrl: string | null;
  sourceUrl: string;
  groups: ExplorerRelation[];
  members: ExplorerRelation[];
  aliases: ExplorerRelation[];
  fetchedAt: string;
}
export interface ExplorerAlbum {
  id: number;
  type: "release" | "master";
  /** Masters must be explored through this release ID, never the master ID. */
  mainReleaseId: number | null;
  title: string;
  artist: string;
  year: number | null;
  thumbUrl: string | null;
  role: string;
  sourceUrl: string;
}
export interface ExplorerDiscography {
  artistId: number;
  page: number;
  pages: number;
  total: number;
  releases: ExplorerAlbum[];
  fetchedAt: string;
}
export interface ExplorerStart {
  copyId: string;
  artist: string;
  title: string;
  discogsId: number | null;
  release: ExplorerRelease | null;
}
