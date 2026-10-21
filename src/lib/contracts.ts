export type Condition =
  "M" | "NM" | "VG+" | "VG" | "G+" | "G" | "F" | "P" | "Not graded";
export type Intent = "keep" | "trade" | "sell";
export interface Collector {
  id: string;
  name: string;
  email: string;
  isOwner: boolean;
}
export interface CatalogCandidate {
  discogsId: number | null;
  artist: string;
  title: string;
  year: number | null;
  genres: string[];
  label: string;
  catalogNumber: string;
  country: string;
  format: string;
  coverUrl: string | null;
  sourceUrl: string | null;
}
export interface CopyFields {
  mediaCondition: Condition;
  sleeveCondition: Condition;
  purchasePrice: string | null;
  currency: string;
  acquiredAt: string | null;
  notes: string;
  rating: number;
  location: string;
  intent: Intent;
  pressingConfirmed: boolean;
}
export interface RecordCopy extends CatalogCandidate, CopyFields {
  id: string;
  releaseId: string;
  version: number;
  addedAt: string;
  updatedAt: string;
  crateIds: string[];
  spinCount: number;
  lastPlayedAt: string | null;
  photoIds: string[];
}
export interface Crate {
  id: string;
  name: string;
  description: string;
  color: string;
  recordIds: string[];
  shares: { userId: string; name: string; email: string }[];
}
export interface SharedRecord extends CatalogCandidate {
  id: string;
  mediaCondition: Condition;
  sleeveCondition: Condition;
  pressingConfirmed: boolean;
}
export interface SharedCrate {
  id: string;
  name: string;
  description: string;
  color: string;
  ownerName: string;
  records: SharedRecord[];
}
export interface ListeningEntry {
  id: string;
  recordId: string;
  artist: string;
  title: string;
  coverUrl: string | null;
  playedAt: string;
  notes: string;
}
export interface LibraryData {
  collector: Collector;
  records: RecordCopy[];
  crates: Crate[];
  sharedCrates: SharedCrate[];
  listens: ListeningEntry[];
}
export interface CreateRecordInput {
  idempotencyKey: string;
  candidate: CatalogCandidate;
  copy: CopyFields;
  allowDuplicate: boolean;
  photoIds?: string[];
  crateIds?: string[];
}
export interface UpdateRecordInput extends Partial<CopyFields> {
  version: number;
  candidate?: CatalogCandidate;
  crateIds?: string[];
}
export interface PhotoIdentification {
  photoId: string;
  clues: {
    artist: string | null;
    title: string | null;
    catalogNumber: string | null;
    barcode: string | null;
    label: string | null;
    uncertainty: string;
  };
  candidates: CatalogCandidate[];
}
export interface WizardChange {
  recordId: string;
  version: number;
  changes: Partial<
    Pick<
      CopyFields,
      | "notes"
      | "rating"
      | "location"
      | "intent"
      | "mediaCondition"
      | "sleeveCondition"
    >
  >;
}
export interface WizardReply {
  reply: string;
  recordIds: string[];
  proposal: { id: string; summary: string; changes: WizardChange[] } | null;
}
export interface ApiFailure {
  error: string;
  code?: string;
  details?: unknown;
}
export const DEFAULT_COPY: CopyFields = {
  mediaCondition: "Not graded",
  sleeveCondition: "Not graded",
  purchasePrice: null,
  currency: "USD",
  acquiredAt: null,
  notes: "",
  rating: 0,
  location: "",
  intent: "keep",
  pressingConfirmed: false,
};
export const EMPTY_CANDIDATE: CatalogCandidate = {
  discogsId: null,
  artist: "",
  title: "",
  year: null,
  genres: [],
  label: "",
  catalogNumber: "",
  country: "",
  format: "Vinyl, LP",
  coverUrl: null,
  sourceUrl: null,
};
export const CONDITIONS: Condition[] = [
  "Not graded",
  "M",
  "NM",
  "VG+",
  "VG",
  "G+",
  "G",
  "F",
  "P",
];
