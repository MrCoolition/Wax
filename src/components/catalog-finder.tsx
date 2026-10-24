"use client";

import { useEffect, useId, useRef, useState } from "react";
import * as Dialog from "@radix-ui/react-dialog";
import {
  ArrowLeft,
  ArrowRight,
  Disc3,
  LoaderCircle,
  Search,
  SlidersHorizontal,
  X,
} from "lucide-react";
import type { CatalogCandidate } from "@/lib/contracts";
import type {
  AlbumSearchResult,
  CatalogAlbum,
  CatalogPage,
  PressingSearchResult,
} from "@/lib/catalog-search";
import { api } from "./library-provider";
import { Button, Cover, ErrorNotice, Field, SourceCredit } from "./ui";
import styles from "./catalog-finder.module.css";

export interface CatalogSearchDraft {
  mode: "album" | "details" | "catalogNumber" | "barcode";
  query?: string;
  artist?: string;
  title?: string;
  catalogNumber?: string;
  barcode?: string;
}
export interface CatalogFinderProps {
  initialSearch?: CatalogSearchDraft;
  initialAlbums?: AlbumSearchResult;
  initialCandidates?: CatalogCandidate[];
  initialAlbum?: CatalogAlbum;
  autoSearch?: boolean;
  active?: boolean;
  onSelect: (candidate: CatalogCandidate) => void;
  onManual?: () => void;
}
type Filters = { country: string; label: string; year: string };
type AlbumView = {
  kind: "albums";
  result: AlbumSearchResult;
  search: CatalogSearchDraft;
};
type PressingView = {
  kind: "pressings";
  result: PressingSearchResult;
  search: CatalogSearchDraft;
  album: CatalogAlbum | null;
  filters: Filters;
  back: AlbumView | null;
};
type View = AlbumView | PressingView;
type Task =
  | { kind: "albums"; search: CatalogSearchDraft; page: number }
  | {
      kind: "pressings";
      search: CatalogSearchDraft;
      album: CatalogAlbum | null;
      filters: Filters;
      page: number;
      back: AlbumView | null;
    }
  | { kind: "release"; candidate: CatalogCandidate };
const EMPTY_FILTERS: Filters = { country: "", label: "", year: "" };
const EMPTY_SEARCH: CatalogSearchDraft = { mode: "album" };
const COUNTRIES = [
  "US",
  "UK",
  "Canada",
  "Germany",
  "France",
  "Japan",
  "Australia",
  "Netherlands",
  "Italy",
  "Spain",
  "Brazil",
  "Europe",
  "Worldwide",
];
function searchParams(search: CatalogSearchDraft) {
  const params = new URLSearchParams();
  if (search.mode === "album" && search.query?.trim())
    params.set("query", search.query.trim());
  if (search.mode === "details") {
    if (search.artist?.trim()) params.set("artist", search.artist.trim());
    if (search.title?.trim()) params.set("title", search.title.trim());
  }
  if (search.mode === "catalogNumber" && search.catalogNumber?.trim())
    params.set("catalogNumber", search.catalogNumber.trim());
  if (search.mode === "barcode" && search.barcode?.trim())
    params.set("barcode", search.barcode.trim());
  return params;
}
function fallbackPage(count: number): CatalogPage {
  return { page: 1, pages: 1, total: count, hasMore: false };
}

export function CatalogFinder({
  initialSearch = EMPTY_SEARCH,
  initialAlbums,
  initialCandidates,
  initialAlbum,
  autoSearch = false,
  active = true,
  onSelect,
  onManual,
}: CatalogFinderProps) {
  const [search, setSearch] = useState(initialSearch);
  const [filters, setFilters] = useState<Filters>(EMPTY_FILTERS);
  const [otherCountry, setOtherCountry] = useState(false);
  const [view, setView] = useState<View | null>(() =>
    initialAlbums
      ? { kind: "albums", result: initialAlbums, search: initialSearch }
      : initialCandidates
        ? {
            kind: "pressings",
            result: {
              candidates: initialCandidates,
              pagination: fallbackPage(initialCandidates.length),
            },
            search: initialSearch,
            album: null,
            filters: EMPTY_FILTERS,
            back: null,
          }
        : null,
  );
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [failed, setFailed] = useState<Task | null>(null);
  const request = useRef<{
    sequence: number;
    controller: AbortController | null;
  }>({ sequence: 0, controller: null });
  const resultHeading = useRef<HTMLHeadingElement>(null);
  const mounted = useRef(false);
  const onSelectRef = useRef(onSelect);
  onSelectRef.current = onSelect;
  const headingId = useId();

  function cancel() {
    request.current.controller?.abort();
    request.current.sequence += 1;
    setBusy("");
    setError("");
    setFailed(null);
  }
  async function run(task: Task, focus = true) {
    request.current.controller?.abort();
    const controller = new AbortController();
    const sequence = ++request.current.sequence;
    request.current.controller = controller;
    const timeout = window.setTimeout(() => controller.abort("timeout"), 25000);
    setBusy(
      task.kind === "release"
        ? `release-${task.candidate.discogsId}`
        : task.kind,
    );
    setError("");
    setFailed(null);
    try {
      if (task.kind === "release") {
        const full = task.candidate.discogsId
          ? await api<CatalogCandidate>(
              `/api/catalog/${task.candidate.discogsId}`,
              { signal: controller.signal },
            )
          : task.candidate;
        if (sequence !== request.current.sequence) return;
        onSelectRef.current(full);
      } else {
        const params =
          task.kind === "pressings" && task.album
            ? new URLSearchParams()
            : searchParams(task.search);
        params.set("page", String(task.page));
        if (task.kind === "albums") {
          const result = await api<AlbumSearchResult>(
            `/api/catalog/albums?${params}`,
            { signal: controller.signal },
          );
          if (sequence !== request.current.sequence) return;
          setView({ kind: "albums", result, search: task.search });
        } else {
          for (const [key, value] of Object.entries(task.filters))
            if (value.trim()) params.set(key, value.trim());
          const path = task.album
            ? `/api/catalog/albums/${task.album.masterId}/versions`
            : "/api/catalog";
          const result = await api<PressingSearchResult>(`${path}?${params}`, {
            signal: controller.signal,
          });
          if (sequence !== request.current.sequence) return;
          setView({
            ...task,
            result: {
              ...result,
              pagination:
                result.pagination || fallbackPage(result.candidates.length),
            },
          });
          setFilters(task.filters);
        }
        if (focus)
          requestAnimationFrame(() =>
            resultHeading.current?.focus({ preventScroll: true }),
          );
      }
    } catch (err) {
      if (
        sequence !== request.current.sequence ||
        (controller.signal.aborted && controller.signal.reason !== "timeout")
      )
        return;
      setError(
        controller.signal.reason === "timeout"
          ? "That search took too long. Your place is saved; try again."
          : err instanceof Error
            ? err.message
            : "The catalog is taking a break. Try again.",
      );
      setFailed(task);
    } finally {
      window.clearTimeout(timeout);
      if (sequence === request.current.sequence) setBusy("");
    }
  }
  useEffect(() => {
    if (!mounted.current) {
      mounted.current = true;
      if (initialAlbum)
        void run(
          {
            kind: "pressings",
            album: initialAlbum,
            search: initialSearch,
            filters: EMPTY_FILTERS,
            page: 1,
            back: initialAlbums
              ? { kind: "albums", result: initialAlbums, search: initialSearch }
              : null,
          },
          false,
        );
      else if (autoSearch && searchParams(initialSearch).size)
        void run(
          {
            kind:
              initialSearch.mode === "album" || initialSearch.mode === "details"
                ? "albums"
                : "pressings",
            search: initialSearch,
            page: 1,
            album: null,
            filters: EMPTY_FILTERS,
            back: null,
          },
          false,
        );
    }
    return () => {
      request.current.sequence += 1;
      request.current.controller?.abort();
      mounted.current = false;
    };
    // Initial state is intentionally scoped to this mounted finder; callers use a key for new clues.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  useEffect(() => {
    if (!active) cancel();
    // A hidden capture tab retains results, but pending work must never change the active draft.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active]);

  function updateSearch(patch: Partial<CatalogSearchDraft>) {
    cancel();
    setSearch((current) => ({ ...current, ...patch }));
  }
  function submitSearch() {
    if (!searchParams(search).size) {
      setError("Enter an artist, title, catalog number, or barcode to start.");
      return;
    }
    setFilters(EMPTY_FILTERS);
    setOtherCountry(false);
    void run(
      search.mode === "album" || search.mode === "details"
        ? { kind: "albums", search, page: 1 }
        : {
            kind: "pressings",
            search,
            album: null,
            filters: EMPTY_FILTERS,
            page: 1,
            back: null,
          },
    );
  }
  function chooseAlbum(album: CatalogAlbum) {
    setFilters(EMPTY_FILTERS);
    setOtherCountry(false);
    void run({
      kind: "pressings",
      album,
      search: view?.search || search,
      filters: EMPTY_FILTERS,
      page: 1,
      back: view?.kind === "albums" ? view : null,
    });
  }
  function loadPage(page: number) {
    if (!view) return;
    void run(
      view.kind === "albums"
        ? { kind: "albums", search: view.search, page }
        : {
            kind: "pressings",
            search: view.search,
            album: view.album,
            filters: view.filters,
            back: view.back,
            page,
          },
    );
  }
  const staleSearch =
    view &&
    searchParams(view.search).toString() !== searchParams(search).toString();
  const pagination = view?.result.pagination;
  return (
    <div className={styles.finder}>
      <form
        className={styles.searchForm}
        onSubmit={(event) => {
          event.preventDefault();
          event.stopPropagation();
          submitSearch();
        }}
      >
        <div
          className={styles.searchModes}
          role="group"
          aria-label="Catalog search method"
        >
          {(
            [
              { mode: "album", label: "Artist + album" },
              { mode: "details", label: "Separate artist & title" },
              { mode: "catalogNumber", label: "Catalog number" },
              { mode: "barcode", label: "Barcode" },
            ] as const
          ).map((item) => (
            <button
              type="button"
              aria-pressed={search.mode === item.mode}
              key={item.mode}
              onClick={() => updateSearch({ mode: item.mode })}
            >
              {item.label}
            </button>
          ))}
        </div>
        <div className={styles.searchLine}>
          {search.mode === "details" ? (
            <div className={styles.structured}>
              <Field label="Search artist">
                <input
                  value={search.artist || ""}
                  maxLength={200}
                  onChange={(event) =>
                    updateSearch({ artist: event.target.value })
                  }
                  placeholder="The Beatles"
                />
              </Field>
              <Field label="Search album title">
                <input
                  value={search.title || ""}
                  maxLength={200}
                  onChange={(event) =>
                    updateSearch({ title: event.target.value })
                  }
                  placeholder="White Album"
                />
              </Field>
            </div>
          ) : (
            <Field
              label={
                search.mode === "album"
                  ? "Search the record catalog"
                  : search.mode === "barcode"
                    ? "Search barcode"
                    : "Search catalog number"
              }
            >
              <input
                value={
                  search.mode === "album"
                    ? search.query || ""
                    : search.mode === "barcode"
                      ? search.barcode || ""
                      : search.catalogNumber || ""
                }
                required
                maxLength={search.mode === "barcode" ? 40 : 200}
                inputMode={search.mode === "barcode" ? "numeric" : undefined}
                onChange={(event) =>
                  updateSearch({
                    [search.mode === "album" ? "query" : search.mode]:
                      event.target.value,
                  })
                }
                placeholder={
                  search.mode === "album"
                    ? "Try Beatles White Album…"
                    : search.mode === "barcode"
                      ? "Keep any leading zeros"
                      : "The number on the spine or label"
                }
              />
            </Field>
          )}
          <Button type="submit" busy={busy === "albums"}>
            <Search size={17} /> Search
          </Button>
        </div>
        <p className={styles.searchHint}>
          {search.mode === "catalogNumber" || search.mode === "barcode"
            ? "Go straight to pressings. Compare the details before choosing."
            : "Start with the album. Then find the country, label, and year on your copy."}
        </p>
      </form>
      {view?.kind === "pressings" && (
        <div className={styles.pressingHeader}>
          {view.back && (
            <button
              type="button"
              className="text-button"
              onClick={() => {
                cancel();
                setView(view.back);
                setSearch(view.back!.search);
              }}
            >
              <ArrowLeft size={15} /> Back to albums
            </button>
          )}
          {view.album && (
            <div className={styles.selectedAlbum}>
              <Cover record={view.album} />
              <div>
                <span className="eyebrow">
                  THE ALBUM / NOW FIND YOUR EDITION
                </span>
                <h3>{view.album.title}</h3>
                <p>
                  {view.album.artist} ·{" "}
                  {view.album.originalYear || "Original year unknown"}
                </p>
                <SourceCredit sourceUrl={view.album.sourceUrl} />
              </div>
            </div>
          )}
          <form
            className={styles.filterForm}
            onSubmit={(event) => {
              event.preventDefault();
              event.stopPropagation();
              void run({
                kind: "pressings",
                search: view.search,
                album: view.album,
                filters,
                page: 1,
                back: view.back,
              });
            }}
          >
            <div className={styles.filterIntro}>
              <SlidersHorizontal size={16} />
              <strong>Narrow the pressings</strong>
              <span>Vinyl · all countries welcome</span>
            </div>
            <div className={styles.filterFields}>
              <Field label="Pressing country">
                <select
                  value={otherCountry ? "other" : filters.country}
                  onChange={(event) => {
                    cancel();
                    const value = event.target.value;
                    setOtherCountry(value === "other");
                    setFilters((current) => ({
                      ...current,
                      country: value === "other" ? "" : value,
                    }));
                  }}
                >
                  <option value="">All countries</option>
                  {COUNTRIES.map((country) => (
                    <option key={country} value={country}>
                      {country}
                    </option>
                  ))}
                  <option value="other">Another country…</option>
                </select>
              </Field>
              {otherCountry && (
                <Field label="Other pressing country">
                  <input
                    value={filters.country}
                    maxLength={100}
                    placeholder="As written in the catalog"
                    onChange={(event) => {
                      cancel();
                      setFilters((current) => ({
                        ...current,
                        country: event.target.value,
                      }));
                    }}
                  />
                </Field>
              )}
              <Field label="Label / company">
                <input
                  value={filters.label}
                  maxLength={200}
                  placeholder="Any label"
                  onChange={(event) => {
                    cancel();
                    setFilters((current) => ({
                      ...current,
                      label: event.target.value,
                    }));
                  }}
                />
              </Field>
              <Field label="Pressing year">
                <input
                  value={filters.year}
                  type="number"
                  min={1877}
                  max={new Date().getFullYear() + 2}
                  placeholder="Any year"
                  onChange={(event) => {
                    cancel();
                    setFilters((current) => ({
                      ...current,
                      year: event.target.value,
                    }));
                  }}
                />
              </Field>
              <Button
                type="submit"
                variant="secondary"
                busy={busy === "pressings"}
              >
                Apply filters
              </Button>
            </div>
            <p className={styles.searchHint}>
              Label matches can include any credited label or company. Each card
              shows the primary label.
            </p>
            {(filters.country ||
              filters.label ||
              filters.year ||
              view.filters.country ||
              view.filters.label ||
              view.filters.year) && (
              <button
                className="text-button"
                type="button"
                onClick={() => {
                  setOtherCountry(false);
                  setFilters(EMPTY_FILTERS);
                  void run({
                    kind: "pressings",
                    search: view.search,
                    album: view.album,
                    filters: EMPTY_FILTERS,
                    page: 1,
                    back: view.back,
                  });
                }}
              >
                Clear pressing filters
              </button>
            )}
          </form>
        </div>
      )}
      {busy && (
        <p className={styles.loading} role="status">
          <LoaderCircle className="spin" size={17} />
          {busy.startsWith("release")
            ? "Pulling the pressing details…"
            : busy === "albums"
              ? "Digging through the albums…"
              : "Finding vinyl pressings…"}
        </p>
      )}
      <ErrorNotice>
        {error && (
          <>
            <span>{error}</span>
            {failed && (
              <Button
                type="button"
                variant="secondary"
                onClick={() => void run(failed)}
              >
                Try catalog again
              </Button>
            )}
          </>
        )}
      </ErrorNotice>
      {view && (
        <section className={styles.results} aria-labelledby={headingId}>
          <div className={styles.resultHeading}>
            <div>
              <span className="eyebrow">
                {view.kind === "albums"
                  ? "FIRST, FIND THE ALBUM"
                  : "NEXT, CHECK THE PRESSING"}
              </span>
              <h3 id={headingId} ref={resultHeading} tabIndex={-1}>
                {view.kind === "albums"
                  ? "Albums to dig into."
                  : "Find your pressing."}
              </h3>
            </div>
            {onManual && (
              <Button
                type="button"
                variant="quiet"
                onClick={() => {
                  cancel();
                  onManual();
                }}
              >
                Add by hand
              </Button>
            )}
          </div>
          <p className={styles.resultCount}>
            {pagination?.total.toLocaleString()}{" "}
            {view.kind === "albums" ? "album matches" : "catalog results"}
            {pagination && pagination.pages > 1
              ? ` · Page ${pagination.page} of ${pagination.pages}`
              : ""}
          </p>
          {view.result.notice && (
            <p className={styles.notice}>{view.result.notice}</p>
          )}
          {staleSearch && (
            <p className={styles.notice}>
              Showing your last search. Press Search to use your edited clues.
            </p>
          )}
          {view.kind === "pressings" &&
            JSON.stringify(filters) !== JSON.stringify(view.filters) && (
              <p className={styles.notice}>
                Apply filters to update these results.
              </p>
            )}
          {view.kind === "albums" ? (
            <div className={styles.albumGrid}>
              {view.result.albums.map((album) => (
                <article key={album.masterId} className={styles.albumCard}>
                  <Cover record={album} />
                  <div>
                    <span className={styles.year}>
                      {album.originalYear || "YEAR UNKNOWN"}
                    </span>
                    <h4>{album.title}</h4>
                    <p>{album.artist}</p>
                    <Button
                      type="button"
                      variant="secondary"
                      onClick={() => chooseAlbum(album)}
                      disabled={Boolean(busy)}
                    >
                      See pressings <ArrowRight size={14} />
                    </Button>
                    <SourceCredit sourceUrl={album.sourceUrl} />
                  </div>
                </article>
              ))}
            </div>
          ) : (
            <div className={styles.pressingGrid}>
              {view.result.candidates.map((item, index) => (
                <article
                  key={item.discogsId || index}
                  className={styles.pressingCard}
                >
                  <Cover record={item} />
                  <div className={styles.pressingBody}>
                    <h4>{item.title}</h4>
                    <p>{item.artist}</p>
                    <dl>
                      <div>
                        <dt>Country</dt>
                        <dd>{item.country || "Unknown"}</dd>
                      </div>
                      <div>
                        <dt>Pressing year</dt>
                        <dd>{item.year || "Unknown"}</dd>
                      </div>
                      <div>
                        <dt>Label</dt>
                        <dd>{item.label || "Unknown"}</dd>
                      </div>
                      <div>
                        <dt>Catalog no.</dt>
                        <dd>{item.catalogNumber || "Unknown"}</dd>
                      </div>
                    </dl>
                    <p className={styles.format}>
                      {item.format || "Format not stated"}
                    </p>
                    <Button
                      type="button"
                      variant="secondary"
                      busy={busy === `release-${item.discogsId}`}
                      disabled={Boolean(busy)}
                      onClick={() =>
                        void run({ kind: "release", candidate: item })
                      }
                    >
                      Review this pressing <ArrowRight size={14} />
                    </Button>
                    <SourceCredit sourceUrl={item.sourceUrl} />
                  </div>
                </article>
              ))}
            </div>
          )}
          {(view.kind === "albums"
            ? view.result.albums.length === 0
            : view.result.candidates.length === 0) && (
            <div className={styles.empty}>
              <Disc3 size={30} />
              <h4>No clear match on this page.</h4>
              <p>
                Try fewer words, clear a filter, or use the catalog number on
                your label. Manual details are always welcome.
              </p>
              {view.kind === "albums" && (
                <Button
                  type="button"
                  variant="secondary"
                  onClick={() =>
                    void run({
                      kind: "pressings",
                      search: view.search,
                      album: null,
                      filters: EMPTY_FILTERS,
                      page: 1,
                      back: view,
                    })
                  }
                >
                  Search individual pressings
                </Button>
              )}
            </div>
          )}
          {pagination && pagination.pages > 1 && (
            <nav className={styles.pagination} aria-label="Catalog pages">
              <Button
                type="button"
                variant="secondary"
                disabled={pagination.page <= 1 || Boolean(busy)}
                onClick={() => loadPage(pagination.page - 1)}
              >
                <ArrowLeft size={15} /> Previous results
              </Button>
              <span>
                Page {pagination.page} of {pagination.pages}
              </span>
              <Button
                type="button"
                variant="secondary"
                disabled={
                  !pagination.hasMore || pagination.page >= 100 || Boolean(busy)
                }
                onClick={() => loadPage(pagination.page + 1)}
              >
                Next results <ArrowRight size={15} />
              </Button>
            </nav>
          )}
          {pagination && pagination.pages > 100 && (
            <p className={styles.searchHint}>
              The catalog shows up to 100 pages. Narrow your search to find a
              particular edition.
            </p>
          )}
        </section>
      )}
      <p className={styles.footer}>
        <Disc3 size={16} /> A match is a starting point. You decide which
        pressing is yours.
      </p>
    </div>
  );
}

export function CatalogFinderDialog({
  open,
  onOpenChange,
  ...props
}: CatalogFinderProps & {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="modal-overlay" />
        <Dialog.Content className={styles.dialog}>
          <div className="modal-top">
            <Dialog.Title>Find catalog matches</Dialog.Title>
            <Dialog.Close
              className="icon-button"
              aria-label="Close catalog search"
            >
              <X size={20} />
            </Dialog.Close>
          </div>
          <Dialog.Description className="modal-description">
            Choose an album, compare its pressings, then review the details.
            Your copy notes stay with you.
          </Dialog.Description>
          <CatalogFinder {...props} />
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

export function ManualCatalogSuggestions({
  artist,
  title,
  onBrowse,
  onAlbum,
}: {
  artist: string;
  title: string;
  onBrowse: (result?: AlbumSearchResult) => void;
  onAlbum: (album: CatalogAlbum, result: AlbumSearchResult) => void;
}) {
  const normalizedArtist = artist.trim().toLowerCase();
  const normalizedTitle = title.trim().toLowerCase();
  const key = JSON.stringify([normalizedArtist, normalizedTitle]);
  const [state, setState] = useState<{
    key: string;
    loading: boolean;
    result?: AlbumSearchResult;
    error?: string;
  }>({ key: "", loading: false });
  const [retry, setRetry] = useState(0);
  const cache = useRef(
    new Map<string, { result?: AlbumSearchResult; error?: string }>(),
  );
  useEffect(() => {
    if (!normalizedArtist || !normalizedTitle) return;
    const cached = cache.current.get(key);
    if (cached) {
      setState({ key, loading: false, ...cached });
      return;
    }
    const controller = new AbortController();
    setState({ key, loading: true });
    let timeout: number | undefined;
    const timer = window.setTimeout(async () => {
      timeout = window.setTimeout(() => controller.abort("timeout"), 20000);
      try {
        const params = new URLSearchParams({
          artist: normalizedArtist,
          title: normalizedTitle,
        });
        const result = await api<AlbumSearchResult>(
          `/api/catalog/albums?${params}`,
          { signal: controller.signal },
        );
        if (controller.signal.aborted) return;
        cache.current.set(key, { result });
        setState({ key, loading: false, result });
      } catch (err) {
        if (controller.signal.aborted && controller.signal.reason !== "timeout")
          return;
        const error =
          err instanceof Error && controller.signal.reason !== "timeout"
            ? err.message
            : "The catalog is taking a little too long.";
        cache.current.set(key, { error });
        setState({ key, loading: false, error });
      } finally {
        window.clearTimeout(timeout);
      }
    }, 750);
    return () => {
      window.clearTimeout(timer);
      window.clearTimeout(timeout);
      controller.abort();
    };
  }, [normalizedArtist, normalizedTitle, key, retry]);
  const current = state.key === key ? state : null;
  return (
    <section className={styles.suggestions} aria-label="Catalog suggestions">
      <div className={styles.suggestionIntro}>
        <div>
          <span className="eyebrow">A LITTLE HELP FROM THE CATALOG</span>
          <h3>Let’s find the finer details.</h3>
          <p>
            We’ll look for the album as you enter its artist and title. Nothing
            replaces your details until you choose a pressing.
          </p>
        </div>
        <Button
          type="button"
          variant="secondary"
          onClick={() => onBrowse(current?.result)}
        >
          <Search size={16} /> Find catalog matches
        </Button>
      </div>
      {normalizedArtist && normalizedTitle && (
        <>
          <div aria-live="polite">
            {current?.loading && (
              <p className={styles.loading}>
                <LoaderCircle className="spin" size={16} /> Searching the
                catalog…
              </p>
            )}
            {current?.error && (
              <p className={styles.notice}>
                {current.error} You can keep entering your record.{" "}
                <button
                  className="text-button"
                  type="button"
                  onClick={() => {
                    cache.current.delete(key);
                    setRetry((value) => value + 1);
                  }}
                >
                  Retry suggestions
                </button>
              </p>
            )}
            {current?.result && (
              <p className={styles.searchHint}>
                {current.result.albums.length
                  ? "These might be your album. Compare the pressings to find your copy."
                  : "No album match yet. Try Find catalog matches to refine the clues or search a catalog number."}
              </p>
            )}
          </div>
          {current?.result && (
            <>
              <div className={styles.suggestionList}>
                {current.result.albums.slice(0, 3).map((album) => (
                  <button
                    type="button"
                    key={album.masterId}
                    onClick={() => onAlbum(album, current.result!)}
                  >
                    <Cover record={album} />
                    <span>
                      <strong>{album.title}</strong>
                      <small>
                        {album.artist} · {album.originalYear || "Year unknown"}
                      </small>
                      <b>
                        See pressings <ArrowRight size={12} />
                      </b>
                    </span>
                  </button>
                ))}
              </div>
              <SourceCredit sourceUrl="https://www.discogs.com" />
            </>
          )}
        </>
      )}
    </section>
  );
}
