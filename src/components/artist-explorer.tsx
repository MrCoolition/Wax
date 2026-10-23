"use client";

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type FormEvent,
} from "react";
import Link from "next/link";
import {
  ArrowDown,
  ArrowLeft,
  ArrowRight,
  ArrowUpRight,
  Check,
  ChevronLeft,
  ChevronRight,
  Disc3,
  GitBranch,
  LoaderCircle,
  Mic2,
  Music2,
  PenLine,
  RefreshCw,
  Search,
  Sparkles,
  SlidersHorizontal,
  Users,
  X,
} from "lucide-react";
import type { CatalogCandidate } from "@/lib/contracts";
import type {
  CreditCategory,
  ExplorerAlbum,
  ExplorerArtist,
  ExplorerCredit,
  ExplorerDiscography,
  ExplorerPerson,
  ExplorerRelation,
  ExplorerRelease,
  ExplorerStart,
} from "@/lib/explorer";
import { api, useLibrary } from "./library-provider";
import { Cover } from "./ui";
import styles from "./artist-explorer.module.css";

interface Frame {
  release: ExplorerRelease;
  trackId: string | null;
  artist: ExplorerArtist | null;
  discography: ExplorerDiscography | null;
}
interface TrailItem {
  label: string;
  frame: Frame;
}
type NavigationRequest =
  | { kind: "artist"; id: number; name: string }
  | { kind: "release"; id: number; name: string };
const categories: { id: CreditCategory; label: string }[] = [
  { id: "performance", label: "The players" },
  { id: "production", label: "Behind the sound" },
  { id: "writing", label: "Words & arrangements" },
  { id: "other", label: "The rest of the story" },
];
const requestSignal = (controller: AbortController) =>
  AbortSignal.any([controller.signal, AbortSignal.timeout(25_000)]);
const messageFor = (error: unknown) =>
  error instanceof DOMException && error.name === "TimeoutError"
    ? "The catalog took too long to respond. Your place is safe — give it another try."
    : error instanceof Error
      ? error.message
      : "The catalog isn’t available right now. Please try again.";

function newFrame(release: ExplorerRelease): Frame {
  return {
    release,
    trackId: release.tracks.find((track) => track.type === "track")?.id ?? null,
    artist: null,
    discography: null,
  };
}

function ArtistLink({
  person,
  onSelect,
  className = "",
}: {
  person: ExplorerPerson;
  onSelect: (person: ExplorerPerson) => void;
  className?: string;
}) {
  return person.id ? (
    <button className={className} onClick={() => onSelect(person)}>
      {person.name}
      <ArrowUpRight size={13} />
    </button>
  ) : (
    <span className={className}>{person.name}</span>
  );
}

function CreditGroups({
  credits,
  onSelect,
  unresolved = false,
}: {
  credits: ExplorerCredit[];
  onSelect: (person: ExplorerPerson) => void;
  unresolved?: boolean;
}) {
  return (
    <div className={styles.creditGroups}>
      {categories.map((category) => {
        const group = credits.filter(
          (credit) => credit.category === category.id,
        );
        return group.length ? (
          <div className={styles.creditGroup} key={category.id}>
            <h4>{category.label}</h4>
            <ul>
              {group.map((credit) => (
                <li key={credit.key}>
                  {credit.artistId ? (
                    <button
                      onClick={() =>
                        onSelect({ id: credit.artistId, name: credit.name })
                      }
                    >
                      <span className={styles.creditMark} aria-hidden="true">
                        {credit.category === "performance" ? (
                          <Music2 size={14} />
                        ) : credit.category === "production" ? (
                          <SlidersHorizontal size={14} />
                        ) : credit.category === "writing" ? (
                          <PenLine size={14} />
                        ) : (
                          <Disc3 size={14} />
                        )}
                      </span>
                      <span className={styles.creditIdentity}>
                        <strong>{credit.name}</strong>
                        <small>{credit.role}</small>
                        {unresolved && credit.trackReference && (
                          <em>Listed tracks: {credit.trackReference}</em>
                        )}
                      </span>
                      <ArrowUpRight size={14} />
                    </button>
                  ) : (
                    <div className={styles.unlinkedCredit}>
                      <strong>{credit.name}</strong>
                      <small>{credit.role}</small>
                      {unresolved && credit.trackReference && (
                        <em>Listed tracks: {credit.trackReference}</em>
                      )}
                      <span className={styles.unlinkedLabel}>
                        No linked artist profile
                      </span>
                    </div>
                  )}
                </li>
              ))}
            </ul>
          </div>
        ) : null;
      })}
    </div>
  );
}

function RelationBranch({
  title,
  relations,
  onSelect,
  className,
}: {
  title: string;
  relations: ExplorerRelation[];
  onSelect: (person: ExplorerPerson) => void;
  className: string;
}) {
  return (
    <div className={`${styles.branch} ${className}`}>
      <h4>
        {title}
        <span>{relations.length}</span>
      </h4>
      {relations.length ? (
        <ul>
          {relations.map((person) => (
            <li key={person.id}>
              <button onClick={() => onSelect(person)}>
                <span>
                  {person.name}
                  {person.active === false && title !== "Also known as" && (
                    <small>
                      {title === "Members" ? "Former member" : "Former group"}
                    </small>
                  )}
                </span>
                <ArrowUpRight size={13} />
              </button>
            </li>
          ))}
        </ul>
      ) : (
        <p>No {title.toLowerCase()} listed.</p>
      )}
    </div>
  );
}

function AlbumCard({
  album,
  onExplore,
  ownedCopyId,
}: {
  album: ExplorerAlbum;
  onExplore: (id: number, title: string) => void;
  ownedCopyId?: string;
}) {
  const releaseId = album.type === "release" ? album.id : album.mainReleaseId;
  const artwork = (
    <Cover
      record={{
        coverUrl: album.thumbUrl,
        title: album.title,
        artist: album.artist,
      }}
      className={styles.albumCover}
    />
  );
  return (
    <li className={styles.albumCard}>
      {releaseId ? (
        <button
          className={styles.albumArtwork}
          onClick={() => onExplore(releaseId, album.title)}
          aria-label={`Explore ${album.title}`}
        >
          {artwork}
          <span>
            <GitBranch size={15} /> Follow this record <ArrowRight size={15} />
          </span>
        </button>
      ) : (
        <div className={styles.albumArtwork}>{artwork}</div>
      )}
      <div className={styles.albumMeta}>
        <span>
          {album.year || "YEAR UNKNOWN"}
          {album.role && <> · {album.role}</>}
        </span>
        <h4>{album.title}</h4>
        <p>{album.artist}</p>
        {ownedCopyId && (
          <Link
            className={styles.ownedAlbum}
            href={`/records/${encodeURIComponent(ownedCopyId)}`}
          >
            <Check size={11} /> In your collection <ArrowUpRight size={11} />
          </Link>
        )}
        {!releaseId && (
          <small>A linked release isn’t available to explore.</small>
        )}
        <a href={album.sourceUrl} target="_blank" rel="noopener noreferrer">
          Discogs <ArrowUpRight size={11} />
        </a>
      </div>
    </li>
  );
}

function ArtistTree({
  artist,
  discography,
  onPerson,
  onAlbum,
  onDiscography,
  loadingDiscography,
  discographyError,
  discographyRetryPage,
  onClose,
  ownedCopies,
}: {
  artist: ExplorerArtist;
  discography: ExplorerDiscography | null;
  onPerson: (person: ExplorerPerson) => void;
  onAlbum: (id: number, title: string) => void;
  onDiscography: (page: number) => void;
  loadingDiscography: boolean;
  discographyError: string;
  discographyRetryPage: number;
  onClose: () => void;
  ownedCopies: Map<number, string>;
}) {
  const [failedImage, setFailedImage] = useState("");
  const heading = useRef<HTMLHeadingElement>(null);
  useEffect(() => {
    heading.current?.scrollIntoView({ behavior: "instant", block: "start" });
    heading.current?.focus({ preventScroll: true });
  }, [artist.id]);
  return (
    <section
      className={styles.artistSection}
      aria-labelledby="artist-tree-heading"
    >
      <div className={styles.sectionBar}>
        <div>
          <span className={styles.kicker}>EVERY RECORD HAS ROOTS</span>
          <h2 id="artist-tree-heading" ref={heading} tabIndex={-1}>
            Artist Tree<span>.</span>
          </h2>
        </div>
        <button
          className={styles.closeTree}
          onClick={onClose}
          aria-label="Close artist tree"
        >
          <X size={18} />
        </button>
      </div>
      <div className={styles.tree}>
        <svg
          className={styles.treeLines}
          viewBox="0 0 800 380"
          preserveAspectRatio="none"
          aria-hidden="true"
        >
          <path d="M400 90V190M400 190H130V240M400 190H670V240M400 190V300" />
          <circle cx="400" cy="190" r="5" />
        </svg>
        <div className={styles.artistNode}>
          <div className={styles.artistPortrait}>
            {artist.imageUrl && failedImage !== artist.imageUrl ? (
              <img
                src={artist.imageUrl}
                alt={artist.name}
                onError={() => setFailedImage(artist.imageUrl!)}
              />
            ) : (
              <Disc3 size={32} strokeWidth={1.2} />
            )}
          </div>
          <span className={styles.kicker}>FOLLOW THE CONNECTION</span>
          <h3>{artist.name}</h3>
          {artist.realName && artist.realName !== artist.name && (
            <p>{artist.realName}</p>
          )}
        </div>
        <div className={styles.branches}>
          <RelationBranch
            title="Groups"
            relations={artist.groups}
            onSelect={onPerson}
            className={styles.groupsBranch}
          />
          <RelationBranch
            title="Members"
            relations={artist.members}
            onSelect={onPerson}
            className={styles.membersBranch}
          />
          <RelationBranch
            title="Also known as"
            relations={artist.aliases}
            onSelect={onPerson}
            className={styles.aliasesBranch}
          />
        </div>
      </div>
      {artist.profile && (
        <details className={styles.artistProfile}>
          <summary>More about {artist.name}</summary>
          <p>{artist.profile}</p>
        </details>
      )}
      <div className={styles.artistSource}>
        <a href={artist.sourceUrl} target="_blank" rel="noopener noreferrer">
          Data provided by Discogs <ArrowUpRight size={12} />
        </a>
        <span>
          Connections are catalog listings, not a complete career history.
        </span>
      </div>
      <section
        className={styles.discography}
        aria-labelledby="artist-records-heading"
      >
        <div className={styles.discographyHeading}>
          <div>
            <span className={styles.kicker}>THE NEXT RABBIT HOLE</span>
            <h3 id="artist-records-heading">Follow the records.</h3>
            <p>Explore a release to meet the people behind it.</p>
          </div>
          {!discography && (
            <button
              className={styles.goldButton}
              onClick={() => onDiscography(1)}
              disabled={loadingDiscography}
            >
              {loadingDiscography ? (
                <LoaderCircle size={16} className="spin" />
              ) : (
                <Disc3 size={16} />
              )}{" "}
              Explore their records <ArrowRight size={15} />
            </button>
          )}
        </div>
        {discographyError && (
          <div className={styles.inlineError} role="alert">
            <p>{discographyError}</p>
            <button
              onClick={() => onDiscography(discographyRetryPage)}
              disabled={loadingDiscography}
            >
              <RefreshCw size={14} /> Try records again
            </button>
          </div>
        )}
        {discography && (
          <>
            <div className={styles.discographyCount} role="status">
              {loadingDiscography ? (
                <>
                  <LoaderCircle size={14} className="spin" /> Loading records…
                </>
              ) : (
                `${discography.total} catalog ${discography.total === 1 ? "entry" : "entries"} · Page ${discography.page} of ${Math.max(1, discography.pages)}`
              )}
            </div>
            {discography.releases.length ? (
              <ul className={styles.albumGrid} aria-busy={loadingDiscography}>
                {discography.releases.map((album) => (
                  <AlbumCard
                    key={`${album.type}-${album.id}`}
                    album={album}
                    onExplore={onAlbum}
                    ownedCopyId={ownedCopies.get(
                      album.type === "release"
                        ? album.id
                        : (album.mainReleaseId ?? -1),
                    )}
                  />
                ))}
              </ul>
            ) : (
              <p className={styles.emptyDiscography}>
                No discography entries are listed for this artist yet.
              </p>
            )}
            {discography.pages > 1 && (
              <nav
                className={styles.pagination}
                aria-label="Artist records pages"
              >
                <button
                  disabled={loadingDiscography || discography.page <= 1}
                  onClick={() => onDiscography(discography.page - 1)}
                >
                  <ChevronLeft size={16} /> Previous records
                </button>
                <span>
                  {discography.page} / {discography.pages}
                </span>
                <button
                  disabled={
                    loadingDiscography ||
                    discography.page >= Math.min(discography.pages, 500)
                  }
                  onClick={() => onDiscography(discography.page + 1)}
                >
                  Next records <ChevronRight size={16} />
                </button>
              </nav>
            )}
            {discography.pages > 500 && (
              <p className={styles.discographyCount}>
                WAX can explore the first 500 pages. See the artist’s Discogs
                page for the complete listing.
              </p>
            )}
          </>
        )}
      </section>
    </section>
  );
}

export function ArtistExplorer({ copyId }: { copyId: string }) {
  const { data } = useLibrary();
  const [start, setStart] = useState<ExplorerStart | null>(null);
  const [frame, setFrame] = useState<Frame | null>(null);
  const [trail, setTrail] = useState<TrailItem[]>([]);
  const [initialLoading, setInitialLoading] = useState(true);
  const [initialError, setInitialError] = useState("");
  const [navigation, setNavigation] = useState<NavigationRequest | null>(null);
  const [navigationError, setNavigationError] = useState("");
  const [navigationRetry, setNavigationRetry] =
    useState<NavigationRequest | null>(null);
  const [discographyLoading, setDiscographyLoading] = useState(false);
  const [discographyError, setDiscographyError] = useState("");
  const [discographyRetryPage, setDiscographyRetryPage] = useState(1);
  const [artistQuery, setArtistQuery] = useState("");
  const [titleQuery, setTitleQuery] = useState("");
  const [candidates, setCandidates] = useState<CatalogCandidate[]>([]);
  const [searched, setSearched] = useState(false);
  const [searchLoading, setSearchLoading] = useState(false);
  const [searchError, setSearchError] = useState("");
  const [showMatchPicker, setShowMatchPicker] = useState(false);
  const frameRef = useRef(frame);
  frameRef.current = frame;
  const initialRequest = useRef<AbortController | null>(null);
  const navigationRequest = useRef<AbortController | null>(null);
  const discographyRequest = useRef<AbortController | null>(null);
  const searchRequest = useRef<AbortController | null>(null);
  const navigationSequence = useRef(0);
  const discographySequence = useRef(0);
  const searchSequence = useRef(0);
  const studio = useRef<HTMLDivElement>(null);

  const loadStart = useCallback(async () => {
    initialRequest.current?.abort();
    const controller = new AbortController();
    initialRequest.current = controller;
    setInitialLoading(true);
    setInitialError("");
    try {
      const result = await api<ExplorerStart>(
        `/api/records/${encodeURIComponent(copyId)}/explore`,
        { signal: requestSignal(controller) },
      );
      if (controller.signal.aborted) return;
      setStart(result);
      setFrame(result.release ? newFrame(result.release) : null);
      setTrail([]);
      setArtistQuery(result.artist);
      setTitleQuery(result.title);
      setShowMatchPicker(!result.release);
    } catch (err) {
      if (!controller.signal.aborted) setInitialError(messageFor(err));
    } finally {
      if (!controller.signal.aborted) setInitialLoading(false);
    }
  }, [copyId]);

  useEffect(() => {
    void loadStart();
    return () => {
      initialRequest.current?.abort();
      navigationRequest.current?.abort();
      discographyRequest.current?.abort();
      searchRequest.current?.abort();
      navigationSequence.current++;
      discographySequence.current++;
      searchSequence.current++;
    };
  }, [loadStart]);

  function resetPendingNavigation() {
    navigationSequence.current++;
    navigationRequest.current?.abort();
    discographySequence.current++;
    discographyRequest.current?.abort();
    setNavigation(null);
    setNavigationError("");
    setNavigationRetry(null);
    setDiscographyLoading(false);
    setDiscographyError("");
  }

  async function navigate(request: NavigationRequest) {
    resetPendingNavigation();
    const sequence = navigationSequence.current;
    const controller = new AbortController();
    navigationRequest.current = controller;
    setNavigation(request);
    try {
      if (request.kind === "artist") {
        if (!frameRef.current) return;
        const artist = await api<ExplorerArtist>(
          `/api/explore/artists/${request.id}`,
          { signal: requestSignal(controller) },
        );
        if (sequence !== navigationSequence.current) return;
        const current = frameRef.current;
        if (!current) return;
        setTrail((history) => [
          ...history.slice(-14),
          {
            label: current.artist?.name || current.release.title,
            frame: current,
          },
        ]);
        setFrame({ ...current, artist, discography: null });
      } else {
        const release = await api<ExplorerRelease>(
          `/api/explore/releases/${request.id}`,
          { signal: requestSignal(controller) },
        );
        if (sequence !== navigationSequence.current) return;
        const current = frameRef.current;
        if (current)
          setTrail((history) => [
            ...history.slice(-14),
            {
              label: current.artist?.name || current.release.title,
              frame: current,
            },
          ]);
        setFrame(newFrame(release));
        setShowMatchPicker(false);
        studio.current?.scrollIntoView({ behavior: "instant", block: "start" });
      }
    } catch (err) {
      if (sequence !== navigationSequence.current) return;
      setNavigationError(messageFor(err));
      setNavigationRetry(request);
    } finally {
      if (sequence === navigationSequence.current) setNavigation(null);
    }
  }

  function selectPerson(person: ExplorerPerson) {
    if (person.id && person.id !== frame?.artist?.id)
      void navigate({ kind: "artist", id: person.id, name: person.name });
  }

  function selectAlbum(id: number, name: string) {
    void navigate({ kind: "release", id, name });
  }

  async function loadDiscography(page: number) {
    const artistId = frameRef.current?.artist?.id;
    if (!artistId) return;
    discographyRequest.current?.abort();
    const controller = new AbortController();
    discographyRequest.current = controller;
    const sequence = ++discographySequence.current;
    setDiscographyLoading(true);
    setDiscographyError("");
    setDiscographyRetryPage(page);
    try {
      const discography = await api<ExplorerDiscography>(
        `/api/explore/artists/${artistId}/releases?page=${page}`,
        { signal: requestSignal(controller) },
      );
      if (sequence !== discographySequence.current) return;
      setFrame((current) =>
        current?.artist?.id === artistId
          ? { ...current, discography }
          : current,
      );
    } catch (err) {
      if (sequence === discographySequence.current)
        setDiscographyError(messageFor(err));
    } finally {
      if (sequence === discographySequence.current)
        setDiscographyLoading(false);
    }
  }

  function restoreTrail(index: number) {
    const entry = trail[index];
    if (!entry) return;
    resetPendingNavigation();
    setFrame(entry.frame);
    setTrail((current) => current.slice(0, index));
    setShowMatchPicker(false);
  }

  async function searchMatches(event: FormEvent) {
    event.preventDefault();
    if (!artistQuery.trim() && !titleQuery.trim()) return;
    searchRequest.current?.abort();
    const controller = new AbortController();
    searchRequest.current = controller;
    const sequence = ++searchSequence.current;
    setSearchLoading(true);
    setSearchError("");
    const query = new URLSearchParams();
    if (artistQuery.trim()) query.set("artist", artistQuery.trim());
    if (titleQuery.trim()) query.set("title", titleQuery.trim());
    try {
      const result = await api<{ candidates: CatalogCandidate[] }>(
        `/api/catalog?${query}`,
        { signal: requestSignal(controller) },
      );
      if (sequence !== searchSequence.current) return;
      setCandidates(result.candidates);
      setSearched(true);
    } catch (err) {
      if (sequence === searchSequence.current) setSearchError(messageFor(err));
    } finally {
      if (sequence === searchSequence.current) setSearchLoading(false);
    }
  }

  const release = frame?.release;
  const track = release?.tracks.find((item) => item.id === frame?.trackId);
  const tracks = release?.tracks.filter((item) => item.type === "track") ?? [];
  const artistNames =
    release?.artists.map((person) => person.name).join(", ") ||
    start?.artist ||
    "";
  const chooseTrack = (id: string) => {
    resetPendingNavigation();
    setFrame((current) =>
      current
        ? { ...current, trackId: id, artist: null, discography: null }
        : current,
    );
  };
  const ownedCopies = new Map<number, string>();
  for (const record of data?.records ?? []) {
    if (record.discogsId && !ownedCopies.has(record.discogsId))
      ownedCopies.set(record.discogsId, record.id);
  }

  function changeSearchQuery(field: "artist" | "title", value: string) {
    resetPendingNavigation();
    searchSequence.current++;
    searchRequest.current?.abort();
    setSearchLoading(false);
    setSearchError("");
    setSearched(false);
    setCandidates([]);
    if (field === "artist") setArtistQuery(value);
    else setTitleQuery(value);
  }

  return (
    <div className={styles.explorer}>
      <div className={styles.topbar}>
        <Link href={`/records/${encodeURIComponent(copyId)}`}>
          <ArrowLeft size={15} /> Back to your record
        </Link>
        <span>
          <GitBranch size={14} /> WAX / ARTIST TREE
        </span>
      </div>
      <header className={styles.hero}>
        <div className={styles.heroCopy}>
          <span className={styles.kicker}>
            <span /> THE GOOD STUFF GOES DEEPER
          </span>
          <h1>
            Follow the <em>groove.</em>
          </h1>
          <p>
            One song. A whole constellation of people.
            <br />
            Meet the players. Trace the connections. Find your next obsession.
          </p>
        </div>
        <div className={styles.heroGraphic} aria-hidden="true">
          <svg viewBox="0 0 280 190" fill="none">
            <path
              d="M140 92 45 45M140 92 225 35M140 92 250 140M140 92 70 155"
              stroke="currentColor"
              strokeWidth="1.5"
            />
            <circle cx="45" cy="45" r="17" />
            <circle cx="225" cy="35" r="10" />
            <circle cx="250" cy="140" r="22" />
            <circle cx="70" cy="155" r="11" />
            <circle cx="140" cy="92" r="45" stroke="currentColor" />
            <circle cx="140" cy="92" r="36" stroke="currentColor" />
            <circle cx="140" cy="92" r="18" fill="currentColor" />
            <circle cx="140" cy="92" r="3" fill="#23231f" />
          </svg>
          <span>EVERY RECORD IS CONNECTED.</span>
        </div>
      </header>
      {initialLoading ? (
        <div className={styles.loading} role="status">
          <Disc3 size={28} className="spin" />
          <h2>Reading the liner notes…</h2>
        </div>
      ) : initialError ? (
        <div className={styles.lightError} role="alert">
          <p>{initialError}</p>
          <button onClick={() => void loadStart()}>
            <RefreshCw size={15} /> Try again
          </button>
        </div>
      ) : (
        <>
          {start && (
            <div className={styles.origin}>
              <span>STARTING IN YOUR COLLECTION</span>
              <p>
                {start.artist} <span>/</span> {start.title}
              </p>
              <span>
                <Check size={12} /> Exploring never changes your copy.
              </span>
            </div>
          )}
          {start && (!start.discogsId || showMatchPicker) && (
            <section
              className={styles.matchPicker}
              aria-labelledby="explorer-match-heading"
            >
              <div>
                <span className={styles.kicker}>FIND THE LINER NOTES</span>
                <h2 id="explorer-match-heading">
                  Choose a catalog release to explore.
                </h2>
                <p>
                  This is a look inside its credits, not confirmation of your
                  pressing. Your saved record stays as it is.
                </p>
              </div>
              {showMatchPicker ? (
                <>
                  <form onSubmit={searchMatches}>
                    <label>
                      Artist
                      <input
                        value={artistQuery}
                        onChange={(event) =>
                          changeSearchQuery("artist", event.target.value)
                        }
                        maxLength={150}
                        placeholder="Artist or group"
                      />
                    </label>
                    <label>
                      Release title
                      <input
                        value={titleQuery}
                        onChange={(event) =>
                          changeSearchQuery("title", event.target.value)
                        }
                        maxLength={150}
                        placeholder="Album or single"
                      />
                    </label>
                    <button
                      disabled={
                        searchLoading ||
                        (!artistQuery.trim() && !titleQuery.trim())
                      }
                    >
                      {searchLoading ? (
                        <LoaderCircle size={16} className="spin" />
                      ) : (
                        <Search size={16} />
                      )}{" "}
                      Find releases
                    </button>
                  </form>
                  <a
                    className={styles.matchAttribution}
                    href="https://www.discogs.com"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    Data provided by Discogs <ArrowUpRight size={11} />
                  </a>
                  {searchError && (
                    <p className={styles.searchError} role="alert">
                      {searchError}
                    </p>
                  )}
                  {searched && !candidates.length && (
                    <p className={styles.searchEmpty}>
                      No catalog matches yet. Try another title or artist
                      spelling.
                    </p>
                  )}
                  {candidates.length > 0 && (
                    <ul className={styles.matches}>
                      {candidates.map((candidate, index) => (
                        <li key={`${candidate.discogsId}-${index}`}>
                          <Cover record={candidate} />
                          <div>
                            <h3>{candidate.title}</h3>
                            <p>{candidate.artist}</p>
                            <small>
                              {[
                                candidate.year,
                                candidate.label,
                                candidate.catalogNumber,
                                candidate.country,
                                candidate.format,
                              ]
                                .filter(Boolean)
                                .join(" · ")}
                            </small>
                          </div>
                          <button
                            disabled={!candidate.discogsId}
                            onClick={() =>
                              candidate.discogsId &&
                              selectAlbum(candidate.discogsId, candidate.title)
                            }
                          >
                            Explore credits <ArrowRight size={14} />
                          </button>
                        </li>
                      ))}
                    </ul>
                  )}
                </>
              ) : (
                <button
                  className={styles.matchAgain}
                  onClick={() => setShowMatchPicker(true)}
                >
                  Choose another catalog release <Search size={14} />
                </button>
              )}
            </section>
          )}
          <div className={styles.studio} ref={studio}>
            {trail.length > 0 && (
              <nav className={styles.trail} aria-label="Exploration trail">
                <button onClick={() => restoreTrail(trail.length - 1)}>
                  <ArrowLeft size={14} /> Back one step
                </button>
                <ol>
                  {trail.slice(-4).map((item, index) => (
                    <li
                      key={`${trail.length - Math.min(trail.length, 4) + index}-${item.frame.release.id}-${item.frame.artist?.id || 0}`}
                    >
                      <button
                        onClick={() =>
                          restoreTrail(
                            trail.length - Math.min(trail.length, 4) + index,
                          )
                        }
                      >
                        {item.label}
                      </button>
                      <ChevronRight size={12} />
                    </li>
                  ))}
                  <li>
                    <span>{frame?.artist?.name || release?.title}</span>
                  </li>
                </ol>
              </nav>
            )}
            {(navigation || navigationError) && (
              <div
                className={
                  navigationError ? styles.inlineError : styles.navigationStatus
                }
                role={navigationError ? "alert" : "status"}
              >
                {navigation ? (
                  <>
                    <LoaderCircle size={16} className="spin" />
                    <span>Following {navigation.name}…</span>
                  </>
                ) : (
                  <>
                    <p>{navigationError}</p>
                    <button
                      onClick={() =>
                        navigationRetry && void navigate(navigationRetry)
                      }
                    >
                      <RefreshCw size={14} /> Try connection again
                    </button>
                  </>
                )}
              </div>
            )}
            {release && frame && (
              <>
                <section
                  className={styles.releaseHeader}
                  aria-labelledby="exploring-release-heading"
                >
                  <div className={styles.releaseCover}>
                    <Cover
                      record={{
                        coverUrl: release.coverUrl,
                        artist: artistNames,
                        title: release.title,
                      }}
                    />
                  </div>
                  <div className={styles.releaseIdentity}>
                    <span className={styles.kicker}>
                      ON THE TURNTABLE / CREDITS EXPLORER
                    </span>
                    <h2 id="exploring-release-heading">{release.title}</h2>
                    <div className={styles.releaseArtists}>
                      {release.artists.map((person, index) => (
                        <ArtistLink
                          key={`${person.id}-${index}`}
                          person={person}
                          onSelect={selectPerson}
                        />
                      ))}
                    </div>
                    <p>
                      {[
                        release.year,
                        release.label,
                        release.catalogNumber,
                        release.country,
                      ]
                        .filter(Boolean)
                        .join(" · ")}
                    </p>
                    <a
                      href={release.sourceUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      Data provided by Discogs <ArrowUpRight size={12} />
                    </a>
                  </div>
                  <span className={styles.trackCount}>
                    <Disc3 size={18} />
                    <strong>{tracks.length.toString().padStart(2, "0")}</strong>
                    <span>TRACKS TO FOLLOW</span>
                  </span>
                </section>
                <div className={styles.releaseBody}>
                  <aside className={styles.trackRail}>
                    <div className={styles.trackRailHeading}>
                      <span className={styles.kicker}>START WITH A SONG</span>
                      <h3>
                        The track list<span>.</span>
                      </h3>
                    </div>
                    {tracks.length ? (
                      <>
                        <label className={styles.mobileTrackSelect}>
                          Choose a song
                          <select
                            value={frame.trackId || ""}
                            onChange={(event) =>
                              chooseTrack(event.target.value)
                            }
                          >
                            {tracks.map((item) => (
                              <option key={item.id} value={item.id}>
                                {item.position ? `${item.position} · ` : ""}
                                {item.title}
                              </option>
                            ))}
                          </select>
                        </label>
                        <ol className={styles.trackList} aria-label="Tracks">
                          {release.tracks.map((item) =>
                            item.type === "heading" ? (
                              <li className={styles.trackHeading} key={item.id}>
                                {item.title}
                              </li>
                            ) : (
                              <li key={item.id}>
                                <button
                                  aria-current={
                                    item.id === frame.trackId
                                      ? "true"
                                      : undefined
                                  }
                                  onClick={() => chooseTrack(item.id)}
                                >
                                  <span>{item.position || "—"}</span>
                                  <strong>{item.title}</strong>
                                  <small>{item.duration}</small>
                                </button>
                              </li>
                            ),
                          )}
                        </ol>
                      </>
                    ) : (
                      <p className={styles.noTracks}>
                        No track list is available for this release. You can
                        still explore the release credits and artists.
                      </p>
                    )}
                    <div className={styles.trackRailFooter}>
                      <Music2 size={16} />
                      <p>No playback. Just the people who made it happen.</p>
                    </div>
                  </aside>
                  <div className={styles.creditsPanel}>
                    <section aria-labelledby="song-credits-heading">
                      <div className={styles.songHeading}>
                        <span className={styles.kicker}>
                          {track
                            ? `TRACK ${track.position || "—"} / LINER NOTES`
                            : "LINER NOTES"}
                        </span>
                        <h3>{track?.title || "The people on this record"}</h3>
                        <span className={styles.creditScope}>
                          <Mic2 size={12} />{" "}
                          <span id="song-credits-heading">On this song</span>
                        </span>
                      </div>
                      {track?.credits.length ? (
                        <CreditGroups
                          credits={track.credits}
                          onSelect={selectPerson}
                        />
                      ) : (
                        <div className={styles.emptyCredits}>
                          <Disc3 size={24} strokeWidth={1.1} />
                          <p>
                            <strong>No specific song credits listed.</strong>
                            That doesn’t mean nobody played. This catalog entry
                            just doesn’t identify the people on this track.
                          </p>
                        </div>
                      )}
                    </section>
                    <section
                      className={styles.releaseCredits}
                      aria-labelledby="release-credits-heading"
                    >
                      <div className={styles.creditSectionHeading}>
                        <h3 id="release-credits-heading">Release credits</h3>
                        <span>TRACKS NOT STATED</span>
                      </div>
                      <p className={styles.scopeNote}>
                        These credits are listed for this release without
                        specific tracks. They aren’t confirmed for the selected
                        song.
                      </p>
                      {release.releaseCredits.length ? (
                        <CreditGroups
                          credits={release.releaseCredits}
                          onSelect={selectPerson}
                        />
                      ) : (
                        <p className={styles.noReleaseCredits}>
                          No additional release credits are listed. Try the main
                          artists above to keep exploring.
                        </p>
                      )}
                    </section>
                    {release.unresolvedCredits.length > 0 && (
                      <details className={styles.unresolved}>
                        <summary>
                          Credits with unresolved track references{" "}
                          <span>{release.unresolvedCredits.length}</span>
                        </summary>
                        <p>
                          The source names tracks we couldn’t reliably match.
                          Original references are shown below.
                        </p>
                        <CreditGroups
                          credits={release.unresolvedCredits}
                          onSelect={selectPerson}
                          unresolved
                        />
                      </details>
                    )}
                    <div className={styles.followHint}>
                      <GitBranch size={18} />
                      <p>
                        Click a name. Find their groups, collaborators, and next
                        chapter.
                      </p>
                      <ArrowDown size={16} />
                    </div>
                  </div>
                </div>
                {frame.artist ? (
                  <ArtistTree
                    artist={frame.artist}
                    discography={frame.discography}
                    onPerson={selectPerson}
                    onAlbum={selectAlbum}
                    onDiscography={(page) => void loadDiscography(page)}
                    loadingDiscography={discographyLoading}
                    discographyError={discographyError}
                    discographyRetryPage={discographyRetryPage}
                    ownedCopies={ownedCopies}
                    onClose={() => {
                      resetPendingNavigation();
                      setFrame((current) =>
                        current
                          ? { ...current, artist: null, discography: null }
                          : current,
                      );
                    }}
                  />
                ) : (
                  <div className={styles.treeInvitation}>
                    <div>
                      <Users size={27} strokeWidth={1.3} />
                      <Sparkles size={16} />
                    </div>
                    <h2>It’s all connected.</h2>
                    <p>
                      Choose an artist or a credited name above to grow your
                      Artist Tree.
                    </p>
                  </div>
                )}
                <footer className={styles.footer}>
                  <p>
                    Credits, artist relationships, and discography supplied by{" "}
                    <a
                      href="https://www.discogs.com"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      Discogs <ArrowUpRight size={11} />
                    </a>
                    . Community listings can be incomplete or incorrect.
                  </p>
                  <span>KEEP FOLLOWING THE GOOD STUFF.</span>
                </footer>
              </>
            )}
          </div>
        </>
      )}
    </div>
  );
}
