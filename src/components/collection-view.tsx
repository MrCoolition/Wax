"use client";
import Link from "next/link";
import { useMemo, useState } from "react";
import {
  ArrowUpRight,
  Plus,
  SlidersHorizontal,
  LayoutGrid,
  List,
  Search,
  ScanLine,
  Camera,
  Disc3,
  Star,
  ArrowRight,
  Boxes,
  X,
} from "lucide-react";
import { useLibrary } from "./library-provider";
import {
  Button,
  Cover,
  PageHeader,
  SourceCredit,
  Vinyl,
  dateLabel,
} from "./ui";
import { CONDITIONS, type RecordCopy } from "@/lib/contracts";

function compareAlbumYear(a: RecordCopy, b: RecordCopy, newestFirst: boolean) {
  if (a.originalYear == null)
    return b.originalYear == null ? a.title.localeCompare(b.title) : 1;
  if (b.originalYear == null) return -1;
  return (
    (newestFirst
      ? b.originalYear - a.originalYear
      : a.originalYear - b.originalYear) ||
    a.artist.localeCompare(b.artist) ||
    a.title.localeCompare(b.title)
  );
}

export function RecordCard({ record }: { record: RecordCopy }) {
  return (
    <article className="record-card">
      <Link href={"/records/" + record.id} className="record-cover-link">
        <Cover record={record} />
        <span className="record-number">
          {record.format.includes("45") ? "45" : "33⅓"}
        </span>
        <span className="record-open">
          <ArrowUpRight size={20} />
        </span>
        {record.rating > 0 && (
          <span className="record-rating">
            <Star size={11} fill="currentColor" />
            {record.rating}
          </span>
        )}
      </Link>
      <div className="record-caption">
        <Link href={"/records/" + record.id}>
          <h3>{record.title}</h3>
          <p>{record.artist}</p>
        </Link>
        <div className="record-facts">
          <span>
            {record.originalYear ?? "Year unknown"}
            {record.genres[0] ? " · " + record.genres[0] : ""}
          </span>
          <span>
            {record.mediaCondition === "Not graded"
              ? "—"
              : record.mediaCondition}
          </span>
        </div>
        <SourceCredit sourceUrl={record.sourceUrl} />
      </div>
    </article>
  );
}
export function RecordGrid({ records }: { records: RecordCopy[] }) {
  return (
    <div className="record-grid">
      {records.map((record) => (
        <RecordCard key={record.id} record={record} />
      ))}
    </div>
  );
}
export function CollectionView() {
  const { data, search, setSearch } = useLibrary();
  const [mode, setMode] = useState<"grid" | "table">("grid");
  const [sort, setSort] = useState("newest");
  const [genre, setGenre] = useState("");
  const [condition, setCondition] = useState("");
  const [location, setLocation] = useState("");
  const [more, setMore] = useState(false);
  const records = data?.records || [];
  const genres = useMemo(
    () => [...new Set(records.flatMap((record) => record.genres))].sort(),
    [records],
  );
  const locations = useMemo(
    () =>
      [
        ...new Set(records.map((record) => record.location).filter(Boolean)),
      ].sort(),
    [records],
  );
  const filtered = useMemo(
    () =>
      records
        .filter(
          (record) =>
            (!search ||
              [
                record.artist,
                record.title,
                record.label,
                record.catalogNumber,
                record.notes,
                ...record.genres,
              ]
                .join(" ")
                .toLowerCase()
                .includes(search.toLowerCase())) &&
            (!genre || record.genres.includes(genre)) &&
            (!condition || record.mediaCondition === condition) &&
            (!location || record.location === location),
        )
        .sort((a, b) =>
          sort === "artist"
            ? a.artist.localeCompare(b.artist) || a.title.localeCompare(b.title)
            : sort === "title"
              ? a.title.localeCompare(b.title)
              : sort === "rating"
                ? b.rating - a.rating
                : sort === "year-oldest"
                  ? compareAlbumYear(a, b, false)
                  : sort === "year-newest"
                    ? compareAlbumYear(a, b, true)
                    : sort === "unplayed"
                      ? a.spinCount - b.spinCount
                      : b.addedAt.localeCompare(a.addedAt),
        ),
    [records, search, genre, condition, location, sort],
  );
  const reset = () => {
    setSearch("");
    setGenre("");
    setCondition("");
    setLocation("");
  };
  return (
    <>
      <PageHeader
        eyebrow="SIDE A / YOUR PERSONAL ARCHIVE"
        title="The collection"
        description="Every sleeve has a story. This is yours."
        action={
          <Link className="button button-primary" href="/add">
            <Plus size={18} /> Add some wax
          </Link>
        }
      />
      <div className="stat-strip">
        <div>
          <strong>{records.length.toLocaleString()}</strong>
          <span>RECORDS & COUNTING</span>
        </div>
        <div>
          <strong>
            {new Set(records.map((r) => r.artist.toLowerCase())).size}
          </strong>
          <span>ARTISTS IN THE MIX</span>
        </div>
        <div>
          <strong>
            {records.reduce((total, record) => total + record.spinCount, 0)}
          </strong>
          <span>SPINS LOGGED</span>
        </div>
        <div className="stat-crates">
          <strong>{data?.crates.length || 0}</strong>
          <span>CRATES OF GOOD STUFF</span>
        </div>
        <Disc3 className="stat-disc" size={50} strokeWidth={1} />
      </div>
      <div className="collection-tabs">
        <span className="active-tab">
          All records <b>{records.length}</b>
        </span>
        <Link href="/crates">
          Your crates <b>{data?.crates.length || 0}</b>
        </Link>
        <Link href="/crates#shared">
          Shared with you <b>{data?.sharedCrates.length || 0}</b>
        </Link>
      </div>
      {records.length === 0 ? (
        <>
          <div className="empty-collection">
            <div className="empty-copy">
              <span className="eyebrow">
                AN EMPTY CRATE. ENDLESS POSSIBILITIES.
              </span>
              <h2>
                EVERY GREAT
                <br />
                COLLECTION
                <br />
                STARTS WITH <em>ONE.</em>
              </h2>
              <p>
                That first pressing. That dollar-bin discovery. The record you
                couldn’t leave behind. Give it a home.
              </p>
              <Link href="/add" className="button button-primary">
                Drop your first record <ArrowUpRight size={18} />
              </Link>
            </div>
            <div className="empty-art" aria-hidden="true">
              <Vinyl />
              <div className="empty-sleeve">
                <span>
                  INDEPENDENT
                  <br />
                  SELECTS
                </span>
                <strong>
                  GOOD
                  <br />
                  STUFF.
                </strong>
                <div>
                  WAX RECORD LIBRARY <span>VOL. 001</span>
                </div>
              </div>
              <div className="round-sticker">
                ALL SOUL
                <br />
                <span>NO SKIPS</span>
              </div>
            </div>
          </div>
          <div className="entry-methods">
            <Link href="/add?mode=search">
              <span className="method-number">01</span>
              <Search />
              <div>
                <h3>Know the record?</h3>
                <p>Search the artist, title, or catalog number.</p>
              </div>
              <ArrowUpRight />
            </Link>
            <Link href="/add?mode=scan">
              <span className="method-number">02</span>
              <ScanLine />
              <div>
                <h3>Find the barcode.</h3>
                <p>Point your camera. Find your pressing.</p>
              </div>
              <ArrowUpRight />
            </Link>
            <Link href="/add?mode=photo">
              <span className="method-number">03</span>
              <Camera />
              <div>
                <h3>Let the sleeve talk.</h3>
                <p>Snap a photo. We’ll help identify it.</p>
              </div>
              <ArrowUpRight />
            </Link>
          </div>
        </>
      ) : (
        <>
          <div className="collection-toolbar">
            <div className="toolbar-left">
              <label className="select-wrap">
                <span className="sr-only">Filter by genre</span>
                <select
                  value={genre}
                  onChange={(e) => setGenre(e.target.value)}
                >
                  <option value="">All genres</option>
                  {genres.map((g) => (
                    <option key={g}>{g}</option>
                  ))}
                </select>
              </label>
              <Button
                variant="quiet"
                className={more ? "is-selected" : ""}
                onClick={() => setMore(!more)}
                aria-expanded={more}
              >
                <SlidersHorizontal size={15} /> Filters{" "}
                {(condition || location) && <i className="filter-dot" />}
              </Button>
              {(search || genre || condition || location) && (
                <button className="clear-filter" onClick={reset}>
                  Clear <X size={13} />
                </button>
              )}
            </div>
            <div className="toolbar-right">
              <label className="select-wrap">
                <span className="sr-only">Sort records</span>
                <select value={sort} onChange={(e) => setSort(e.target.value)}>
                  <option value="newest">Recently added</option>
                  <option value="artist">Artist A–Z</option>
                  <option value="title">Title A–Z</option>
                  <option value="year-oldest">Album year · oldest first</option>
                  <option value="year-newest">Album year · newest first</option>
                  <option value="rating">Highest rated</option>
                  <option value="unplayed">Least played</option>
                </select>
              </label>
              <div className="view-toggle">
                <button
                  aria-label="Grid view"
                  aria-pressed={mode === "grid"}
                  onClick={() => setMode("grid")}
                >
                  <LayoutGrid size={17} />
                </button>
                <button
                  aria-label="Table view"
                  aria-pressed={mode === "table"}
                  onClick={() => setMode("table")}
                >
                  <List size={18} />
                </button>
              </div>
            </div>
          </div>
          {more && (
            <div className="filter-panel">
              <label>
                Media condition
                <select
                  value={condition}
                  onChange={(e) => setCondition(e.target.value)}
                >
                  <option value="">Any condition</option>
                  {CONDITIONS.map((c) => (
                    <option key={c}>{c}</option>
                  ))}
                </select>
              </label>
              <label>
                Shelf location
                <select
                  value={location}
                  onChange={(e) => setLocation(e.target.value)}
                >
                  <option value="">Any location</option>
                  {locations.map((l) => (
                    <option key={l}>{l}</option>
                  ))}
                </select>
              </label>
            </div>
          )}
          <div className="result-count">
            {filtered.length} {filtered.length === 1 ? "record" : "records"}
            {search && (
              <>
                {" "}
                matching <strong>“{search}”</strong>
              </>
            )}
            <span>FILED WITH LOVE.</span>
          </div>
          {filtered.length === 0 ? (
            <div className="empty-small">
              <Search size={32} />
              <h3>No records in this groove.</h3>
              <p>Try another search or loosen up the filters.</p>
              <Button variant="secondary" onClick={reset}>
                Clear filters
              </Button>
            </div>
          ) : mode === "grid" ? (
            <RecordGrid records={filtered} />
          ) : (
            <div className="table-scroll">
              <table className="record-table">
                <thead>
                  <tr>
                    <th>RECORD</th>
                    <th>ALBUM YEAR / GENRE</th>
                    <th>CONDITION</th>
                    <th>LOCATION</th>
                    <th>ADDED</th>
                  </tr>
                </thead>
                <tbody>
                  {filtered.map((record) => (
                    <tr key={record.id}>
                      <td>
                        <Link
                          className="table-record"
                          href={"/records/" + record.id}
                        >
                          <Cover record={record} />
                          <span>
                            <strong>{record.title}</strong>
                            <small>{record.artist}</small>
                          </span>
                        </Link>
                      </td>
                      <td>
                        {record.originalYear ?? "Year unknown"}
                        <small>{record.genres.join(", ") || "—"}</small>
                      </td>
                      <td>
                        <span className="grade-badge">
                          {record.mediaCondition}
                        </span>
                      </td>
                      <td>{record.location || "Not filed yet"}</td>
                      <td>{dateLabel(record.addedAt)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          <div className="end-of-crate">
            <span />
            <Disc3 size={18} />
            <p>That’s the crate. Keep digging.</p>
            <span />
          </div>
        </>
      )}
      <div className="bottom-callout">
        <Boxes size={28} />
        <div>
          <strong>A place for every pressing.</strong>
          <p>
            Build a crate for your Sunday mornings, all-time favorites, or next
            listening session.
          </p>
        </div>
        <Link href="/crates">
          Make it a crate <ArrowRight size={17} />
        </Link>
      </div>
    </>
  );
}
