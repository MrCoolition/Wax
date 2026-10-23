"use client";
import { useState, type FormEvent } from "react";
import { createPortal } from "react-dom";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  ArrowLeft,
  Pencil,
  Trash2,
  QrCode,
  Printer,
  Disc3,
  LockKeyhole,
  Check,
  RefreshCw,
  GitBranch,
} from "lucide-react";
import type { CatalogCandidate, CopyFields, RecordCopy } from "@/lib/contracts";
import { api, useLibrary } from "./library-provider";
import {
  Button,
  Cover,
  ErrorNotice,
  Modal,
  SourceCredit,
  dateLabel,
  Field,
  SuccessNotice,
} from "./ui";
import { CandidateFields, CopyForm } from "./record-form";

export function RecordDetail({ id }: { id: string }) {
  const { data, refresh } = useLibrary();
  const router = useRouter();
  const record = data?.records.find((item) => item.id === id);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState<RecordCopy | null>(null);
  const [removing, setRemoving] = useState(false);
  const [qr, setQr] = useState("");
  const [qrOpen, setQrOpen] = useState(false);
  const [spinOpen, setSpinOpen] = useState(false);
  const [spinNotes, setSpinNotes] = useState("");
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  async function save(event: FormEvent) {
    event.preventDefault();
    if (!draft) return;
    setBusy("save");
    setError("");
    try {
      const {
        artist,
        title,
        year,
        genres,
        label,
        catalogNumber,
        country,
        format,
        coverUrl,
        sourceUrl,
        discogsId,
        mediaCondition,
        sleeveCondition,
        purchasePrice,
        currency,
        acquiredAt,
        notes,
        rating,
        location,
        intent,
        pressingConfirmed,
        version,
        crateIds,
      } = draft;
      await api("/api/records/" + id, {
        method: "PATCH",
        body: JSON.stringify({
          version,
          candidate: {
            artist,
            title,
            year,
            genres: genres.filter(Boolean),
            label,
            catalogNumber,
            country,
            format,
            coverUrl,
            sourceUrl,
            discogsId,
          },
          mediaCondition,
          sleeveCondition,
          purchasePrice,
          currency,
          acquiredAt,
          notes,
          rating,
          location,
          intent,
          pressingConfirmed,
          crateIds,
        }),
      });
      await refresh();
      setEditing(false);
      setSuccess("Your copy is up to date.");
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Could not save these changes.",
      );
    } finally {
      setBusy("");
    }
  }
  async function remove() {
    if (!record) return;
    setBusy("delete");
    setError("");
    try {
      await api("/api/records/" + id, {
        method: "DELETE",
        body: JSON.stringify({ version: record.version }),
      });
      await refresh();
      router.push("/collection");
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Could not remove this copy.",
      );
    } finally {
      setBusy("");
    }
  }
  async function makeQr() {
    setError("");
    setBusy("qr");
    try {
      const QR = await import("qrcode");
      setQr(
        await QR.toDataURL(window.location.origin + "/records/" + id, {
          width: 480,
          margin: 2,
          color: { dark: "#23231f", light: "#ffffff" },
        }),
      );
      setQrOpen(true);
    } catch {
      setError("Could not create that label. Please try again.");
    } finally {
      setBusy("");
    }
  }
  async function logSpin(event: FormEvent) {
    event.preventDefault();
    setBusy("spin");
    setError("");
    try {
      await api("/api/listens", {
        method: "POST",
        body: JSON.stringify({
          recordId: id,
          notes: spinNotes,
          playedAt: new Date().toISOString(),
        }),
      });
      await refresh();
      setSpinOpen(false);
      setSpinNotes("");
      setSuccess("Another good spin, on the record.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not log this spin.");
    } finally {
      setBusy("");
    }
  }
  async function refreshMetadata() {
    if (!record?.discogsId) return;
    setBusy("metadata");
    setError("");
    try {
      await api<CatalogCandidate>(
        "/api/catalog/" + record.discogsId + "?refresh=true",
      );
      await refresh();
      setSuccess(
        "Discogs artwork refreshed. Your saved collection details are unchanged.",
      );
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Could not refresh the catalog artwork.",
      );
    } finally {
      setBusy("");
    }
  }
  if (!record)
    return (
      <div className="empty-small">
        <Disc3 size={38} />
        <h3>This record isn’t in your collection.</h3>
        <p>It may have been removed, or it belongs to another collector.</p>
        <Link className="button button-secondary" href="/collection">
          Back to the collection
        </Link>
      </div>
    );
  const listens = data!.listens.filter((item) => item.recordId === id);
  const crates = data!.crates.filter((crate) => crate.recordIds.includes(id));
  return (
    <>
      <Link href="/collection" className="detail-back">
        <ArrowLeft size={15} /> Back to the collection
      </Link>
      <SuccessNotice>{success}</SuccessNotice>
      {!editing && !removing && !spinOpen && <ErrorNotice>{error}</ErrorNotice>}
      {editing && draft ? (
        <form onSubmit={save}>
          <div className="section-intro">
            <div>
              <span className="eyebrow">MAKE IT YOURS</span>
              <h2>Edit your record.</h2>
              <p>Your edits stay separate from catalog metadata.</p>
            </div>
            <Button
              variant="quiet"
              type="button"
              onClick={() => {
                setEditing(false);
                setError("");
              }}
            >
              Cancel
            </Button>
          </div>
          <div className="review-layout">
            <div>
              <div className="section-title">
                <span>01 / SAVED COLLECTION DETAILS</span>
              </div>
              <CandidateFields
                value={draft}
                onChange={(candidate) => setDraft({ ...draft, ...candidate })}
              />
              <div className="section-title">
                <span>02 / YOUR COPY</span>
              </div>
              <CopyForm
                value={draft}
                onChange={(copy: CopyFields) => setDraft({ ...draft, ...copy })}
              />
              {data!.crates.length > 0 && (
                <fieldset className="crate-checkboxes">
                  <legend>File in your crates</legend>
                  {data!.crates.map((crate) => (
                    <label key={crate.id}>
                      <input
                        type="checkbox"
                        checked={draft.crateIds.includes(crate.id)}
                        onChange={(e) =>
                          setDraft({
                            ...draft,
                            crateIds: e.target.checked
                              ? [...draft.crateIds, crate.id]
                              : draft.crateIds.filter((v) => v !== crate.id),
                          })
                        }
                      />
                      <i style={{ background: crate.color }} />
                      {crate.name}
                    </label>
                  ))}
                </fieldset>
              )}
              <ErrorNotice>{error}</ErrorNotice>
              <div className="form-actions">
                <Button type="submit" busy={busy === "save"}>
                  Save changes <Check size={17} />
                </Button>
                <Button
                  type="button"
                  variant="quiet"
                  onClick={() => {
                    setEditing(false);
                    setError("");
                  }}
                >
                  Cancel
                </Button>
              </div>
            </div>
            <aside className="review-sleeve">
              <Cover record={draft} />
              <p className="small-note">
                One physical copy. Every detail belongs to this record.
              </p>
            </aside>
          </div>
        </form>
      ) : (
        <>
          <div className="record-detail">
            <aside className="detail-art">
              <Cover record={record} />
              <div className="detail-art-actions">
                <Button
                  onClick={() => {
                    setSpinOpen(true);
                    setError("");
                  }}
                >
                  <Disc3 size={17} /> Log a spin
                </Button>
                <Button
                  variant="secondary"
                  busy={busy === "qr"}
                  onClick={() => void makeQr()}
                >
                  <QrCode size={16} /> QR label
                </Button>
              </div>
              <SourceCredit sourceUrl={record.sourceUrl} />
              <Link
                className="button button-secondary"
                href={`/records/${record.id}/explore`}
              >
                <GitBranch size={16} /> Artist Tree
              </Link>
              {record.photoIds.length > 0 && (
                <div className="detail-photos">
                  {record.photoIds.map((photo) => (
                    <a
                      key={photo}
                      href={"/api/photos/" + photo}
                      target="_blank"
                      rel="noreferrer"
                      aria-label="Open your private record photo"
                    >
                      <img
                        src={"/api/photos/" + photo}
                        alt="Your photograph of this record"
                      />
                    </a>
                  ))}
                </div>
              )}
            </aside>
            <section>
              <div className="detail-heading">
                <span className="eyebrow">
                  FILED WITH LOVE / {record.format}
                </span>
                <h1>{record.title}</h1>
                <p className="detail-artist">{record.artist}</p>
                <div className="detail-tags">
                  {record.genres.map((genre) => (
                    <span key={genre}>{genre}</span>
                  ))}
                  <span>
                    {record.pressingConfirmed
                      ? "Pressing confirmed"
                      : "Pressing unconfirmed"}
                  </span>
                  {record.rating > 0 && (
                    <span>{"★".repeat(record.rating)}</span>
                  )}
                </div>
              </div>
              <dl className="detail-facts">
                {[
                  ["Year", record.year || "Unknown"],
                  ["Label", record.label || "Not recorded"],
                  ["Catalog no.", record.catalogNumber || "Not recorded"],
                  ["Country", record.country || "Not recorded"],
                  ["Media condition", record.mediaCondition],
                  ["Sleeve condition", record.sleeveCondition],
                  ["Shelf / location", record.location || "Not filed yet"],
                  [
                    "The plan",
                    record.intent === "keep"
                      ? "A keeper"
                      : record.intent === "trade"
                        ? "Open to trade"
                        : "Future shop stock",
                  ],
                  ["Acquired", dateLabel(record.acquiredAt)],
                  [
                    "Purchase price",
                    record.purchasePrice
                      ? record.currency + " " + record.purchasePrice
                      : "Not recorded",
                  ],
                  ["Spins", record.spinCount],
                  [
                    "Last played",
                    record.lastPlayedAt
                      ? dateLabel(record.lastPlayedAt)
                      : "Ready for a first spin",
                  ],
                ].map(([label, value]) => (
                  <div key={label}>
                    <dt>{label}</dt>
                    <dd>{value}</dd>
                  </div>
                ))}
              </dl>
              {record.notes && (
                <div className="detail-private">
                  <span className="detail-notes-label">
                    <LockKeyhole size={12} /> YOUR PRIVATE LINER NOTES
                  </span>
                  <p>{record.notes}</p>
                </div>
              )}
              {crates.length > 0 && (
                <div className="detail-tags">
                  {crates.map((crate) => (
                    <Link key={crate.id} href={"/crates#" + crate.id}>
                      <span style={{ borderLeft: "3px solid " + crate.color }}>
                        {crate.name}
                      </span>
                    </Link>
                  ))}
                </div>
              )}
              <div className="detail-meta-actions">
                <Button
                  variant="secondary"
                  onClick={() => {
                    setDraft({ ...record });
                    setEditing(true);
                    setError("");
                    setSuccess("");
                  }}
                >
                  <Pencil size={15} /> Edit this copy
                </Button>
                {record.discogsId && (
                  <Button
                    variant="quiet"
                    busy={busy === "metadata"}
                    onClick={() => void refreshMetadata()}
                  >
                    <RefreshCw size={14} /> Refresh Discogs artwork
                  </Button>
                )}
                <Button
                  variant="quiet"
                  onClick={() => {
                    setRemoving(true);
                    setError("");
                  }}
                >
                  <Trash2 size={14} /> Remove
                </Button>
              </div>
              <p className="small-note" style={{ marginTop: 20 }}>
                Added {dateLabel(record.addedAt)} · Private copy details are
                only visible to you.
              </p>
            </section>
          </div>
          {listens.length > 0 && (
            <section style={{ marginTop: 40 }}>
              <div className="section-intro">
                <div>
                  <h2>The listening history.</h2>
                  <p>Good records deserve another turn.</p>
                </div>
                <Link className="text-button" href="/listening">
                  All your spins
                </Link>
              </div>
              {listens.map((listen) => (
                <div className="listen-row" key={listen.id}>
                  <Disc3 size={24} />
                  <div>
                    <h3>{dateLabel(listen.playedAt)}</h3>
                    <p>{listen.notes || "Just you and the music."}</p>
                  </div>
                </div>
              ))}
            </section>
          )}
        </>
      )}
      <Modal
        open={removing}
        onOpenChange={setRemoving}
        title="Take this copy off the shelf?"
        description="This removes the physical copy and its listening logs from your collection. Other copies stay in place."
      >
        <ErrorNotice>{error}</ErrorNotice>
        <div className="modal-actions">
          <Button variant="secondary" onClick={() => setRemoving(false)}>
            Keep it
          </Button>
          <Button
            variant="danger"
            busy={busy === "delete"}
            onClick={() => void remove()}
          >
            Remove this copy
          </Button>
        </div>
      </Modal>
      <Modal
        open={spinOpen}
        onOpenChange={setSpinOpen}
        title="Put it on the record."
        description={record.artist + " — " + record.title}
      >
        <form onSubmit={logSpin}>
          <Field label="Listening notes">
            <textarea
              value={spinNotes}
              onChange={(e) => setSpinNotes(e.target.value)}
              maxLength={5000}
              placeholder="The mood, the moment, the track that hit…"
            />
          </Field>
          <ErrorNotice>{error}</ErrorNotice>
          <div className="modal-actions">
            <Button busy={busy === "spin"} type="submit">
              <Disc3 size={17} /> Log this spin
            </Button>
          </div>
        </form>
      </Modal>
      <Modal
        open={qrOpen}
        onOpenChange={setQrOpen}
        title="A little label. A whole story."
        description="The QR opens this record’s private page. You’ll need to be signed in to your collection."
      >
        <div className="qr-preview">
          {qr && <img src={qr} alt="QR code for your private record page" />}
          <h3>{record.title}</h3>
          <p>{record.artist}</p>
        </div>
        <div className="modal-actions">
          <Button onClick={() => window.print()}>
            <Printer size={17} /> Print label
          </Button>
        </div>
      </Modal>
      {qr &&
        typeof document !== "undefined" &&
        createPortal(
          <div className="print-label">
            <img src={qr} alt="Record QR" />
            <h2>{record.title}</h2>
            <p>{record.artist}</p>
            <small>WAX · YOUR PERSONAL RECORD ARCHIVE</small>
          </div>,
          document.body,
        )}
    </>
  );
}
