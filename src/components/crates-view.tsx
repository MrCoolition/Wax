"use client";
import { useState, type CSSProperties, type FormEvent } from "react";
import {
  ArrowUpRight,
  Boxes,
  Disc3,
  Plus,
  Share2,
  Users,
  Pencil,
  Trash2,
} from "lucide-react";
import type { Crate, SharedCrate } from "@/lib/contracts";
import { api, useLibrary } from "./library-provider";
import {
  Button,
  Cover,
  ErrorNotice,
  Field,
  Modal,
  PageHeader,
  SourceCredit,
} from "./ui";
import { RecordGrid } from "./collection-view";

type CrateDraft = {
  id: string | null;
  name: string;
  description: string;
  color: string;
  recordIds: string[];
};
export function CratesView() {
  const { data, refresh } = useLibrary();
  const [draft, setDraft] = useState<CrateDraft | null>(null);
  const [view, setView] = useState<string | null>(null);
  const [shareId, setShareId] = useState<string | null>(null);
  const [sharedView, setSharedView] = useState<SharedCrate | null>(null);
  const [removeId, setRemoveId] = useState<string | null>(null);
  const [email, setEmail] = useState("");
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [search, setSearch] = useState("");
  const sharing = data!.crates.find((c) => c.id === shareId);
  const selected = data!.crates.find((c) => c.id === view);
  function edit(crate?: Crate) {
    setError("");
    setSearch("");
    setDraft(
      crate
        ? { ...crate }
        : {
            id: null,
            name: "",
            description: "",
            color: "#d4ad4b",
            recordIds: [],
          },
    );
  }
  async function save(e: FormEvent) {
    e.preventDefault();
    if (!draft) return;
    setBusy("save");
    setError("");
    try {
      if (draft.id) {
        await api("/api/crates/" + draft.id, {
          method: "PATCH",
          body: JSON.stringify({
            name: draft.name,
            description: draft.description,
            color: draft.color,
            recordIds: draft.recordIds,
          }),
        });
      } else {
        const made = await api<Crate>("/api/crates", {
          method: "POST",
          body: JSON.stringify({
            name: draft.name,
            description: draft.description,
            color: draft.color,
          }),
        });
        setDraft({ ...draft, id: made.id });
        if (draft.recordIds.length)
          await api("/api/crates/" + made.id, {
            method: "PATCH",
            body: JSON.stringify({ recordIds: draft.recordIds }),
          });
      }
      await refresh();
      setDraft(null);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Could not save that crate.",
      );
    } finally {
      setBusy("");
    }
  }
  async function remove() {
    if (!removeId) return;
    setBusy("remove");
    setError("");
    try {
      await api("/api/crates/" + removeId, { method: "DELETE" });
      await refresh();
      setRemoveId(null);
      setView(null);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Could not remove that crate.",
      );
    } finally {
      setBusy("");
    }
  }
  async function share(e: FormEvent) {
    e.preventDefault();
    setBusy("share");
    setError("");
    try {
      await api("/api/crates/" + shareId + "/shares", {
        method: "POST",
        body: JSON.stringify({ email }),
      });
      await refresh();
      setEmail("");
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Could not share that crate.",
      );
    } finally {
      setBusy("");
    }
  }
  async function revoke(userId: string) {
    setBusy(userId);
    setError("");
    try {
      await api("/api/crates/" + shareId + "/shares", {
        method: "DELETE",
        body: JSON.stringify({ userId }),
      });
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not revoke access.");
    } finally {
      setBusy("");
    }
  }
  return (
    <>
      <PageHeader
        eyebrow="A GOOD HOME FOR EVERY GOOD RECORD"
        title="Your crates"
        description="File by feeling. By genre. By whatever gets you listening."
        action={
          <Button onClick={() => edit()}>
            <Plus size={17} /> New crate
          </Button>
        }
      />
      {selected ? (
        <section>
          <div className="section-intro">
            <div>
              <button className="text-button" onClick={() => setView(null)}>
                ← All your crates
              </button>
              <h2>{selected.name}</h2>
              <p>
                {selected.description ||
                  selected.recordIds.length + " records, picked by you."}
              </p>
            </div>
            <Button variant="secondary" onClick={() => edit(selected)}>
              <Pencil size={14} /> Edit crate
            </Button>
          </div>
          {selected.recordIds.length ? (
            <RecordGrid
              records={data!.records.filter((r) =>
                selected.recordIds.includes(r.id),
              )}
            />
          ) : (
            <div className="empty-small">
              <Boxes size={33} />
              <h3>Ready for the good stuff.</h3>
              <p>Add a few records to give this crate its own sound.</p>
              <Button variant="secondary" onClick={() => edit(selected)}>
                Choose records
              </Button>
            </div>
          )}
        </section>
      ) : data!.crates.length ? (
        <div className="crate-grid">
          {data!.crates.map((crate) => (
            <article
              className="crate-card"
              id={crate.id}
              key={crate.id}
              style={{ "--crate-color": crate.color } as CSSProperties}
            >
              <div className="crate-card-header">
                <button onClick={() => setView(crate.id)}>
                  <h2>{crate.name}</h2>
                </button>
                <button
                  className="icon-button"
                  aria-label={"Edit " + crate.name}
                  onClick={() => edit(crate)}
                >
                  <Pencil size={15} />
                </button>
              </div>
              <p>{crate.description || "A crate with your name on it."}</p>
              <div className="crate-count">
                <Disc3 size={13} />
                {crate.recordIds.length}{" "}
                {crate.recordIds.length === 1 ? "record" : "records"}
              </div>
              <button
                style={{ width: "100%", display: "block", textAlign: "left" }}
                onClick={() => setView(crate.id)}
                aria-label={"Open " + crate.name}
              >
                {crate.recordIds.length ? (
                  <div className="crate-previews">
                    {data!.records
                      .filter((r) => crate.recordIds.includes(r.id))
                      .slice(0, 3)
                      .map((record) => (
                        <Cover key={record.id} record={record} />
                      ))}
                  </div>
                ) : (
                  <div className="crate-empty-art">
                    <Boxes size={28} />
                    <span>ROOM FOR THE GOOD STUFF</span>
                  </div>
                )}
              </button>
              <div className="crate-card-footer">
                <button
                  onClick={() => {
                    setShareId(crate.id);
                    setError("");
                  }}
                >
                  <Share2 size={13} />
                  {crate.shares.length
                    ? "Shared with " + crate.shares.length
                    : "Share crate"}
                </button>
                <button onClick={() => setView(crate.id)}>
                  Dig in <ArrowUpRight size={14} />
                </button>
                <button
                  aria-label={"Delete " + crate.name}
                  onClick={() => {
                    setRemoveId(crate.id);
                    setError("");
                  }}
                >
                  <Trash2 size={13} />
                </button>
              </div>
            </article>
          ))}
        </div>
      ) : (
        <div className="empty-small">
          <Boxes size={40} />
          <h3>Every mood deserves a crate.</h3>
          <p>
            Sunday soul. Late-night jazz. Your forever favorites. Make a home
            for the records that belong together.
          </p>
          <Button onClick={() => edit()}>
            <Plus size={16} /> Make your first crate
          </Button>
        </div>
      )}
      <section id="shared" className="shared-section">
        <div className="section-intro">
          <div>
            <h2>Good records. Better company.</h2>
            <p>
              Crates shared with you, straight from another collector’s shelf.
            </p>
          </div>
          <Users size={25} />
        </div>
        {data!.sharedCrates.length ? (
          <div className="crate-grid">
            {data!.sharedCrates.map((crate) => (
              <article
                className="crate-card"
                key={crate.id}
                style={{ "--crate-color": crate.color } as CSSProperties}
              >
                <h2>{crate.name}</h2>
                <div className="shared-owner">
                  FROM {crate.ownerName.toUpperCase()}
                </div>
                <p>{crate.description || "A selection worth sharing."}</p>
                <div className="crate-previews">
                  {crate.records.slice(0, 3).map((record) => (
                    <Cover key={record.id} record={record} />
                  ))}
                </div>
                <div className="crate-card-footer">
                  <span>{crate.records.length} records · Read only</span>
                  <button onClick={() => setSharedView(crate)}>
                    Take a look <ArrowUpRight size={15} />
                  </button>
                </div>
              </article>
            ))}
          </div>
        ) : (
          <div className="empty-small">
            <Share2 size={28} />
            <h3>Good company is on the way.</h3>
            <p>
              When another invited collector shares a crate with you, it lands
              here.
            </p>
          </div>
        )}
      </section>
      <Modal
        open={Boolean(draft)}
        onOpenChange={(open) => {
          if (!open) setDraft(null);
        }}
        title={draft?.id ? "Tune up your crate." : "Make a little room."}
        description="One record can live in as many crates as you like."
      >
        {draft && (
          <form onSubmit={save}>
            <div className="form-grid">
              <Field label="Crate name">
                <input
                  required
                  maxLength={100}
                  value={draft.name}
                  onChange={(e) => setDraft({ ...draft, name: e.target.value })}
                  placeholder="Sunday morning soul"
                />
              </Field>
              <Field label="Divider color">
                <input
                  type="color"
                  value={draft.color}
                  onChange={(e) =>
                    setDraft({ ...draft, color: e.target.value })
                  }
                />
              </Field>
              <Field
                label="A few words about this crate"
                className="full-width"
              >
                <textarea
                  rows={2}
                  value={draft.description}
                  onChange={(e) =>
                    setDraft({ ...draft, description: e.target.value })
                  }
                  maxLength={2000}
                  placeholder="What brings these records together?"
                />
              </Field>
            </div>
            <div className="section-title">
              <span>PICK YOUR RECORDS · {draft.recordIds.length} SELECTED</span>
            </div>
            {data!.records.length ? (
              <>
                <input
                  aria-label="Find records for this crate"
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  placeholder="Find a record…"
                />
                <div className="record-check-list">
                  {data!.records
                    .filter((r) =>
                      (r.title + " " + r.artist)
                        .toLowerCase()
                        .includes(search.toLowerCase()),
                    )
                    .map((record) => (
                      <label key={record.id}>
                        <input
                          type="checkbox"
                          checked={draft.recordIds.includes(record.id)}
                          onChange={(e) =>
                            setDraft({
                              ...draft,
                              recordIds: e.target.checked
                                ? [...draft.recordIds, record.id]
                                : draft.recordIds.filter(
                                    (id) => id !== record.id,
                                  ),
                            })
                          }
                        />
                        <Cover record={record} />
                        <span>
                          {record.title}
                          <small>{record.artist}</small>
                        </span>
                      </label>
                    ))}
                </div>
              </>
            ) : (
              <p className="small-note">
                Add records to your collection, then file them here.
              </p>
            )}
            <ErrorNotice>{error}</ErrorNotice>
            <div className="modal-actions">
              <Button
                variant="secondary"
                type="button"
                onClick={() => setDraft(null)}
              >
                Cancel
              </Button>
              <Button type="submit" busy={busy === "save"}>
                Save crate
              </Button>
            </div>
          </form>
        )}
      </Modal>
      <Modal
        open={Boolean(shareId)}
        onOpenChange={(open) => {
          if (!open) setShareId(null);
        }}
        title={"Share " + (sharing?.name || "your crate")}
        description="Share read-only with an existing invited collector. Prices, private notes, shelf locations, photos, and listening history stay yours."
      >
        {sharing?.shares.map((person) => (
          <div key={person.userId} className="share-row">
            <div>
              {person.name}
              <small>{person.email}</small>
            </div>
            <button
              disabled={busy === person.userId}
              onClick={() => void revoke(person.userId)}
            >
              {busy === person.userId ? "Removing…" : "Revoke access"}
            </button>
          </div>
        ))}
        <form className="share-form" onSubmit={share}>
          <input
            type="email"
            required
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="Collector’s email address"
            aria-label="Collector email"
          />
          <Button type="submit" busy={busy === "share"}>
            Share crate
          </Button>
        </form>
        <ErrorNotice>{error}</ErrorNotice>
      </Modal>
      <Modal
        open={Boolean(removeId)}
        onOpenChange={(open) => {
          if (!open) setRemoveId(null);
        }}
        title="Retire this crate?"
        description="Your records stay in your collection. The crate and its sharing permissions will be removed."
      >
        <ErrorNotice>{error}</ErrorNotice>
        <div className="modal-actions">
          <Button variant="secondary" onClick={() => setRemoveId(null)}>
            Keep it
          </Button>
          <Button
            variant="danger"
            busy={busy === "remove"}
            onClick={() => void remove()}
          >
            Remove crate
          </Button>
        </div>
      </Modal>
      <Modal
        open={Boolean(sharedView)}
        onOpenChange={(open) => {
          if (!open) setSharedView(null);
        }}
        title={sharedView?.name || "Shared crate"}
        description={"Shared by " + sharedView?.ownerName + " · Read only"}
      >
        {sharedView?.records.map((record) => (
          <article
            className="candidate-card"
            key={record.id}
            style={{ marginTop: 15 }}
          >
            <Cover record={record} />
            <div>
              <h3>{record.title}</h3>
              <p>{record.artist}</p>
              <small>
                {[
                  record.year,
                  record.label,
                  record.catalogNumber,
                  record.country,
                  record.format,
                ]
                  .filter(Boolean)
                  .join(" · ")}
              </small>
              <small>
                Media {record.mediaCondition} · Sleeve {record.sleeveCondition}
              </small>
              <small>
                {record.pressingConfirmed
                  ? "Pressing confirmed"
                  : "Pressing unconfirmed"}
              </small>
              <SourceCredit sourceUrl={record.sourceUrl} />
            </div>
          </article>
        ))}
        {sharedView?.records.length === 0 && (
          <p className="small-note">
            This crate is waiting for its first record.
          </p>
        )}
      </Modal>
    </>
  );
}
