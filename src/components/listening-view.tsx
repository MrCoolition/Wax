"use client";
import { useState, type FormEvent } from "react";
import Link from "next/link";
import { AudioLines, Disc3, Sparkles } from "lucide-react";
import { api, useLibrary } from "./library-provider";
import {
  Button,
  Cover,
  ErrorNotice,
  Field,
  PageHeader,
  SuccessNotice,
  dateLabel,
} from "./ui";
export function ListeningView() {
  const { data, refresh } = useLibrary();
  const [recordId, setRecordId] = useState("");
  const [notes, setNotes] = useState("");
  const [playedAt, setPlayedAt] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  async function submit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError("");
    setSuccess("");
    try {
      await api("/api/listens", {
        method: "POST",
        body: JSON.stringify({
          recordId,
          notes,
          playedAt: playedAt
            ? new Date(playedAt).toISOString()
            : new Date().toISOString(),
        }),
      });
      await refresh();
      setNotes("");
      setPlayedAt("");
      setSuccess("Spin logged. Long live the long play.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not log this spin.");
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <PageHeader
        eyebrow="SIDE B / TAKE IT FOR A SPIN"
        title="On the deck"
        description="Less collecting dust. More collecting moments."
        action={
          <Link className="button button-secondary" href="/wizard">
            <Sparkles size={15} /> Pick my next spin
          </Link>
        }
      />
      <div className="listening-layout">
        <section>
          <div className="section-intro">
            <div>
              <h2>The listening journal.</h2>
              <p>{data!.listens.length} spins, and counting.</p>
            </div>
            <AudioLines size={25} />
          </div>
          {data!.listens.length ? (
            data!.listens.map((listen) => (
              <article className="listen-row" key={listen.id}>
                <Cover record={listen} />
                <div>
                  <Link href={"/records/" + listen.recordId}>
                    <h3>{listen.title}</h3>
                    <p>{listen.artist}</p>
                  </Link>
                  {listen.notes && <small>{listen.notes}</small>}
                </div>
                <time dateTime={listen.playedAt}>
                  {dateLabel(listen.playedAt)}
                </time>
              </article>
            ))
          ) : (
            <div className="empty-small">
              <Disc3 size={36} />
              <h3>Drop the needle.</h3>
              <p>
                Every listening session starts a little story. Log your first
                spin and keep the good ones close.
              </p>
            </div>
          )}
        </section>
        <aside className="listening-form">
          <h2>What’s spinning?</h2>
          <p>A whole album, one perfect side, or that track on repeat.</p>
          {data!.records.length ? (
            <form onSubmit={submit}>
              <Field label="Pick a record">
                <select
                  required
                  value={recordId}
                  onChange={(e) => setRecordId(e.target.value)}
                >
                  <option value="">Choose from your collection</option>
                  {data!.records.map((record) => (
                    <option key={record.id} value={record.id}>
                      {record.artist} — {record.title}
                    </option>
                  ))}
                </select>
              </Field>
              <Field label="When" hint="Leave blank to log right now.">
                <input
                  type="datetime-local"
                  value={playedAt}
                  onChange={(e) => setPlayedAt(e.target.value)}
                />
              </Field>
              <Field label="Listening notes">
                <textarea
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                  maxLength={5000}
                  placeholder="Set the scene. What stood out?"
                />
              </Field>
              <ErrorNotice>{error}</ErrorNotice>
              <SuccessNotice>{success}</SuccessNotice>
              <Button type="submit" busy={busy}>
                <Disc3 size={17} /> Log this spin
              </Button>
            </form>
          ) : (
            <Link href="/add" className="button button-primary">
              Add your first record
            </Link>
          )}
        </aside>
      </div>
    </>
  );
}
