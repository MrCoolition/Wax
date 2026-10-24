"use client";
import { useRef, useState, type FormEvent } from "react";
import Link from "next/link";
import {
  ArrowUp,
  ArrowUpRight,
  Check,
  Disc3,
  Compass,
  Sparkles,
  X,
} from "lucide-react";
import type { WizardReply } from "@/lib/contracts";
import { api, useLibrary } from "./library-provider";
import { Button, Cover, ErrorNotice, PageHeader, Vinyl } from "./ui";
import styles from "./wizard-view.module.css";
interface Message {
  role: "user" | "assistant";
  content: string;
  reply?: WizardReply;
  proposalStatus?: "approved" | "rejected";
}
export function WizardView() {
  const { data, refresh } = useLibrary();
  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const textarea = useRef<HTMLTextAreaElement>(null);
  async function ask(e: FormEvent) {
    e.preventDefault();
    if (!message.trim() || busy) return;
    const question = message.trim();
    setBusy("chat");
    setError("");
    try {
      const reply = await api<WizardReply>("/api/wizard", {
        method: "POST",
        body: JSON.stringify({
          message: question,
          history: messages
            .slice(-12)
            .map(({ role, content }) => ({ role, content })),
        }),
      });
      setMessages((previous) => [
        ...previous,
        { role: "user", content: question },
        { role: "assistant", content: reply.reply, reply },
      ]);
      setMessage((current) => (current.trim() === question ? "" : current));
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "The Wizard lost the groove. Try again in a moment.",
      );
    } finally {
      setBusy("");
    }
  }
  async function decide(index: number, action: "approve" | "reject") {
    const proposal = messages[index]?.reply?.proposal;
    if (!proposal) return;
    setBusy(proposal.id);
    setError("");
    try {
      await api("/api/wizard/proposals/" + proposal.id, {
        method: "POST",
        body: JSON.stringify({ action }),
      });
      setMessages((previous) =>
        previous.map((item, i) =>
          i === index
            ? {
                ...item,
                proposalStatus: action === "approve" ? "approved" : "rejected",
              }
            : item,
        ),
      );
      if (action === "approve") await refresh();
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Could not update that proposal.",
      );
    } finally {
      setBusy("");
    }
  }
  const prompts = [
    {
      text: "Which Michael Jackson album should I buy first, and why?",
      icon: Compass,
    },
    {
      text: "Who played on Aja, and what else should I hear them on?",
      icon: Sparkles,
    },
    {
      text: "How do I identify a great pressing of the White Album?",
      icon: Disc3,
    },
    { text: "Find me something I’ve overlooked.", icon: Disc3 },
  ];
  return (
    <>
      <PageHeader
        eyebrow="MUSIC OBSESSED. VINYL TO THE CORE."
        title="Wax Wizard"
        description="A second set of ears for your first love."
      />
      <div className="wizard-intro">
        <div>
          <h2>
            YOU’VE GOT GOOD TASTE.
            <br />
            LET’S DIG A LITTLE DEEPER.
          </h2>
          <p>
            The players. The sessions. The pressing worth hunting. From your
            next must-have to the story behind the grooves—let’s get into it.
          </p>
        </div>
        <Vinyl label="WIZ" />
      </div>
      <div className="wizard-prompts">
        {prompts.map(({ text, icon: Icon }) => (
          <button
            key={text}
            onClick={() => {
              setMessage(text);
              textarea.current?.focus();
            }}
          >
            <Icon size={19} />
            {text}
          </button>
        ))}
      </div>
      <section
        className="conversation"
        aria-live="polite"
        aria-label="Conversation with Wax Wizard"
      >
        {messages.map((item, index) => (
          <article className={"chat-message " + item.role} key={index}>
            <div className="message-label">
              {item.role === "assistant" ? (
                <>
                  <Sparkles size={13} /> WAX WIZARD
                </>
              ) : (
                "YOU, THE COLLECTOR"
              )}
            </div>
            <p>{item.content}</p>
            {!!item.reply?.sources?.length && (
              <nav
                className={styles.sources}
                aria-label="Sources checked for this answer"
              >
                <span>LINER NOTES / SOURCES CHECKED</span>
                {item.reply.sources.map((source) => (
                  <a
                    key={source.id}
                    href={source.url}
                    target="_blank"
                    rel="noreferrer"
                  >
                    {source.title} <ArrowUpRight size={12} />
                  </a>
                ))}
              </nav>
            )}
            {!!item.reply?.recommendations?.length && (
              <section
                className={styles.discovery}
                aria-label="Albums to explore"
              >
                <div className={styles.divider}>
                  <Compass size={15} /> YOUR NEXT DIG
                </div>
                <div className={styles.cards}>
                  {item.reply.recommendations.map((album, albumIndex) => (
                    <article className={styles.card} key={album.masterId}>
                      <div className={styles.sleeve}>
                        <Cover
                          record={{
                            artist: album.artist,
                            title: album.title,
                            coverUrl: album.coverUrl,
                          }}
                        />
                        <span className={styles.number}>
                          {String(albumIndex + 1).padStart(2, "0")}
                        </span>
                      </div>
                      <div className={styles.details}>
                        <span className="eyebrow">
                          {albumIndex === 0 ? "START HERE" : "KEEP DIGGING"}
                        </span>
                        <h3>{album.title}</h3>
                        <p className={styles.artist}>
                          {album.artist}
                          {album.originalYear ? ` · ${album.originalYear}` : ""}
                        </p>
                        <p className={styles.reason}>{album.reason}</p>
                        <Link
                          className="button button-secondary"
                          href={`/add?${new URLSearchParams({ artist: album.artist, title: album.title, master: String(album.masterId) })}`}
                        >
                          Explore pressings <ArrowUpRight size={16} />
                        </Link>
                        <a
                          className={styles.credit}
                          href={album.sourceUrl}
                          target="_blank"
                          rel="noreferrer"
                        >
                          Discogs album details ↗
                        </a>
                      </div>
                    </article>
                  ))}
                </div>
              </section>
            )}
            {item.reply?.recordIds.length ? (
              <section aria-label="From your collection">
                <div className={styles.divider}>
                  <Disc3 size={15} /> FROM YOUR COLLECTION
                </div>
                <div className="wizard-records">
                  {item.reply.recordIds.map((id) => {
                    const record = data!.records.find((r) => r.id === id);
                    return record ? (
                      <Link key={id} href={"/records/" + id}>
                        <Cover record={record} />
                        <span>
                          <strong>{record.title}</strong>
                          <small>{record.artist}</small>
                        </span>
                      </Link>
                    ) : null;
                  })}
                </div>
              </section>
            ) : null}
            {item.reply?.proposal && (
              <div className="proposal-box">
                <h3>
                  {item.proposalStatus === "approved"
                    ? "Changes approved."
                    : item.proposalStatus === "rejected"
                      ? "Proposal set aside."
                      : "A proposal, for your ears."}
                </h3>
                <p>{item.reply.proposal.summary}</p>
                {item.reply.proposal.changes.map((change) => {
                  const record = data!.records.find(
                    (r) => r.id === change.recordId,
                  );
                  return (
                    <div className="proposal-change" key={change.recordId}>
                      <strong>
                        {record
                          ? record.artist + " — " + record.title
                          : "Record no longer available"}
                      </strong>
                      <dl>
                        {Object.entries(change.changes).map(([key, value]) => (
                          <div key={key}>
                            <dt>{key.replace(/([A-Z])/g, " $1")}</dt>
                            <dd>{String(value) || "(clear this field)"}</dd>
                          </div>
                        ))}
                      </dl>
                    </div>
                  );
                })}
                {!item.proposalStatus ? (
                  <>
                    <p className="small-note" style={{ marginTop: 15 }}>
                      Nothing has changed yet. Approving applies these exact
                      edits to your collection.
                    </p>
                    <div className="proposal-actions">
                      <Button
                        disabled={Boolean(busy)}
                        busy={busy === item.reply.proposal.id}
                        onClick={() => void decide(index, "approve")}
                      >
                        <Check size={15} /> Approve changes
                      </Button>
                      <Button
                        variant="secondary"
                        disabled={Boolean(busy)}
                        onClick={() => void decide(index, "reject")}
                      >
                        <X size={14} /> Leave it as is
                      </Button>
                    </div>
                  </>
                ) : (
                  <p className="small-note" style={{ marginTop: 15 }}>
                    {item.proposalStatus === "approved"
                      ? "Saved to your collection."
                      : "Your records were not changed."}
                  </p>
                )}
              </div>
            )}
          </article>
        ))}
      </section>
      {busy === "chat" && (
        <p className={styles.thinking} role="status">
          <Sparkles size={16} /> Digging into your question…
        </p>
      )}
      <ErrorNotice>{error}</ErrorNotice>
      <div className="wizard-composer">
        <form onSubmit={ask}>
          <textarea
            ref={textarea}
            aria-label="Ask Wax Wizard"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            maxLength={4000}
            required
            placeholder={
              data!.records.length
                ? "What should I buy next? Who played on this? What should I spin?"
                : "Name an artist you love. Let’s find your first great record."
            }
            rows={2}
            onKeyDown={(e) => {
              if (
                e.key === "Enter" &&
                !e.shiftKey &&
                !e.nativeEvent.isComposing
              ) {
                e.preventDefault();
                e.currentTarget.form?.requestSubmit();
              }
            }}
          />
          <Button
            type="submit"
            disabled={!message.trim() || Boolean(busy)}
            busy={busy === "chat"}
            aria-label="Ask Wizard"
          >
            <ArrowUp size={18} />
          </Button>
        </form>
        <p>
          Discover beyond your shelf. Catalog matches to explore. Collection
          edits always need your approval.
        </p>
      </div>
    </>
  );
}
