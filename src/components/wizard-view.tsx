"use client";
import { useRef, useState, type FormEvent } from "react";
import Link from "next/link";
import {
  ArrowUp,
  Check,
  Disc3,
  Headphones,
  MapPin,
  Sparkles,
  X,
} from "lucide-react";
import type { WizardReply } from "@/lib/contracts";
import { api, useLibrary } from "./library-provider";
import { Button, Cover, ErrorNotice, PageHeader, Vinyl } from "./ui";
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
      setMessage("");
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
    { text: "Find me something I’ve overlooked.", icon: Disc3 },
    { text: "What should I spin tonight?", icon: Headphones },
    { text: "Which records still need a shelf location?", icon: MapPin },
  ];
  return (
    <>
      <PageHeader
        eyebrow="YOUR COLLECTION. A LITTLE EXTRA SOUL."
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
            Ask about your actual collection. Find a forgotten favorite. Sort
            out the details. Any proposed changes are yours to review before
            they happen.
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
            {item.reply?.recordIds.length ? (
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
                ? "Ask about your records, or find your next spin…"
                : "Ask the Wizard where to start with your collection…"
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
          Grounded in your collection. Suggested edits always need your
          approval.
        </p>
      </div>
    </>
  );
}
