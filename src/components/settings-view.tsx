"use client";
import { useEffect, useState, type FormEvent } from "react";
import { useRouter } from "next/navigation";
import {
  Check,
  Copy,
  Download,
  LogOut,
  MailPlus,
  ShieldCheck,
} from "lucide-react";
import { api, useLibrary } from "./library-provider";
import { authClient } from "./auth-client";
import { Button, ErrorNotice, Field, PageHeader, dateLabel } from "./ui";
interface Invitation {
  id: string;
  email: string;
  kind: string;
  expiresAt: string;
  consumedAt: string | null;
  revokedAt: string | null;
}
export function SettingsView() {
  const { data } = useLibrary();
  const router = useRouter();
  const [invitations, setInvitations] = useState<Invitation[]>([]);
  const [email, setEmail] = useState("");
  const [link, setLink] = useState("");
  const [expires, setExpires] = useState("");
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [copied, setCopied] = useState(false);
  const owner = data!.collector.isOwner;
  async function load() {
    const result = await api<{ invitations: Invitation[] }>("/api/invitations");
    setInvitations(result.invitations);
  }
  useEffect(() => {
    if (!owner) return;
    let active = true;
    void api<{ invitations: Invitation[] }>("/api/invitations")
      .then((result) => {
        if (active) setInvitations(result.invitations);
      })
      .catch((err) => {
        if (active)
          setError(
            err instanceof Error ? err.message : "Could not load invitations.",
          );
      });
    return () => {
      active = false;
    };
  }, [owner]);
  async function invite(e: FormEvent) {
    e.preventDefault();
    setBusy("invite");
    setError("");
    try {
      const result = await api<{ url: string; expiresAt: string }>(
        "/api/invitations",
        { method: "POST", body: JSON.stringify({ email }) },
      );
      setLink(result.url);
      setExpires(result.expiresAt);
      setEmail("");
      setCopied(false);
      await load();
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Could not make that invitation.",
      );
    } finally {
      setBusy("");
    }
  }
  async function revoke(id: string) {
    setBusy(id);
    setError("");
    try {
      await api("/api/invitations/" + id, { method: "DELETE" });
      await load();
      setLink("");
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Could not revoke that invitation.",
      );
    } finally {
      setBusy("");
    }
  }
  async function signOut() {
    setBusy("out");
    setError("");
    try {
      const result = await authClient.signOut();
      if (result.error)
        throw new Error(result.error.message || "Could not sign out.");
      router.replace("/sign-in");
      router.refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not sign out.");
    } finally {
      setBusy("");
    }
  }
  async function copy() {
    try {
      await navigator.clipboard.writeText(link);
      setCopied(true);
    } catch {
      setError(
        "Your browser could not copy the invitation. Select and copy the link below.",
      );
    }
  }
  return (
    <>
      <PageHeader
        eyebrow="KEEP YOUR HOUSE IN ORDER"
        title="Back of the shop"
        description="Your account, your data, your circle of collectors."
      />
      <ErrorNotice>{error}</ErrorNotice>
      <div className="settings-grid">
        <section>
          {owner ? (
            <div className="settings-panel">
              <h2>Good company, by invitation.</h2>
              <p>
                Invite a collector to their own private library. Create a link
                here, then share it with them directly.
              </p>
              <form onSubmit={invite}>
                <Field label="Collector’s email address">
                  <input
                    type="email"
                    required
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="a.friend@example.com"
                  />
                </Field>
                <Button type="submit" busy={busy === "invite"}>
                  <MailPlus size={16} /> Create invitation
                </Button>
              </form>
              {link && (
                <div className="invite-link">
                  <strong>One invitation. One collector.</strong>
                  <p style={{ marginTop: 7 }}>
                    Expires {dateLabel(expires)}. This link is private.
                  </p>
                  <input
                    aria-label="Private invitation link"
                    readOnly
                    value={link}
                    onFocus={(e) => e.currentTarget.select()}
                    style={{ marginTop: 10, fontSize: 10 }}
                  />
                  <Button variant="secondary" onClick={() => void copy()}>
                    {copied ? <Check size={14} /> : <Copy size={14} />}{" "}
                    {copied ? "Copied" : "Copy invitation link"}
                  </Button>
                </div>
              )}
              {invitations.length > 0 && (
                <div style={{ marginTop: 25 }}>
                  <span className="eyebrow">THE GUEST LIST</span>
                  {invitations.map((invitation) => (
                    <div key={invitation.id} className="invite-row">
                      <div>
                        {invitation.email}
                        <small>
                          {invitation.consumedAt
                            ? "Joined " + dateLabel(invitation.consumedAt)
                            : invitation.revokedAt
                              ? "Revoked"
                              : new Date(invitation.expiresAt) < new Date()
                                ? "Expired"
                                : "Expires " + dateLabel(invitation.expiresAt)}
                        </small>
                      </div>
                      {!invitation.consumedAt &&
                        !invitation.revokedAt &&
                        new Date(invitation.expiresAt) > new Date() && (
                          <button
                            disabled={busy === invitation.id}
                            onClick={() => void revoke(invitation.id)}
                          >
                            {busy === invitation.id ? "Revoking…" : "Revoke"}
                          </button>
                        )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          ) : (
            <div className="settings-panel">
              <ShieldCheck size={28} />
              <h2 style={{ marginTop: 15 }}>Your shelf. Your rules.</h2>
              <p>
                Your library is private to you. Share selected crates with other
                invited collectors from the crates page. Only the collection
                owner can invite new collectors.
              </p>
            </div>
          )}
          <div className="settings-panel">
            <h2>Take your collection with you.</h2>
            <p>
              Download every physical copy and its collection details as a CSV.
              Your data always belongs to you.
            </p>
            <a className="button button-secondary" href="/api/export" download>
              <Download size={15} /> Export my collection
            </a>
          </div>
        </section>
        <aside>
          <div className="settings-panel">
            <span className="eyebrow">THE COLLECTOR</span>
            <div className="profile-info">
              <span>{data!.collector.name.slice(0, 1).toUpperCase()}</span>
              <div>
                <strong>{data!.collector.name}</strong>
                <small>{data!.collector.email}</small>
              </div>
            </div>
            <p style={{ marginTop: 22 }}>
              {owner
                ? "Owner & independent collector"
                : "Independent collector"}{" "}
              · {data!.records.length} records
            </p>
            <Button
              variant="secondary"
              busy={busy === "out"}
              onClick={() => void signOut()}
            >
              <LogOut size={15} /> Sign out
            </Button>
          </div>
          <div className="settings-panel">
            <h2>Built for the long play.</h2>
            <p>
              Keep the records. Keep the stories. Everything you save here is
              part of your real collection.
            </p>
            <span className="eyebrow" style={{ color: "var(--orange)" }}>
              WAX® · KEEP THE GOOD STUFF.
            </span>
          </div>
        </aside>
      </div>
    </>
  );
}
