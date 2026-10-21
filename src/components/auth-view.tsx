"use client";
import { useEffect, useState, type FormEvent } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { ArrowRight, LockKeyhole } from "lucide-react";
import { authClient } from "./auth-client";
import { api } from "./library-provider";
import { Button, ErrorNotice, Field, Vinyl } from "./ui";
export function AuthView({ join = false }: { join?: boolean }) {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [token, setToken] = useState("");
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(join);
  const [error, setError] = useState("");
  const [accepted, setAccepted] = useState(false);
  useEffect(() => {
    if (!join) return;
    const invitation =
      new URLSearchParams(window.location.hash.slice(1)).get("token") || "";
    if (!invitation) {
      setError(
        "This invitation link is missing its private code. Open the complete link from the collection owner.",
      );
      setLoading(false);
      return;
    }
    setToken(invitation);
    let active = true;
    void api<{ email: string }>(
      "/api/invitations/accept?" + new URLSearchParams({ token: invitation }),
    )
      .then((result) => {
        if (active) setEmail(result.email);
      })
      .catch((err) => {
        if (active)
          setError(
            err instanceof Error
              ? err.message
              : "This invitation could not be opened.",
          );
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [join]);
  async function submit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      if (join && !accepted) {
        await api("/api/invitations/accept", {
          method: "POST",
          body: JSON.stringify({ token, name, email, password }),
        });
        setAccepted(true);
        window.history.replaceState(null, "", "/join");
      }
      const result = await authClient.signIn.email({ email, password });
      if (result.error)
        throw new Error(
          result.error.message ||
            "That email and password did not match. Try again.",
        );
      router.replace("/collection");
      router.refresh();
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Could not open your collection. Please try again.",
      );
    } finally {
      setBusy(false);
    }
  }
  return (
    <div className="auth-layout">
      <aside className="auth-art">
        <Link href="/sign-in" className="wordmark">
          WAX<span>®</span>
        </Link>
        <span className="eyebrow">FOR THE LOVE OF THE RECORD.</span>
        <h1>
          GOOD
          <br />
          RECORDS.
          <br />
          GREAT
          <br />
          <em>STORIES.</em>
        </h1>
        <p>
          A home for the records you love, the copies you treasure, and the next
          great find.
        </p>
        <Vinyl />
        <div className="auth-art-footer">
          <span>INDEPENDENT SPIRIT. ENDLESS GROOVES.</span>
          <span>33⅓ RPM</span>
        </div>
      </aside>
      <main className="auth-content">
        <div className="auth-form">
          <div className="auth-mobile-logo">
            <Link href="/sign-in" className="wordmark">
              WAX<span>®</span>
            </Link>
          </div>
          <span className="eyebrow">
            {join
              ? "YOUR NAME’S ON THE LIST."
              : "WELCOME BACK TO THE GOOD STUFF."}
          </span>
          <h2>{join ? "Come on in." : "Your records await."}</h2>
          <p>
            {join
              ? "Make yourself at home. Your own private collection starts right here."
              : "The finds. The favorites. The ones you’ll never let go. All in one good place."}
          </p>
          {loading ? (
            <p role="status">Checking your invitation…</p>
          ) : (
            <form onSubmit={submit}>
              {join && (
                <Field label="Full name">
                  <input
                    required
                    autoComplete="name"
                    maxLength={150}
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    placeholder="What should we call you?"
                  />
                </Field>
              )}
              <Field label="Email address">
                <input
                  type="email"
                  required
                  autoComplete="email"
                  value={email}
                  readOnly={join}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="you@example.com"
                />
              </Field>
              <Field
                label="Password"
                hint={
                  join
                    ? "At least 12 characters. Make it a good one."
                    : undefined
                }
              >
                <input
                  type="password"
                  required
                  minLength={join ? 12 : 1}
                  maxLength={128}
                  autoComplete={join ? "new-password" : "current-password"}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder={join ? "Choose a password" : "Your password"}
                />
              </Field>
              <ErrorNotice>{error}</ErrorNotice>
              <Button
                type="submit"
                busy={busy}
                disabled={join && (!email || !token)}
              >
                {join ? "Open my collection" : "Sign in"}{" "}
                <ArrowRight size={16} />
              </Button>
            </form>
          )}
          {loading && <ErrorNotice>{error}</ErrorNotice>}
          <p className="auth-bottom">
            <LockKeyhole
              size={12}
              style={{
                display: "inline",
                verticalAlign: "middle",
                marginRight: 6,
              }}
            />
            {join
              ? "Your collection is private. Share only the crates you choose."
              : "WAX is an invited circle of collectors. Ask the owner for an invitation to start your collection."}
          </p>
          {join && (
            <Link className="text-button" href="/sign-in">
              Already have a collection? Sign in <ArrowRight size={13} />
            </Link>
          )}
        </div>
      </main>
    </div>
  );
}
