"use client";
import { useEffect, useState, type FormEvent } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { ArrowRight, LockKeyhole } from "lucide-react";
import { authClient } from "./auth-client";
import { api } from "./library-provider";
import { Button, ErrorNotice, Field, SuccessNotice, Vinyl } from "./ui";
export function AuthView({
  join = false,
  recovery = false,
}: {
  join?: boolean;
  recovery?: boolean;
}) {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmation, setConfirmation] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [name, setName] = useState("");
  const [token, setToken] = useState("");
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(join || recovery);
  const [error, setError] = useState("");
  const [accepted, setAccepted] = useState(false);
  const choosingPassword = (join || recovery) && !accepted;
  useEffect(() => {
    if (!join && !recovery) return;
    const invitation =
      new URLSearchParams(window.location.hash.slice(1)).get("token") || "";
    if (!invitation) {
      setError(
        recovery
          ? "This recovery link is missing its private code. Open the complete password reset link."
          : "This invitation link is missing its private code. Open the complete link from the collection owner.",
      );
      setLoading(false);
      return;
    }
    setToken(invitation);
    let active = true;
    void api<{ email: string }>(
      recovery
        ? "/api/recovery/inspect"
        : "/api/invitations/accept?" +
            new URLSearchParams({ token: invitation }),
      recovery
        ? { method: "POST", body: JSON.stringify({ token: invitation }) }
        : undefined,
    )
      .then((result) => {
        if (active) setEmail(result.email);
      })
      .catch((err) => {
        if (active)
          setError(
            err instanceof Error
              ? err.message
              : "This private link could not be opened. Please try again.",
          );
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [join, recovery]);
  async function submit(e: FormEvent) {
    e.preventDefault();
    if (busy) return;
    if (recovery && !accepted && password !== confirmation) {
      setError("The passwords do not match. Enter the same password twice.");
      return;
    }
    setBusy(true);
    setError("");
    try {
      if (recovery && !accepted) {
        await api("/api/recovery", {
          method: "POST",
          body: JSON.stringify({ token, password }),
        });
        setAccepted(true);
        setConfirmation("");
        setShowPassword(false);
        window.history.replaceState(null, "", "/recover");
        return;
      }
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
            {recovery
              ? "BACK TO THE GOOD STUFF."
              : join
                ? "YOUR NAME’S ON THE LIST."
                : "WELCOME BACK TO THE GOOD STUFF."}
          </span>
          <h2>
            {recovery
              ? accepted
                ? "You’re back in tune."
                : "A fresh start."
              : join
                ? "Come on in."
                : "Your records await."}
          </h2>
          <p>
            {recovery
              ? "Choose a new password for your existing account. Your records stay right where you left them."
              : join
                ? "Make yourself at home. Your own private collection starts right here."
                : "The finds. The favorites. The ones you’ll never let go. All in one good place."}
          </p>
          {loading ? (
            <p role="status">
              {recovery
                ? "Checking your recovery link…"
                : "Checking your invitation…"}
            </p>
          ) : (
            <form method="post" autoComplete="on" onSubmit={submit}>
              {recovery && accepted && (
                <SuccessNotice>
                  Your password is updated. Save it in your password manager,
                  then sign in below.
                </SuccessNotice>
              )}
              {join && (
                <Field label="Full name">
                  <input
                    required
                    id="wax-name"
                    name="name"
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
                  id="wax-email"
                  name="username"
                  required
                  autoComplete="username"
                  value={email}
                  readOnly={join || recovery}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="you@example.com"
                />
              </Field>
              <Field
                label={recovery && !accepted ? "New password" : "Password"}
                hint={
                  choosingPassword
                    ? "At least 12 characters. Save this password in your browser or password manager."
                    : undefined
                }
              >
                <input
                  type={showPassword ? "text" : "password"}
                  id="wax-password"
                  name="password"
                  required
                  minLength={choosingPassword ? 12 : 1}
                  maxLength={128}
                  autoComplete={
                    choosingPassword ? "new-password" : "current-password"
                  }
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder={
                    choosingPassword ? "Choose a password" : "Your password"
                  }
                />
              </Field>
              {recovery && !accepted && (
                <Field label="Confirm new password">
                  <input
                    type={showPassword ? "text" : "password"}
                    id="wax-password-confirmation"
                    name="password-confirmation"
                    required
                    minLength={12}
                    maxLength={128}
                    autoComplete="new-password"
                    value={confirmation}
                    onChange={(e) => setConfirmation(e.target.value)}
                    placeholder="Enter it once more"
                  />
                </Field>
              )}
              <button
                className="text-button"
                type="button"
                aria-controls={
                  recovery && !accepted
                    ? "wax-password wax-password-confirmation"
                    : "wax-password"
                }
                aria-pressed={showPassword}
                onClick={() => setShowPassword((visible) => !visible)}
              >
                {showPassword ? "Hide password" : "Show password"}
              </button>
              <ErrorNotice>{error}</ErrorNotice>
              <Button
                type="submit"
                busy={busy}
                disabled={(join || recovery) && !accepted && (!email || !token)}
              >
                {recovery && !accepted
                  ? "Update my password"
                  : join && !accepted
                    ? "Open my collection"
                    : "Sign in"}{" "}
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
            {recovery
              ? "This private recovery link works once. Updating your password signs out your other sessions."
              : join
                ? "Your collection is private. Share only the crates you choose."
                : "WAX is an invited circle of collectors. Ask the owner for an invitation to start your collection."}
          </p>
          {(join || recovery) && (
            <Link className="text-button" href="/sign-in">
              Already have a collection? Sign in <ArrowRight size={13} />
            </Link>
          )}
        </div>
      </main>
    </div>
  );
}
