"use client";
export default function Error({
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <main className="auth-content" style={{ minHeight: "70vh" }}>
      <div className="auth-form">
        <span className="eyebrow">A LITTLE SKIP IN THE GROOVE</span>
        <h2>Let’s try that again.</h2>
        <p>
          Something interrupted the page. Your saved collection is still there.
        </p>
        <button className="button button-primary" onClick={reset}>
          Try again
        </button>
      </div>
    </main>
  );
}
