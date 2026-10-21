import Link from "next/link";
export default function NotFound() {
  return (
    <main className="auth-content" style={{ minHeight: "100vh" }}>
      <div className="auth-form">
        <span className="eyebrow">WRONG GROOVE / 404</span>
        <h2>This side is blank.</h2>
        <p>That page may have moved. Your collection is one click away.</p>
        <Link className="button button-primary" href="/collection">
          Back to the good stuff
        </Link>
      </div>
    </main>
  );
}
