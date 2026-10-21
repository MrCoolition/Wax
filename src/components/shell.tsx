"use client";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import {
  Disc3,
  LibraryBig,
  Boxes,
  AudioLines,
  Sparkles,
  Settings2,
  Search,
  ArrowUpRight,
  Plus,
  RefreshCw,
} from "lucide-react";
import { useLibrary } from "./library-provider";
import { Button, ErrorNotice } from "./ui";
import type { ReactNode } from "react";

const navigation = [
  {
    href: "/collection",
    label: "The collection",
    icon: LibraryBig,
    short: "Collection",
  },
  { href: "/crates", label: "Your crates", icon: Boxes, short: "Crates" },
  {
    href: "/listening",
    label: "On the deck",
    icon: AudioLines,
    short: "Listening",
  },
  { href: "/wizard", label: "Wax Wizard", icon: Sparkles, short: "Wizard" },
];
export function Shell({ children }: { children: ReactNode }) {
  const path = usePathname();
  const router = useRouter();
  const { data, search, setSearch, error, loading, refresh } = useLibrary();
  return (
    <div className="app-shell">
      <a className="skip-link" href="#main-content">
        Skip to collection
      </a>
      <aside className="sidebar">
        <Link href="/collection" className="wordmark" aria-label="WAX home">
          WAX<span>®</span>
        </Link>
        <p className="brand-tagline">
          FOR THE LOVE
          <br />
          OF THE RECORD.
        </p>
        <div className="side-divider" />
        <p className="side-caption">YOUR LITTLE RECORD SHOP</p>
        <nav className="side-nav" aria-label="Main navigation">
          {navigation.map(({ href, label, icon: Icon }) => (
            <Link
              key={href}
              href={href}
              className={path.startsWith(href) ? "active" : ""}
            >
              <Icon size={19} />
              <span>{label}</span>
              {href === "/collection" && data && <b>{data.records.length}</b>}
            </Link>
          ))}
        </nav>
        <Link href="/add" className="side-add">
          <Plus size={17} /> Add some wax <ArrowUpRight size={16} />
        </Link>
        <div className="sidebar-bottom">
          <div className="side-note">
            <Disc3 size={26} />
            <p>
              Good records.
              <br />
              <strong>Better company.</strong>
            </p>
          </div>
          <Link
            href="/settings"
            className={
              "settings-link " + (path === "/settings" ? "active" : "")
            }
          >
            <Settings2 size={18} /> Settings & invites
          </Link>
          <div className="collector-badge">
            <span>
              {data?.collector.name?.slice(0, 1).toUpperCase() || "W"}
            </span>
            <div>
              <strong>{data?.collector.name || "Your collection"}</strong>
              <small>INDEPENDENT COLLECTOR</small>
            </div>
          </div>
        </div>
      </aside>
      <div className="main-shell">
        <header className="topbar">
          <form
            className="global-search"
            onSubmit={(e) => {
              e.preventDefault();
              router.push("/collection");
            }}
          >
            <Search size={18} />
            <input
              aria-label="Search your collection"
              placeholder="Dig through your collection…"
              value={search}
              onChange={(e) => {
                setSearch(e.target.value);
                if (path !== "/collection") router.push("/collection");
              }}
            />
            <kbd>↵</kbd>
          </form>
          <span className="topbar-note">
            <i /> A little soul. A lot of wax.
          </span>
          <Link
            href="/settings"
            className="mobile-avatar"
            aria-label="Your settings"
          >
            {data?.collector.name?.slice(0, 1) || "W"}
          </Link>
        </header>
        <main id="main-content" className="main-content">
          {error && (
            <div className="load-error">
              <ErrorNotice>{error}</ErrorNotice>
              <Button variant="secondary" onClick={() => void refresh()}>
                <RefreshCw size={16} /> Try again
              </Button>
            </div>
          )}
          {loading ? (
            <div className="collection-loading" role="status">
              <Disc3 className="spin" size={44} />
              <span>Pulling your records…</span>
            </div>
          ) : data ? (
            children
          ) : !error ? (
            <div className="collection-loading">Opening your collection…</div>
          ) : null}
        </main>
        <footer className="app-footer">
          <span>WAX® · KEEP THE GOOD STUFF.</span>
          <span>
            Made for the long play. <Disc3 size={13} />
          </span>
        </footer>
      </div>
      <nav className="mobile-nav" aria-label="Mobile navigation">
        {navigation.map(({ href, short, icon: Icon }) => (
          <Link
            key={href}
            href={href}
            className={path.startsWith(href) ? "active" : ""}
          >
            <Icon size={19} />
            <span>{short}</span>
          </Link>
        ))}
        <Link
          href="/add"
          className={"mobile-add " + (path === "/add" ? "active" : "")}
        >
          <Plus size={21} />
          <span>Add wax</span>
        </Link>
      </nav>
    </div>
  );
}
