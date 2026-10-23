"use client";
import {
  useEffect,
  useState,
  type ButtonHTMLAttributes,
  type ReactNode,
} from "react";
import * as Dialog from "@radix-ui/react-dialog";
import { X, Disc3, LoaderCircle, ArrowUpRight } from "lucide-react";
import type { CatalogCandidate } from "@/lib/contracts";

export function Button({
  children,
  variant = "primary",
  className = "",
  busy,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary" | "quiet" | "danger";
  busy?: boolean;
}) {
  return (
    <button
      {...props}
      disabled={props.disabled || busy}
      className={"button button-" + variant + " " + className}
    >
      {busy && <LoaderCircle size={16} className="spin" />}
      {children}
    </button>
  );
}
export function Field({
  label,
  hint,
  children,
  className = "",
}: {
  label: string;
  hint?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <label className={"field " + className}>
      <span className="field-label">{label}</span>
      {children}
      {hint && <span className="field-hint">{hint}</span>}
    </label>
  );
}
export function ErrorNotice({ children }: { children?: ReactNode }) {
  return children ? (
    <div className="notice notice-error" role="alert">
      {children}
    </div>
  ) : null;
}
export function SuccessNotice({ children }: { children?: ReactNode }) {
  return children ? (
    <div className="notice notice-success" role="status">
      {children}
    </div>
  ) : null;
}
export function Modal({
  open,
  onOpenChange,
  title,
  description,
  children,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  description?: string;
  children: ReactNode;
}) {
  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="modal-overlay" />
        <Dialog.Content className="modal-content">
          <div className="modal-top">
            <Dialog.Title>{title}</Dialog.Title>
            <Dialog.Close className="icon-button" aria-label="Close dialog">
              <X size={20} />
            </Dialog.Close>
          </div>
          <Dialog.Description
            className={description ? "modal-description" : "sr-only"}
          >
            {description || title}
          </Dialog.Description>
          {children}
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
export function Vinyl({
  className = "",
  label = "WAX",
}: {
  className?: string;
  label?: string;
}) {
  return (
    <div className={"vinyl-art " + className} aria-hidden="true">
      <div className="vinyl-groove groove-one" />
      <div className="vinyl-groove groove-two" />
      <div className="vinyl-groove groove-three" />
      <div className="vinyl-label">
        <span>{label}</span>
        <i />
      </div>
    </div>
  );
}
export function Cover({
  record,
  className = "",
}: {
  record: Pick<CatalogCandidate, "coverUrl" | "title" | "artist">;
  className?: string;
}) {
  const [failedSource, setFailedSource] = useState<string | null>(null);
  const [retry, setRetry] = useState({ source: "", attempt: 0 });
  const source = record.coverUrl;
  const attempt = retry.source === source ? retry.attempt : 0;
  const imageKey = `${source}:${attempt}`;
  useEffect(() => {
    if (!source || failedSource !== imageKey) return;
    const recover = () => {
      setRetry({ source, attempt: attempt + 1 });
      setFailedSource(null);
    };
    const onVisible = () => {
      if (document.visibilityState === "visible") recover();
    };
    // One automatic retry; subsequent retries wait for a reconnect or return.
    // Covers also live inside links and labels, so no nested retry controls.
    const timer = attempt === 0 ? window.setTimeout(recover, 2000) : undefined;
    window.addEventListener("online", recover);
    window.addEventListener("focus", recover);
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      window.clearTimeout(timer);
      window.removeEventListener("online", recover);
      window.removeEventListener("focus", recover);
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, [source, failedSource, imageKey, attempt]);
  return source && failedSource !== imageKey ? (
    <img
      key={imageKey}
      className={"cover " + className}
      src={source}
      alt={record.artist + " — " + record.title + " sleeve"}
      loading="lazy"
      onError={() => setFailedSource(imageKey)}
    />
  ) : (
    <div className={"cover-fallback " + className}>
      <Disc3 />
      <strong>{record.title || "The good stuff"}</strong>
      <span>{record.artist || "Waiting to be discovered"}</span>
    </div>
  );
}
export function SourceCredit({ sourceUrl }: { sourceUrl: string | null }) {
  return sourceUrl ? (
    <a
      className="source-credit"
      href={sourceUrl}
      target="_blank"
      rel="noreferrer"
    >
      Data provided by Discogs <ArrowUpRight size={12} />
    </a>
  ) : null;
}
export function PageHeader({
  eyebrow,
  title,
  description,
  action,
}: {
  eyebrow: string;
  title: string;
  description?: string;
  action?: ReactNode;
}) {
  return (
    <header className="page-header">
      <div>
        <span className="eyebrow">{eyebrow}</span>
        <h1>
          {title}
          <span className="heading-dot">.</span>
        </h1>
        {description && <p className="page-description">{description}</p>}
      </div>
      {action && <div className="page-action">{action}</div>}
    </header>
  );
}
export const dateLabel = (value: string | null) =>
  value
    ? new Date(
        /^\d{4}-\d{2}-\d{2}$/.test(value) ? value + "T12:00:00" : value,
      ).toLocaleDateString(undefined, {
        month: "short",
        day: "numeric",
        year: "numeric",
      })
    : "—";
