"use client";

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type FormEvent,
} from "react";
import {
  ArrowUpRight,
  BookHeart,
  Check,
  LockKeyhole,
  MapPin,
  Plus,
  RefreshCw,
  Star,
  Trash2,
  Users,
} from "lucide-react";
import {
  SHOP_AVATARS,
  type SavedShop,
  type SaveShopInput,
  type ShopEmployee,
  type ShopNotebook,
  type ShopSnapshot,
} from "@/lib/shop-notebook";
import { api, ApiError } from "./library-provider";
import { ShopAvatar } from "./shop-avatar";
import { Button, ErrorNotice, Field, Modal } from "./ui";

const endpoint = "/api/shops/notebook";
const timeout = () => AbortSignal.timeout(20_000);

function errorMessage(error: unknown) {
  if (error instanceof DOMException && error.name === "TimeoutError") {
    return "That took too long. Your changes are still here. Try again when your connection is back.";
  }
  return error instanceof Error
    ? error.message
    : "Your shop notes couldn’t be saved. Please try again.";
}

export function useShopNotebook() {
  const [shops, setShops] = useState<SavedShop[]>([]);
  const [status, setStatus] = useState<"loading" | "ready" | "error">(
    "loading",
  );
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const requestVersion = useRef(0);

  const load = useCallback(async () => {
    const version = ++requestVersion.current;
    setStatus("loading");
    setError("");
    try {
      const result = await api<ShopNotebook>(endpoint, { signal: timeout() });
      if (version === requestVersion.current) {
        setShops(result.shops);
        setStatus("ready");
      }
      return result.shops;
    } catch (err) {
      if (version === requestVersion.current) {
        setError(errorMessage(err));
        setStatus("error");
      }
      throw err;
    }
  }, []);

  useEffect(() => {
    void load().catch(() => {});
    return () => {
      requestVersion.current++;
    };
  }, [load]);

  async function save(input: SaveShopInput) {
    const entry = await api<SavedShop>(endpoint, {
      method: "PUT",
      signal: timeout(),
      body: JSON.stringify(input),
    });
    requestVersion.current++;
    setShops((current) => [
      entry,
      ...current.filter((item) => item.shop.id !== entry.shop.id),
    ]);
    setStatus("ready");
    setError("");
    setMessage(`${entry.shop.name} saved to your shops.`);
    return entry;
  }

  async function remove(entry: SavedShop) {
    await api<{ ok: true }>(endpoint, {
      method: "DELETE",
      signal: timeout(),
      body: JSON.stringify({ shopId: entry.shop.id, version: entry.version }),
    });
    requestVersion.current++;
    setShops((current) =>
      current.filter((item) => item.shop.id !== entry.shop.id),
    );
    setStatus("ready");
    setError("");
    setMessage(`${entry.shop.name} removed from your saved shops.`);
  }

  return { shops, status, error, message, load, save, remove };
}

export function ShopRating({ rating }: { rating: number | null }) {
  return rating ? (
    <span
      className="shop-rating"
      role="img"
      aria-label={`Your rating: ${rating} out of 5 stars`}
    >
      {[1, 2, 3, 4, 5].map((star) => (
        <Star
          key={star}
          size={13}
          aria-hidden="true"
          fill={star <= rating ? "currentColor" : "none"}
        />
      ))}
      <span aria-hidden="true">{rating}/5</span>
    </span>
  ) : (
    <span className="shop-unrated">Not rated yet</span>
  );
}

export function ShopPeople({ employees }: { employees: ShopEmployee[] }) {
  if (!employees.length) return null;
  return (
    <div className="shop-people-preview">
      <div className="shop-avatar-stack" aria-hidden="true">
        {employees.slice(0, 4).map((person) => (
          <ShopAvatar key={person.id} avatar={person.avatar} />
        ))}
        {employees.length > 4 && <span>+{employees.length - 4}</span>}
      </div>
      <p>
        {employees
          .slice(0, 2)
          .map((person) => person.name)
          .join(" & ")}
        {employees.length > 2 ? ` + ${employees.length - 2} more` : ""}
      </p>
    </div>
  );
}

export function SavedShops({
  notebook,
  onEdit,
}: {
  notebook: ReturnType<typeof useShopNotebook>;
  onEdit: (shop: ShopSnapshot, saved: SavedShop | null) => void;
}) {
  return (
    <section
      className="shop-notebook"
      aria-labelledby="shop-notebook-heading"
      aria-busy={notebook.status === "loading"}
    >
      <div className="shop-notebook-heading">
        <div>
          <span className="eyebrow">GOOD PLACES. GOOD PEOPLE.</span>
          <h2 id="shop-notebook-heading">
            Your shops<span>.</span>
          </h2>
          <p>Your ratings and people. Only you can see these.</p>
        </div>
        <span className="shop-private-badge">
          <LockKeyhole size={12} /> JUST FOR YOU
        </span>
      </div>
      {notebook.status === "loading" ? (
        <p className="shop-notebook-loading" role="status">
          <RefreshCw size={16} className="spin" /> Opening your shop notebook…
        </p>
      ) : notebook.status === "error" ? (
        <div className="shop-notebook-error">
          <ErrorNotice>
            We couldn’t load your saved shops. {notebook.error}
          </ErrorNotice>
          <Button
            variant="secondary"
            onClick={() => void notebook.load().catch(() => {})}
          >
            <RefreshCw size={14} /> Retry saved shops
          </Button>
        </div>
      ) : notebook.shops.length ? (
        <ul className="saved-shops-grid">
          {notebook.shops.map((entry) => (
            <li className="saved-shop-card" key={entry.shop.id}>
              <ShopRating rating={entry.rating} />
              <h3>{entry.shop.name}</h3>
              {entry.shop.address && (
                <p className="saved-shop-address">
                  <MapPin size={13} /> {entry.shop.address}
                </p>
              )}
              <ShopPeople employees={entry.employees} />
              {entry.employees.length ? (
                <p className="saved-shop-roles">
                  {entry.employees
                    .slice(0, 2)
                    .map(
                      (person) =>
                        `${person.name}${person.role ? ` · ${person.role}` : ""}`,
                    )
                    .join(" / ")}
                </p>
              ) : (
                <p className="saved-shop-no-people">
                  Every good shop has good people. Add yours.
                </p>
              )}
              <div className="saved-shop-actions">
                <Button
                  variant="secondary"
                  disabled={notebook.status !== "ready"}
                  onClick={() => onEdit(entry.shop, entry)}
                >
                  Edit shop notes <ArrowUpRight size={14} />
                </Button>
                <a
                  className="shop-text-link"
                  href={`https://www.google.com/maps/dir/?api=1&destination=${encodeURIComponent(`${entry.shop.latitude},${entry.shop.longitude}`)}`}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  Directions <ArrowUpRight size={13} />
                </a>
              </div>
            </li>
          ))}
        </ul>
      ) : (
        <div className="shop-notebook-empty">
          <BookHeart size={27} strokeWidth={1.4} />
          <p>
            <strong>Remember the spots worth coming back to.</strong>Find a shop
            below, leave your rating, and put a name to that friendly face
            behind the counter.
          </p>
        </div>
      )}
      <p className="sr-only" role="status">
        {notebook.message}
      </p>
    </section>
  );
}

export function ShopNotebookEditor({
  shop,
  saved,
  notebook,
  onClose,
}: {
  shop: ShopSnapshot;
  saved: SavedShop | null;
  notebook: ReturnType<typeof useShopNotebook>;
  onClose: () => void;
}) {
  const [baseline, setBaseline] = useState(saved);
  const [rating, setRating] = useState<number | null>(saved?.rating ?? null);
  const [employees, setEmployees] = useState<ShopEmployee[]>(
    () => saved?.employees.map((person) => ({ ...person })) ?? [],
  );
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [conflict, setConflict] = useState(false);
  const [confirmation, setConfirmation] = useState<
    "" | "discard" | "remove" | "reload"
  >("");
  const newPerson = useRef<string | null>(null);
  const dirty =
    rating !== (baseline?.rating ?? null) ||
    JSON.stringify(employees) !== JSON.stringify(baseline?.employees ?? []);

  useEffect(() => {
    if (!dirty) return;
    const warn = (event: BeforeUnloadEvent) => {
      event.preventDefault();
      event.returnValue = "";
    };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [dirty]);

  useEffect(() => {
    if (newPerson.current) {
      document.getElementById(`shop-person-name-${newPerson.current}`)?.focus();
      newPerson.current = null;
    }
  }, [employees]);

  function close() {
    if (busy) return;
    if (dirty) setConfirmation("discard");
    else onClose();
  }

  function addPerson() {
    const id = crypto.randomUUID();
    newPerson.current = id;
    setEmployees((current) => [
      ...current,
      {
        id,
        name: "",
        role: "",
        avatar: SHOP_AVATARS[current.length % SHOP_AVATARS.length].id,
      },
    ]);
    setConfirmation("");
  }

  function updatePerson(id: string, patch: Partial<ShopEmployee>) {
    setEmployees((current) =>
      current.map((person) =>
        person.id === id ? { ...person, ...patch } : person,
      ),
    );
  }

  function handleError(err: unknown) {
    if (err instanceof ApiError && (err.status === 409 || err.status === 404)) {
      setConflict(true);
      setError(
        "This shop changed in another tab. Your unsaved changes are still here. Load the latest notes before editing again.",
      );
    } else setError(errorMessage(err));
  }

  async function save(event: FormEvent) {
    event.preventDefault();
    if (busy || conflict || confirmation) return;
    const blankName = employees.find((person) => !person.name.trim());
    if (blankName) {
      setError("Give each person a name before saving.");
      document.getElementById(`shop-person-name-${blankName.id}`)?.focus();
      return;
    }
    const blankRole = employees.find((person) => !person.role.trim());
    if (blankRole) {
      setError("Add a role for each person before saving.");
      document.getElementById(`shop-person-role-${blankRole.id}`)?.focus();
      return;
    }
    setBusy(true);
    setError("");
    setConfirmation("");
    try {
      await notebook.save({
        shop: baseline?.shop ?? shop,
        rating,
        employees: employees.map((person) => ({
          ...person,
          name: person.name.trim(),
          role: person.role.trim(),
        })),
        version: baseline?.version ?? null,
      });
      onClose();
    } catch (err) {
      handleError(err);
    } finally {
      setBusy(false);
    }
  }

  async function remove() {
    if (!baseline || busy) return;
    setBusy(true);
    setError("");
    try {
      await notebook.remove(baseline);
      onClose();
    } catch (err) {
      handleError(err);
      setConfirmation("");
    } finally {
      setBusy(false);
    }
  }

  async function reload() {
    setBusy(true);
    setError("");
    try {
      const entries = await notebook.load();
      const latest = entries.find((entry) => entry.shop.id === shop.id) ?? null;
      setBaseline(latest);
      setRating(latest?.rating ?? null);
      setEmployees(latest?.employees.map((person) => ({ ...person })) ?? []);
      setConflict(false);
      setConfirmation("");
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <Modal
      open
      onOpenChange={(open) => {
        if (!open) close();
      }}
      title={shop.name}
      description="Your ratings and people. Only you can see these."
    >
      <form className="shop-notebook-editor" onSubmit={save} aria-busy={busy}>
        <fieldset className="shop-editor-controls" disabled={busy}>
          <fieldset className="shop-rating-picker">
            <legend>Your rating</legend>
            <div className="shop-stars">
              {[1, 2, 3, 4, 5].map((value) => (
                <label
                  key={value}
                  className={value <= (rating ?? 0) ? "is-filled" : ""}
                >
                  <input
                    type="radio"
                    name="shop-rating"
                    value={value}
                    checked={rating === value}
                    onChange={() => setRating(value)}
                    aria-label={`${value} ${value === 1 ? "star" : "stars"}`}
                  />
                  <Star size={32} aria-hidden="true" />
                </label>
              ))}
              <span className="shop-rating-word" aria-hidden="true">
                {rating
                  ? [
                      "",
                      "Not my spot",
                      "A little offbeat",
                      "A good dig",
                      "Heavy rotation",
                      "All-time favorite",
                    ][rating]
                  : "How was the dig?"}
              </span>
            </div>
            <button
              className="shop-clear-rating"
              type="button"
              disabled={rating === null}
              onClick={() => setRating(null)}
            >
              Clear rating
            </button>
          </fieldset>
          <div className="shop-people-heading">
            <div>
              <h3>The people behind the counter</h3>
              <p>Names, roles, and a little personality.</p>
            </div>
            <Users size={21} />
          </div>
          <div className="shop-people-editor">
            {employees.map((person, index) => (
              <fieldset className="shop-person-editor" key={person.id}>
                <legend>Person {index + 1}</legend>
                <div className="shop-person-top">
                  <ShopAvatar
                    avatar={person.avatar}
                    name={person.name || `Person ${index + 1}`}
                  />
                  <div>
                    <strong>{person.name || "A familiar face"}</strong>
                    <span>
                      {person.role ||
                        "The person with the good recommendations."}
                    </span>
                  </div>
                  <button
                    className="icon-button"
                    type="button"
                    aria-label={`Remove person ${index + 1}`}
                    onClick={() =>
                      setEmployees((current) =>
                        current.filter((item) => item.id !== person.id),
                      )
                    }
                  >
                    <Trash2 size={17} />
                  </button>
                </div>
                <div className="shop-person-fields">
                  <Field label="Name">
                    <input
                      id={`shop-person-name-${person.id}`}
                      value={person.name}
                      onChange={(event) =>
                        updatePerson(person.id, { name: event.target.value })
                      }
                      required
                      maxLength={80}
                      autoComplete="off"
                      placeholder="e.g. Sam"
                    />
                  </Field>
                  <Field label="Role">
                    <input
                      id={`shop-person-role-${person.id}`}
                      required
                      value={person.role}
                      onChange={(event) =>
                        updatePerson(person.id, { role: event.target.value })
                      }
                      maxLength={80}
                      autoComplete="off"
                      placeholder="e.g. Owner, jazz expert"
                    />
                  </Field>
                </div>
                <fieldset className="shop-avatar-picker">
                  <legend>Pick their character</legend>
                  <div>
                    {SHOP_AVATARS.map((avatar) => (
                      <label key={avatar.id}>
                        <input
                          type="radio"
                          name={`avatar-${person.id}`}
                          value={avatar.id}
                          checked={person.avatar === avatar.id}
                          onChange={() =>
                            updatePerson(person.id, { avatar: avatar.id })
                          }
                          aria-label={avatar.label}
                        />
                        <span className="shop-avatar-choice">
                          <ShopAvatar avatar={avatar.id} />
                          <span>{avatar.label}</span>
                          {person.avatar === avatar.id && (
                            <Check
                              size={13}
                              className="shop-avatar-check"
                              aria-hidden="true"
                            />
                          )}
                        </span>
                      </label>
                    ))}
                  </div>
                </fieldset>
              </fieldset>
            ))}
          </div>
          <Button
            className="shop-add-person"
            variant="secondary"
            type="button"
            onClick={addPerson}
            disabled={employees.length >= 30}
          >
            <Plus size={16} /> Add a person
          </Button>
          {employees.length >= 30 && (
            <p className="field-hint">
              You can keep up to 30 people at each shop.
            </p>
          )}
        </fieldset>
        <ErrorNotice>{error}</ErrorNotice>
        {conflict && (
          <Button
            variant="secondary"
            type="button"
            disabled={busy}
            onClick={() => setConfirmation("reload")}
          >
            <RefreshCw size={14} /> Load latest notes
          </Button>
        )}
        {confirmation ? (
          <div className="shop-editor-confirmation" role="alert">
            <p>
              {confirmation === "discard"
                ? "Leave without saving your changes?"
                : confirmation === "remove"
                  ? "Remove this shop, your rating, and all its people from your notebook?"
                  : "Replace your unsaved changes with the latest saved notes?"}
            </p>
            <div>
              <Button
                type="button"
                variant="secondary"
                disabled={busy}
                onClick={() => setConfirmation("")}
              >
                {confirmation === "discard" ? "Keep editing" : "Cancel"}
              </Button>
              <Button
                type="button"
                variant="danger"
                busy={busy}
                onClick={() => {
                  if (confirmation === "discard") onClose();
                  else if (confirmation === "remove") void remove();
                  else void reload();
                }}
              >
                {confirmation === "discard"
                  ? "Discard changes"
                  : confirmation === "remove"
                    ? "Remove shop"
                    : "Replace with latest"}
              </Button>
            </div>
          </div>
        ) : (
          <div className="shop-editor-footer">
            {baseline && (
              <button
                className="shop-remove-saved"
                type="button"
                disabled={busy}
                onClick={() => setConfirmation("remove")}
              >
                <Trash2 size={13} /> Remove saved shop
              </button>
            )}
            <div>
              <Button
                type="button"
                variant="secondary"
                disabled={busy}
                onClick={close}
              >
                Cancel
              </Button>
              <Button type="submit" busy={busy} disabled={conflict}>
                Save changes <Check size={16} />
              </Button>
            </div>
          </div>
        )}
      </form>
    </Modal>
  );
}
