"use client";
import type { CatalogCandidate, CopyFields } from "@/lib/contracts";
import { CONDITIONS } from "@/lib/contracts";
import { Field } from "./ui";

export function CandidateFields({
  value,
  onChange,
}: {
  value: CatalogCandidate;
  onChange: (value: CatalogCandidate) => void;
}) {
  const update = (patch: Partial<CatalogCandidate>) =>
    onChange({ ...value, ...patch });
  return (
    <div className="form-grid">
      <Field label="Artist">
        <input
          required
          maxLength={300}
          value={value.artist}
          onChange={(e) => update({ artist: e.target.value })}
          placeholder="Who’s on the record?"
        />
      </Field>
      <Field label="Record title">
        <input
          required
          maxLength={300}
          value={value.title}
          onChange={(e) => update({ title: e.target.value })}
          placeholder="The name on the sleeve"
        />
      </Field>
      <Field label="Release year">
        <input
          type="number"
          min="1877"
          max={new Date().getFullYear() + 2}
          value={value.year || ""}
          onChange={(e) =>
            update({ year: e.target.value ? Number(e.target.value) : null })
          }
          placeholder="Unknown is fine"
        />
      </Field>
      <Field label="Genres" hint="Separate genres with commas.">
        <input
          value={value.genres.join(", ")}
          onChange={(e) =>
            update({ genres: e.target.value.split(",").map((s) => s.trim()) })
          }
          placeholder="Soul, Funk, Jazz…"
        />
      </Field>
      <Field label="Label">
        <input
          value={value.label}
          onChange={(e) => update({ label: e.target.value })}
          placeholder="Record label"
        />
      </Field>
      <Field label="Catalog number">
        <input
          value={value.catalogNumber}
          onChange={(e) => update({ catalogNumber: e.target.value })}
          placeholder="Check the spine or label"
        />
      </Field>
      <Field label="Country">
        <input
          value={value.country}
          onChange={(e) => update({ country: e.target.value })}
          placeholder="Country of this pressing"
        />
      </Field>
      <Field label="Format">
        <input
          value={value.format}
          onChange={(e) => update({ format: e.target.value })}
          placeholder="Vinyl, LP, 33⅓ RPM"
        />
      </Field>
    </div>
  );
}
export function CopyForm({
  value,
  onChange,
}: {
  value: CopyFields;
  onChange: (value: CopyFields) => void;
}) {
  const update = (patch: Partial<CopyFields>) =>
    onChange({ ...value, ...patch });
  return (
    <>
      <div className="form-grid">
        <Field label="Media condition">
          <select
            value={value.mediaCondition}
            onChange={(e) =>
              update({
                mediaCondition: e.target.value as CopyFields["mediaCondition"],
              })
            }
          >
            {CONDITIONS.map((c) => (
              <option key={c}>{c}</option>
            ))}
          </select>
        </Field>
        <Field label="Sleeve condition">
          <select
            value={value.sleeveCondition}
            onChange={(e) =>
              update({
                sleeveCondition: e.target
                  .value as CopyFields["sleeveCondition"],
              })
            }
          >
            {CONDITIONS.map((c) => (
              <option key={c}>{c}</option>
            ))}
          </select>
        </Field>
        <Field label="Shelf / location">
          <input
            value={value.location}
            onChange={(e) => update({ location: e.target.value })}
            placeholder="Living room · Shelf A"
            maxLength={200}
          />
        </Field>
        <Field label="Your rating">
          <select
            value={value.rating}
            onChange={(e) => update({ rating: Number(e.target.value) })}
          >
            <option value={0}>Not rated yet</option>
            {[1, 2, 3, 4, 5].map((n) => (
              <option key={n} value={n}>
                {"★".repeat(n)} — {n} / 5
              </option>
            ))}
          </select>
        </Field>
        <Field label="Purchase price">
          <input
            type="number"
            min="0"
            step="0.01"
            value={value.purchasePrice || ""}
            onChange={(e) => update({ purchasePrice: e.target.value || null })}
            placeholder="Optional"
          />
        </Field>
        <Field label="Currency">
          <input
            value={value.currency}
            onChange={(e) => update({ currency: e.target.value.toUpperCase() })}
            maxLength={3}
            minLength={3}
            placeholder="USD"
          />
        </Field>
        <Field label="Acquired on">
          <input
            type="date"
            value={value.acquiredAt?.slice(0, 10) || ""}
            onChange={(e) => update({ acquiredAt: e.target.value || null })}
          />
        </Field>
        <Field label="What’s the plan?">
          <select
            value={value.intent}
            onChange={(e) =>
              update({ intent: e.target.value as CopyFields["intent"] })
            }
          >
            <option value="keep">A keeper</option>
            <option value="trade">Open to a trade</option>
            <option value="sell">Future shop stock</option>
          </select>
        </Field>
      </div>
      <Field
        label="Your liner notes"
        hint="Private to you, even when this record is in a shared crate."
      >
        <textarea
          rows={4}
          value={value.notes}
          onChange={(e) => update({ notes: e.target.value })}
          placeholder="Where you found it. The track that gets you. That tiny sleeve crease…"
          maxLength={10000}
        />
      </Field>
      <label className="checkbox-row">
        <input
          type="checkbox"
          checked={value.pressingConfirmed}
          onChange={(e) => update({ pressingConfirmed: e.target.checked })}
        />
        <span>
          <strong>I’ve checked this pressing.</strong>
          <small>
            Match the label, catalog number, country, and format. Leave
            unchecked if you’re still figuring it out.
          </small>
        </span>
      </label>
    </>
  );
}
