from __future__ import annotations

import base64
import datetime as dt
import io
import json
import os
import re
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd
import qrcode
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas


APP_DIR = Path(__file__).parent
ASSETS_DIR = APP_DIR / "assets"


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def _now_naive() -> dt.datetime:
    return dt.datetime.now().replace(tzinfo=None)


def _iso_now() -> str:
    return _now_naive().replace(microsecond=0).isoformat(sep=" ")


def _parse_csv_list(s: str) -> List[str]:
    if not s:
        return []
    items = [x.strip() for x in s.split(",")]
    items = [x for x in items if x]
    # de-dupe while preserving order
    seen = set()
    out = []
    for x in items:
        key = x.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(x)
    return out


def _parse_bulk_line(line: str) -> Optional[Tuple[str, str]]:
    if not line:
        return None
    parts = re.split(r"\s*[–—-]\s*", line, maxsplit=1)
    if len(parts) < 2:
        return None
    artist, album = [p.strip() for p in parts]
    if not artist or not album:
        return None
    return artist, album


def _stars(rating: int) -> str:
    r = max(0, min(5, int(rating)))
    return "★" * r + "☆" * (5 - r)


def _slug(s: str, max_len: int = 60) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s[:max_len] or "file"


def _sanitize_filename(name: str) -> str:
    name = os.path.basename(name or "file")
    name = re.sub(r"[^a-zA-Z0-9._ -]+", "_", name).strip()
    if not name:
        name = "file"
    return name


def _resolve_data_root() -> Path:
    override = os.environ.get("VINYL_VAULT_DATA_DIR")
    candidates: List[Path] = []
    if override:
        candidates.append(Path(override))
    candidates.extend([APP_DIR / "data", Path.home() / ".vinyl_vault"])
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            test = candidate / ".write_test"
            test.write_text("ok", encoding="utf-8")
            test.unlink(missing_ok=True)
            return candidate
        except Exception:
            continue
    fallback = Path(tempfile.gettempdir()) / "vinyl_vault"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


DATA_DIR = _resolve_data_root()
DB_PATH = DATA_DIR / "vinyl_vault.json"
MEDIA_DIR = DATA_DIR / "media"
MEDIA_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_OPENAI_MODEL = "gpt-5.2"


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def _atomic_write_text(path: Path, text: str) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"))


def _read_json(path: Path) -> Optional[dict]:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json(path: Path, obj: dict) -> None:
    _atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2))


def _parse_json_payload(text: str) -> dict:
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def _load_library() -> dict:
    data = _read_json(DB_PATH)
    if not data or "records" not in data:
        return {"records": []}
    return data


def _save_library(data: dict) -> None:
    _write_json(DB_PATH, data)


def _add_record(data: dict, record: dict) -> None:
    data.setdefault("records", []).append(record)
    _save_library(data)


def _remove_records(data: dict, record_ids: List[str]) -> int:
    existing = data.get("records", [])
    record_ids = set(record_ids)
    remaining = [r for r in existing if r.get("id") not in record_ids]
    removed = len(existing) - len(remaining)
    data["records"] = remaining
    if removed:
        _save_library(data)
    return removed


def _records_dataframe(records: List[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(
            columns=["artist", "album", "year", "genre", "rating", "added_at", "notes"]
        )
    df = pd.DataFrame.from_records(records)
    df["rating"] = df["rating"].fillna(0).astype(int)
    return df.sort_values(by=["added_at", "artist", "album"], ascending=False)


def _build_label_pdf(record: dict) -> bytes:
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    pdf.setTitle("Vinyl Vault Label")
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(72, height - 72, "Vinyl Vault Label")
    pdf.setFont("Helvetica", 12)
    lines = [
        f"Artist: {record.get('artist', '')}",
        f"Album: {record.get('album', '')}",
        f"Year: {record.get('year', '')}",
        f"Genre: {record.get('genre', '')}",
        f"Rating: {_stars(record.get('rating', 0))}",
        f"Notes: {record.get('notes', '')}",
    ]
    y = height - 120
    for line in lines:
        pdf.drawString(72, y, line)
        y -= 18
    pdf.setStrokeColor(colors.HexColor("#4C6EF5"))
    pdf.rect(60, height - 180, width - 120, 120, stroke=1, fill=0)
    pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def _record_summary(record: dict) -> str:
    year = record.get("year")
    year_text = f" ({year})" if year else ""
    return f"{record.get('artist', 'Unknown')} — {record.get('album', 'Untitled')}{year_text}"


def _get_openai_key() -> str:
    if "openai_api_key" in st.secrets:
        return st.secrets["openai_api_key"]
    return os.getenv("OPENAI_API_KEY", "")


def _build_vault_context(records: List[dict]) -> str:
    if not records:
        return "The Vinyl Vault is currently empty."
    payload = [
        {
            "artist": record.get("artist"),
            "album": record.get("album"),
            "year": record.get("year"),
            "genre": record.get("genre"),
            "rating": record.get("rating"),
            "notes": record.get("notes"),
            "added_at": record.get("added_at"),
        }
        for record in records
    ]
    return json.dumps(payload, ensure_ascii=False)


def _build_vault_digest(records: List[dict]) -> str:
    if not records:
        return "No records yet."
    lines = []
    for record in records:
        artist = record.get("artist", "Unknown")
        album = record.get("album", "Untitled")
        year = record.get("year")
        genre = record.get("genre") or "Unknown genre"
        rating = record.get("rating") or 0
        notes = record.get("notes") or ""
        year_text = f" ({year})" if year else ""
        note_text = f" — {notes}" if notes else ""
        lines.append(f"- {artist} — {album}{year_text} · {genre} · {_stars(rating)}{note_text}")
    return "\n".join(lines)


def _build_virtuoso_prompt(records: List[dict]) -> str:
    context = _build_vault_context(records)
    digest = _build_vault_digest(records)
    return (
        "You are the Vinyl Virtuoso, an expert guide to a user's vinyl collection. "
        "Answer with friendly, confident detail while staying grounded in the vault data. "
        "If asked about records that are not present, say so and suggest related entries "
        "that are present. When the user wants to update the vault, confirm what you "
        "changed and keep the tone calm, immersive, and relaxing.\n"
        "Response guidelines:\n"
        "- Lead with a direct answer, then add supporting detail or a short list.\n"
        "- Offer 1-3 recommendations when appropriate, each with a clear reason tied to the data.\n"
        "- If the request is ambiguous, make a sensible assumption and state it briefly.\n"
        "- Keep responses concise and avoid inventing details not in the vault.\n"
        "- Use soothing, confident language that makes the user feel relaxed and in control.\n"
        "Vault digest:\n"
        f"{digest}\n"
        "Vault data (JSON):\n"
        f"{context}"
    )


def _build_wax_wizard_prompt(records: List[dict]) -> str:
    context = _build_vault_context(records)
    digest = _build_vault_digest(records)
    return (
        "You are Wax Wizard, the user's Vinyl Virtuoso with full control of the Vinyl Vault. "
        "Interpret requests to add, update, or delete records. "
        "Return JSON that includes a calming assistant reply and a list of actions.\n"
        "Rules:\n"
        "- Only include actions when the user clearly wants a change.\n"
        "- Use action types: add, update, delete.\n"
        "- For add: include artist, album, year (or null), genre, rating (0-5), notes.\n"
        "- For update/delete: include a match object with artist, album, year, or id.\n"
        "- Keep assistant_reply immersive and reassuring.\n"
        "Vault digest:\n"
        f"{digest}\n"
        "Vault data (JSON):\n"
        f"{context}"
    )


def _normalize_text(value: Optional[str]) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def _match_records(records: List[dict], match: Dict[str, Any]) -> List[dict]:
    if not match:
        return []
    match_id = _normalize_text(match.get("id"))
    match_artist = _normalize_text(match.get("artist"))
    match_album = _normalize_text(match.get("album"))
    match_year = match.get("year")

    matched = []
    for record in records:
        if match_id and _normalize_text(record.get("id")) != match_id:
            continue
        if match_artist and _normalize_text(record.get("artist")) != match_artist:
            continue
        if match_album and _normalize_text(record.get("album")) != match_album:
            continue
        if match_year is not None and record.get("year") != match_year:
            continue
        matched.append(record)
    return matched


def _apply_vault_actions(library: dict, actions: List[dict]) -> Dict[str, List[str]]:
    results: Dict[str, List[str]] = {"added": [], "updated": [], "deleted": [], "skipped": []}
    records = library.get("records", [])
    for action in actions:
        action_type = (action.get("action") or "").lower()
        if action_type == "add":
            record = action.get("record") or {}
            new_record = {
                "id": uuid.uuid4().hex,
                "artist": (record.get("artist") or "Unknown Artist").strip(),
                "album": (record.get("album") or "Untitled Album").strip(),
                "year": record.get("year"),
                "genre": (record.get("genre") or "").strip(),
                "rating": _safe_int(record.get("rating"), default=0),
                "notes": (record.get("notes") or "").strip(),
                "added_at": _iso_now(),
            }
            records.append(new_record)
            results["added"].append(_record_summary(new_record))
        elif action_type in {"update", "delete"}:
            match = action.get("match") or {}
            matched = _match_records(records, match)
            if not matched:
                results["skipped"].append(f"{action_type}: no match for {match}")
                continue
            if action_type == "delete":
                ids = {r.get("id") for r in matched}
                library["records"] = [r for r in records if r.get("id") not in ids]
                for record in matched:
                    results["deleted"].append(_record_summary(record))
                records = library["records"]
            else:
                updates = action.get("updates") or {}
                for record in matched:
                    if "artist" in updates:
                        record["artist"] = (updates.get("artist") or record.get("artist") or "").strip()
                    if "album" in updates:
                        record["album"] = (updates.get("album") or record.get("album") or "").strip()
                    if "year" in updates:
                        record["year"] = updates.get("year")
                    if "genre" in updates:
                        record["genre"] = (updates.get("genre") or "").strip()
                    if "rating" in updates:
                        record["rating"] = _safe_int(updates.get("rating"), default=record.get("rating", 0))
                    if "notes" in updates:
                        record["notes"] = (updates.get("notes") or "").strip()
                    results["updated"].append(_record_summary(record))
        else:
            results["skipped"].append(f"Unknown action: {action}")

    if results["added"] or results["updated"] or results["deleted"]:
        _save_library(library)
    return results


def _summarize_action_results(results: Dict[str, List[str]]) -> str:
    lines = []
    if results["added"]:
        lines.append("Added:")
        lines.extend([f"- {item}" for item in results["added"]])
    if results["updated"]:
        lines.append("Updated:")
        lines.extend([f"- {item}" for item in results["updated"]])
    if results["deleted"]:
        lines.append("Deleted:")
        lines.extend([f"- {item}" for item in results["deleted"]])
    if results["skipped"]:
        lines.append("Skipped:")
        lines.extend([f"- {item}" for item in results["skipped"]])
    return "\n".join(lines).strip()


def _render_virtuoso(library: dict, records: List[dict]) -> None:
    st.subheader("🪄 Wax Wizard")
    st.caption(
        "Slip into the groove. Ask for picks, vault insights, or tell the wizard what to update."
    )

    api_key = _get_openai_key()
    model = st.text_input(
        "Model",
        value=DEFAULT_OPENAI_MODEL,
        help="Defaults to GPT-5.2 for Wax Wizard.",
        key="virtuoso_model",
    )
    reasoning_effort = st.selectbox(
        "Reasoning effort",
        ["none", "low", "medium", "high", "xhigh"],
        index=2,
        help="Higher effort uses more reasoning tokens for deeper answers.",
    )

    if not api_key:
        st.info("Add your OpenAI API key to secrets as `openai_api_key` to enable chat.")

    if "virtuoso_messages" not in st.session_state:
        st.session_state["virtuoso_messages"] = []

    chat_col, action_col = st.columns([4, 1])
    with action_col:
        if st.button("Clear chat", use_container_width=True):
            st.session_state["virtuoso_messages"] = []

    with chat_col:
        for message in st.session_state["virtuoso_messages"]:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        prompt = st.chat_input("Tell the Wax Wizard what you need")
        if prompt:
            st.session_state["virtuoso_messages"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            if not api_key:
                with st.chat_message("assistant"):
                    st.warning("Add an OpenAI API key to enable Vinyl Virtuoso.")
                return

            with st.chat_message("assistant"):
                with st.spinner("Digging through the vault..."):
                    client = OpenAI(api_key=api_key)
                    command_response = client.responses.create(
                        model=model,
                        input=[
                            {"role": "system", "content": _build_wax_wizard_prompt(records)},
                            *st.session_state["virtuoso_messages"],
                        ],
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": "wax_wizard_actions",
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "assistant_reply": {"type": "string"},
                                        "actions": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "action": {
                                                        "type": "string",
                                                        "enum": ["add", "update", "delete"],
                                                    },
                                                    "record": {
                                                        "type": "object",
                                                        "properties": {
                                                            "artist": {"type": "string"},
                                                            "album": {"type": "string"},
                                                            "year": {"type": ["integer", "null"]},
                                                            "genre": {"type": "string"},
                                                            "rating": {"type": "integer"},
                                                            "notes": {"type": "string"},
                                                        },
                                                        "required": ["artist", "album"],
                                                    },
                                                    "match": {
                                                        "type": "object",
                                                        "properties": {
                                                            "id": {"type": "string"},
                                                            "artist": {"type": "string"},
                                                            "album": {"type": "string"},
                                                            "year": {"type": ["integer", "null"]},
                                                        },
                                                    },
                                                    "updates": {
                                                        "type": "object",
                                                        "properties": {
                                                            "artist": {"type": "string"},
                                                            "album": {"type": "string"},
                                                            "year": {"type": ["integer", "null"]},
                                                            "genre": {"type": "string"},
                                                            "rating": {"type": "integer"},
                                                            "notes": {"type": "string"},
                                                        },
                                                    },
                                                },
                                                "required": ["action"],
                                            },
                                        },
                                    },
                                    "required": ["assistant_reply", "actions"],
                                },
                            },
                        },
                        reasoning={"effort": reasoning_effort},
                    )
                    command_payload = _parse_json_payload(command_response.output_text or "")
                    assistant_reply = command_payload.get("assistant_reply", "")
                    actions = command_payload.get("actions") or []
                    action_results = _apply_vault_actions(library, actions)
                    action_summary = _summarize_action_results(action_results)

                    if not assistant_reply:
                        assistant_reply = (
                            "I'm here with the lights low and the needle ready. "
                            "Tell me what you'd like to spin or change in the vault."
                        )
                    assistant_text = assistant_reply
                    if action_summary:
                        assistant_text = f"{assistant_text}\n\nVault update results:\n{action_summary}"

                    st.write(assistant_text)
                    st.session_state["virtuoso_messages"].append(
                        {"role": "assistant", "content": assistant_text}
                    )
                    if action_results["added"] or action_results["updated"] or action_results["deleted"]:
                        st.rerun()


def _filtered_records(records: List[dict], query: str, genres: List[str]) -> List[dict]:
    if not query and not genres:
        return records
    query_lower = query.lower().strip()
    filtered = []
    for record in records:
        genre = str(record.get("genre") or "")
        if genres and genre not in genres:
            continue
        if query_lower:
            haystack = " ".join(
                [
                    str(record.get("artist") or ""),
                    str(record.get("album") or ""),
                    str(record.get("notes") or ""),
                    genre,
                ]
            ).lower()
            if query_lower not in haystack:
                continue
        filtered.append(record)
    return filtered


st.set_page_config(page_title="Vinyl Vault", page_icon="🎧", layout="wide")
st.title("🎧 Vinyl Vault")
st.caption(
    "Slip into a calm groove while the Wax Wizard keeps your collection organized, "
    "with quick summaries, labels, and QR codes."
)

library = _load_library()
records = library.get("records", [])

with st.sidebar:
    st.header("Add a record")
    with st.form("add_record"):
        artist = st.text_input("Artist", placeholder="Fleetwood Mac")
        album = st.text_input("Album", placeholder="Rumours")
        year = _safe_int(st.text_input("Year", placeholder="1977"), default=0)
        genre = st.text_input("Genre", placeholder="Rock")
        rating = st.slider("Rating", 0, 5, 4)
        notes = st.text_area("Notes", placeholder="Favorite tracks, pressing info, etc.")
        submitted = st.form_submit_button("Save record")

    if submitted:
        record = {
            "id": uuid.uuid4().hex,
            "artist": artist.strip() or "Unknown Artist",
            "album": album.strip() or "Untitled Album",
            "year": year if year else None,
            "genre": genre.strip(),
            "rating": rating,
            "notes": notes.strip(),
            "added_at": _iso_now(),
        }
        _add_record(library, record)
        st.success("Saved to Vinyl Vault.")
        st.rerun()

    st.divider()
    st.header("Bulk add")
    with st.form("bulk_add"):
        bulk_text = st.text_area(
            "Paste one record per line (Artist – Album)",
            placeholder="Fleetwood Mac – Rumours",
            height=160,
        )
        bulk_rating = st.slider("Default rating", 0, 5, 4, key="bulk_rating")
        bulk_submitted = st.form_submit_button("Add records")

    if bulk_submitted:
        lines = [line.strip() for line in bulk_text.splitlines() if line.strip()]
        added = 0
        skipped = []
        for line in lines:
            parsed = _parse_bulk_line(line)
            if not parsed:
                skipped.append(line)
                continue
            artist_name, album_name = parsed
            record = {
                "id": uuid.uuid4().hex,
                "artist": artist_name,
                "album": album_name,
                "year": None,
                "genre": "",
                "rating": bulk_rating,
                "notes": "",
                "added_at": _iso_now(),
            }
            _add_record(library, record)
            added += 1

        if added:
            st.success(f"Added {added} record(s) from bulk upload.")
        if skipped:
            st.warning(
                "Skipped lines without an Artist – Album format:\n"
                + "\n".join(f"• {line}" for line in skipped)
            )
        if added:
            st.rerun()

    st.divider()
    st.header("Filter")
    query = st.text_input("Search", placeholder="artist, album, notes")
    genre_options = sorted({r.get("genre") for r in records if r.get("genre")})
    selected_genres = st.multiselect("Genres", genre_options)

filtered = _filtered_records(records, query, selected_genres)
df = _records_dataframe(filtered)

col_a, col_b, col_c = st.columns([1, 1, 1])
with col_a:
    st.metric("Total records", len(records))
with col_b:
    ratings = [r.get("rating") for r in records if r.get("rating") is not None]
    avg_rating = int(np.mean(ratings)) if ratings else 0
    st.metric("Average rating", _stars(avg_rating))
with col_c:
    latest = records[-1]["added_at"] if records else "—"
    st.metric("Last added", latest)

st.subheader("Collection overview")
if not df.empty:
    chart = (
        alt.Chart(df)
        .mark_bar(color="#4C6EF5")
        .encode(
            x=alt.X("rating:O", title="Rating"),
            y=alt.Y("count():Q", title="Count"),
            tooltip=["rating:O", "count():Q"],
        )
        .properties(height=220)
    )
    st.altair_chart(chart, width="stretch")
else:
    st.info("Add your first record to see stats.")

st.subheader("Library")
st.dataframe(
    df[["artist", "album", "year", "genre", "rating", "added_at", "notes"]],
    width="stretch",
    hide_index=True,
)

if filtered:
    st.subheader("Record tools")
    record_map = {_record_summary(r): r for r in filtered}
    selection = st.selectbox("Select a record", list(record_map.keys()))
    record = record_map[selection]

    tool_cols = st.columns([1, 1, 2])
    with tool_cols[0]:
        st.markdown("**Quick view**")
        st.write(_record_summary(record))
        st.write(f"Rating: {_stars(record.get('rating', 0))}")
    with tool_cols[1]:
        qr_payload = json.dumps(
            {
                "artist": record.get("artist"),
                "album": record.get("album"),
                "year": record.get("year"),
                "genre": record.get("genre"),
            },
            ensure_ascii=False,
        )
        qr_img = qrcode.make(qr_payload)
        qr_buffer = io.BytesIO()
        qr_img.get_image().save(qr_buffer, format="PNG")
        st.image(qr_buffer.getvalue(), caption="QR: record details", width=180)
    with tool_cols[2]:
        pdf_bytes = _build_label_pdf(record)
        st.download_button(
            "Download label PDF",
            data=pdf_bytes,
            file_name=f"{_slug(_record_summary(record))}-label.pdf",
            mime="application/pdf",
        )

    with st.expander("Remove records"):
        remove_options = st.multiselect(
            "Select records to remove",
            [f"{_record_summary(r)} ({r.get('id')})" for r in filtered],
        )
        if st.button("Delete selected records", type="primary", disabled=not remove_options):
            ids = [opt.split("(")[-1].rstrip(")") for opt in remove_options]
            removed = _remove_records(library, ids)
            st.success(f"Removed {removed} record(s).")
            st.rerun()
else:
    st.info("No records match your filters.")

st.divider()
_render_virtuoso(library, records)
