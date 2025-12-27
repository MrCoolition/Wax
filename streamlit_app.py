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
    preferred = APP_DIR / "data"
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        test = preferred / ".write_test"
        test.write_text("ok", encoding="utf-8")
        test.unlink(missing_ok=True)
        return preferred
    except Exception:
        fallback = Path(tempfile.gettempdir()) / "vinyl_vault"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


DATA_DIR = _resolve_data_root()
DB_PATH = DATA_DIR / "vinyl_vault.json"
MEDIA_DIR = DATA_DIR / "media"
MEDIA_DIR.mkdir(parents=True, exist_ok=True)


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
st.caption("Keep your vinyl collection organized with quick summaries, labels, and QR codes.")

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
        st.experimental_rerun()

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
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("Add your first record to see stats.")

st.subheader("Library")
st.dataframe(
    df[["artist", "album", "year", "genre", "rating", "added_at", "notes"]],
    use_container_width=True,
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
        st.image(qr_img, caption="QR: record details", width=180)
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
            st.experimental_rerun()
else:
    st.info("No records match your filters.")
