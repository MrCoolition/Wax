from __future__ import annotations

import datetime as dt
import io
import json
import os
import re
from urllib.parse import quote_plus
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd
import qrcode
import streamlit as st
from openai import OpenAI
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from auth0_streamlit import AuthError, get_auth0_config, get_auth0_debug_log, logout_url, require_auth0_login
from vinyl_repo import VinylRepo, VinylRepoError


APP_DIR = Path(__file__).parent
ASSETS_DIR = APP_DIR / "assets"
DEFAULT_OPENAI_MODEL = "gpt-5.2"


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None or x == "":
            return default
        return int(x)
    except Exception:
        return default


def _now_naive() -> dt.datetime:
    return dt.datetime.now().replace(tzinfo=None)


def _iso_now() -> str:
    return _now_naive().replace(microsecond=0).isoformat(sep=" ")


def _parse_list_line(line: str) -> Optional[Tuple[str, str]]:
    if not line:
        return None
    cleaned = re.sub(r"^[\s•*-]+\s*", "", line.strip())
    if not cleaned:
        return None
    parts = None
    for delimiter in (" – ", " — ", " - "):
        if delimiter in cleaned:
            parts = cleaned.split(delimiter, 1)
            break
    if not parts:
        parts = re.split(r"\s+[–—-]\s+", cleaned, maxsplit=1)
    if len(parts) < 2:
        return None
    artist, album = [p.strip() for p in parts]
    if not artist or not album:
        return None
    return artist, album


def _normalize_list_input(text: str) -> List[str]:
    if not text:
        return []
    normalized = text.replace("\\n", "\n")
    return [line.strip() for line in normalized.splitlines() if line.strip()]


def _stars(rating: int) -> str:
    r = max(0, min(5, int(rating)))
    return "★" * r + "☆" * (5 - r)


def _slug(s: str, max_len: int = 60) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s[:max_len] or "file"


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


def _build_repo_dsn(secrets: Dict[str, Any]) -> Optional[str]:
    explicit = (
        secrets.get("POSTGRES_DSN")
        or secrets.get("postgres_dsn")
        or secrets.get("DATABASE_URL")
        or secrets.get("database_url")
    )
    if explicit:
        return explicit

    host = secrets.get("AIVEN_HOST")
    port = secrets.get("AIVEN_PORT") or "5432"
    user = secrets.get("AIVEN_USER")
    password = secrets.get("AIVEN_PASSWORD")
    db_name = secrets.get("AIVEN_DB")
    if not all([host, user, password, db_name]):
        return None

    user_enc = quote_plus(str(user))
    password_enc = quote_plus(str(password))
    return f"postgresql://{user_enc}:{password_enc}@{host}:{port}/{db_name}"


def _build_vault_context(records: List[dict]) -> str:
    if not records:
        return "[]"
    payload = [
        {
            "id": r.get("id"),
            "artist": r.get("artist"),
            "album": r.get("album"),
            "year": r.get("year"),
            "genre": r.get("genre"),
            "rating": r.get("rating"),
            "notes": r.get("notes"),
            "added_at": r.get("added_at"),
        }
        for r in records
    ]
    return json.dumps(payload, ensure_ascii=False)


def _build_vault_digest(records: List[dict]) -> str:
    if not records:
        return "No records yet."
    lines = []
    for record in records[:300]:
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


def _parse_json_payload(text: str) -> dict:
    if not text:
        return {}
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return {}
        return {}


def _summarize_action_results(results: Dict[str, List[str]]) -> str:
    lines = []
    if results.get("added"):
        lines.append("Added:")
        lines.extend([f"- {x}" for x in results["added"]])
    if results.get("updated"):
        lines.append("Updated:")
        lines.extend([f"- {x}" for x in results["updated"]])
    if results.get("deleted"):
        lines.append("Deleted:")
        lines.extend([f"- {x}" for x in results["deleted"]])
    if results.get("skipped"):
        lines.append("Skipped:")
        lines.extend([f"- {x}" for x in results["skipped"]])
    return "\n".join(lines).strip()


st.set_page_config(page_title="Vinyl Vault", page_icon="🎧", layout="wide")
st.title("🎧 Vinyl Vault")
st.caption(
    "Slip into a calm groove while the Wax Wizard keeps your collection organized, "
    "with quick summaries, labels, and QR codes."
)

try:
    claims = require_auth0_login()
except AuthError as e:
    st.error(str(e))
    debug_log = get_auth0_debug_log()
    if debug_log:
        with st.expander("Auth0 troubleshooting log"):
            st.json(debug_log, expanded=False)
    st.stop()

auth0_sub = str(claims.get("sub") or "")
email = str(claims.get("email") or "")
display_name = str(claims.get("name") or claims.get("nickname") or email or "Vinyl Collector")

if not auth0_sub or not email:
    st.error("Auth0 did not provide required claims (sub/email). Enable email scope/claim in Auth0.")
    st.stop()

try:
    secrets = st.secrets
    repo_dsn = _build_repo_dsn(secrets)
    REPO = VinylRepo(dsn=repo_dsn)
except VinylRepoError as e:
    st.error(str(e))
    st.stop()

if "user_id" not in st.session_state:
    st.session_state["user_id"] = REPO.ensure_user_from_auth0(
        auth0_sub=auth0_sub,
        email=email,
        display_name=display_name,
    )

user_id = st.session_state["user_id"]

cfg = get_auth0_config()
top_left, top_right = st.columns([3, 1])
with top_left:
    st.write(f"Logged in as **{display_name}** (`{email}`)")
with top_right:
    st.link_button("Logout", logout_url(cfg), width="stretch")

all_records = REPO.list_records(user_id=user_id, limit=10000)

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
        REPO.add_record(
            user_id=user_id,
            artist=artist.strip() or "Unknown Artist",
            album=album.strip() or "Untitled Album",
            year=year if year else None,
            genre=genre.strip(),
            rating=rating,
            notes=notes.strip(),
        )
        st.success("Saved to Vinyl Vault.")
        st.rerun()

    st.divider()
    st.header("Paste a list")
    with st.form("list_add"):
        list_text = st.text_area(
            "Paste one record per line (Artist – Album).",
            placeholder="Fleetwood Mac – Rumours",
            height=180,
            help="If your list has literal \\n characters, this will split them automatically.",
        )
        list_rating = st.slider("Default rating", 0, 5, 4, key="list_rating")
        list_submitted = st.form_submit_button("Add list")

    if list_submitted:
        lines = _normalize_list_input(list_text)
        if not lines:
            st.warning("Paste at least one line in the Artist – Album format to add a list.")
        else:
            added = 0
            skipped_lines: List[str] = []
            failed_lines: List[str] = []
            for line in lines:
                parsed = _parse_list_line(line)
                if not parsed:
                    skipped_lines.append(line)
                    continue
                artist_name, album_name = parsed
                try:
                    REPO.add_record(
                        user_id=user_id,
                        artist=artist_name,
                        album=album_name,
                        year=None,
                        genre="",
                        rating=list_rating,
                        notes="",
                    )
                    added += 1
                except Exception as exc:
                    failed_lines.append(f"{line} -> {exc}")

            if added:
                st.success(f"Added {added} record(s) from your list.")
            if skipped_lines:
                st.warning(
                    "Skipped lines without an Artist – Album format:\n"
                    + "\n".join(f"• {line}" for line in skipped_lines)
                )
            if failed_lines:
                st.warning("Some rows failed to insert:\n" + "\n".join(f"• {x}" for x in failed_lines[:20]))
            if added:
                st.rerun()

    st.divider()
    st.header("Filter")
    query = st.text_input("Search", placeholder="artist, album, notes")
    genre_options = sorted({r.get("genre") for r in all_records if r.get("genre")})
    selected_genres = st.multiselect("Genres", genre_options)

filtered = REPO.list_records(user_id=user_id, query=query, genres=selected_genres, limit=10000)
df = pd.DataFrame.from_records(filtered) if filtered else pd.DataFrame(
    columns=["artist", "album", "year", "genre", "rating", "added_at", "notes"]
)
if not df.empty:
    df["rating"] = df["rating"].fillna(0).astype(int)
    df = df.sort_values(by=["added_at", "artist", "album"], ascending=False)

col_a, col_b, col_c = st.columns([1, 1, 1])
with col_a:
    st.metric("Total records", len(all_records))
with col_b:
    ratings = [r.get("rating") for r in all_records if r.get("rating") is not None]
    avg_rating = int(np.mean(ratings)) if ratings else 0
    st.metric("Average rating", _stars(avg_rating))
with col_c:
    latest = max((r.get("added_at") for r in all_records), default="—")
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
    df[["artist", "album", "year", "genre", "rating", "added_at", "notes"]] if not df.empty else df,
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
            removed = REPO.delete_records(user_id=user_id, record_ids=ids)
            st.success(f"Removed {removed} record(s).")
            st.rerun()
else:
    st.info("No records match your filters.")

st.divider()

st.subheader("🪄 Wax Wizard")
st.caption("Ask for picks, vault insights, or tell the wizard what to update.")

api_key = _get_openai_key()
model = st.text_input("Model", value=DEFAULT_OPENAI_MODEL, key="virtuoso_model")
reasoning_effort = st.selectbox("Reasoning effort", ["none", "low", "medium", "high", "xhigh"], index=2)

if not api_key:
    st.info("Add your OpenAI API key to secrets as `openai_api_key` to enable chat.")

if "virtuoso_messages" not in st.session_state:
    st.session_state["virtuoso_messages"] = []

chat_col, action_col = st.columns([4, 1])
with action_col:
    if st.button("Clear chat", width="stretch"):
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
                st.warning("Add an OpenAI API key to enable Wax Wizard.")
            st.stop()

        with st.chat_message("assistant"):
            with st.spinner("Digging through the vault..."):
                client = OpenAI(api_key=api_key)
                command_response = client.responses.create(
                    model=model,
                    input=[
                        {"role": "system", "content": _build_wax_wizard_prompt(all_records)},
                        *st.session_state["virtuoso_messages"],
                    ],
                    reasoning={"effort": reasoning_effort},
                )

                raw_text = (command_response.output_text or "").strip()
                command_payload = _parse_json_payload(raw_text)
                assistant_reply = (command_payload.get("assistant_reply") or "").strip()
                actions = command_payload.get("actions") or []

                if not assistant_reply:
                    assistant_reply = "Tell me what you want changed in the vault, and I’ll do it carefully."

                action_summary = ""
                if actions:
                    results = REPO.apply_wax_actions(
                        user_id=user_id,
                        raw_text=raw_text,
                        parsed_payload=command_payload,
                        actions=actions,
                    )
                    action_summary = _summarize_action_results(results)

                assistant_text = assistant_reply
                if action_summary:
                    assistant_text = f"{assistant_text}\n\nVault update results:\n{action_summary}"

                st.write(assistant_text)
                st.session_state["virtuoso_messages"].append(
                    {"role": "assistant", "content": assistant_text}
                )

                if actions:
                    st.rerun()
