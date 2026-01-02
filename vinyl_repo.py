from __future__ import annotations

import json
import os
from urllib.parse import quote_plus
from typing import Any, Dict, List, Optional, Sequence, Tuple

import psycopg
from psycopg import errors
from psycopg.rows import dict_row


class VinylRepoError(RuntimeError):
    pass


def _dsn(explicit_dsn: Optional[str] = None) -> str:
    dsn = explicit_dsn or os.getenv("POSTGRES_DSN") or os.getenv("DATABASE_URL") or ""
    if not dsn:
        host = os.getenv("AIVEN_HOST")
        port = os.getenv("AIVEN_PORT") or "5432"
        user = os.getenv("AIVEN_USER")
        password = os.getenv("AIVEN_PASSWORD")
        db_name = os.getenv("AIVEN_DB")
        if host and user and password and db_name:
            user_enc = quote_plus(user)
            password_enc = quote_plus(password)
            dsn = f"postgresql://{user_enc}:{password_enc}@{host}:{port}/{db_name}"
        else:
            raise VinylRepoError(
                "Missing Postgres DSN (POSTGRES_DSN or DATABASE_URL or explicit_dsn) "
                "or AIVEN_HOST/AIVEN_USER/AIVEN_PASSWORD/AIVEN_DB."
            )
    return dsn


def _set_session(conn: psycopg.Connection, *, user_id: Optional[str]) -> None:
    uid = str(user_id or "")
    with conn.cursor() as cur:
        cur.execute("SELECT set_config('app.user_id', %s, true);", (uid,))


class VinylRepo:
    def __init__(self, dsn: Optional[str] = None):
        self._dsn = _dsn(dsn)

    def _connect(self, *, user_id: Optional[str]) -> psycopg.Connection:
        conn = psycopg.connect(self._dsn, row_factory=dict_row)
        conn.autocommit = False
        _set_session(conn, user_id=user_id)
        return conn

    def ensure_user_from_auth0(
        self,
        *,
        auth0_sub: str,
        email: str,
        display_name: Optional[str],
    ) -> str:
        if not auth0_sub:
            raise VinylRepoError("auth0_sub required")
        if not email:
            raise VinylRepoError("email required")

        try:
            with self._connect(user_id=None) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT id
                        FROM vinyl.app_user
                        WHERE auth0_sub = %s
                        LIMIT 1;
                        """,
                        (auth0_sub,),
                    )
                    row = cur.fetchone()
                    if row:
                        user_id = str(row["id"])
                        cur.execute(
                            """
                            UPDATE vinyl.app_user
                            SET email = %s,
                                display_name = COALESCE(%s, display_name)
                            WHERE id = %s;
                            """,
                            (email, display_name, user_id),
                        )
                        conn.commit()
                        return user_id

                    cur.execute(
                        """
                        SELECT id
                        FROM vinyl.app_user
                        WHERE email = %s
                        LIMIT 1;
                        """,
                        (email,),
                    )
                    row = cur.fetchone()
                    if row:
                        user_id = str(row["id"])
                        cur.execute(
                            """
                            UPDATE vinyl.app_user
                            SET auth0_sub = %s,
                                display_name = COALESCE(%s, display_name)
                            WHERE id = %s;
                            """,
                            (auth0_sub, display_name, user_id),
                        )
                        conn.commit()
                        return user_id

                    cur.execute(
                        """
                        INSERT INTO vinyl.app_user (auth0_sub, email, display_name)
                        VALUES (%s, %s, %s)
                        RETURNING id;
                        """,
                        (auth0_sub, email, display_name),
                    )
                    user_id = str(cur.fetchone()["id"])
                    conn.commit()
                    return user_id
        except errors.InsufficientPrivilege as exc:
            raise VinylRepoError(
                "Database user lacks privileges for vinyl.app_user. "
                "Grant SELECT/INSERT/UPDATE on vinyl.app_user and USAGE on schema vinyl "
                "to the configured database user."
            ) from exc

    def list_records(
        self,
        *,
        user_id: str,
        query: str = "",
        genres: Sequence[str] = (),
        limit: int = 10000,
    ) -> List[Dict[str, Any]]:
        q = (query or "").strip().lower()
        genres = [g for g in (genres or ()) if g]

        where = ["user_id = %s"]
        params: List[Any] = [user_id]

        if genres:
            where.append("genre = ANY(%s)")
            params.append(genres)

        if q:
            where.append(
                "("
                "lower(coalesce(artist,'')) LIKE %s OR "
                "lower(coalesce(album,'')) LIKE %s OR "
                "lower(coalesce(notes,'')) LIKE %s OR "
                "lower(coalesce(genre,'')) LIKE %s"
                ")"
            )
            like = f"%{q}%"
            params.extend([like, like, like, like])

        where_sql = " AND ".join(where)

        sql = f"""
        SELECT id, artist, album, year, genre, rating, added_at, notes, album_id
        FROM vinyl.v_record_flat
        WHERE {where_sql}
        ORDER BY added_at DESC, artist ASC, album ASC
        LIMIT %s;
        """
        params.append(int(limit))

        with self._connect(user_id=user_id) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
                conn.commit()

        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append(
                {
                    "id": str(r["id"]),
                    "artist": (r.get("artist") or "").strip(),
                    "album": (r.get("album") or "").strip(),
                    "year": r.get("year"),
                    "genre": (r.get("genre") or "").strip(),
                    "rating": int(r.get("rating") or 0),
                    "added_at": str(r.get("added_at")),
                    "notes": (r.get("notes") or "").strip(),
                    "album_id": str(r.get("album_id")) if r.get("album_id") else None,
                }
            )
        return out

    def _get_or_create_artist_id(self, cur, *, name: str) -> str:
        cur.execute("SELECT id FROM vinyl.artist WHERE name = %s LIMIT 1;", (name,))
        row = cur.fetchone()
        if row:
            return str(row["id"])
        cur.execute("INSERT INTO vinyl.artist (name) VALUES (%s) RETURNING id;", (name,))
        return str(cur.fetchone()["id"])

    def _get_or_create_genre_id(self, cur, *, name: Optional[str]) -> Optional[str]:
        if not name:
            return None
        cur.execute("SELECT id FROM vinyl.genre WHERE name = %s LIMIT 1;", (name,))
        row = cur.fetchone()
        if row:
            return str(row["id"])
        cur.execute("INSERT INTO vinyl.genre (name) VALUES (%s) RETURNING id;", (name,))
        return str(cur.fetchone()["id"])

    def _get_or_create_album_id(
        self,
        cur,
        *,
        artist_id: str,
        title: str,
        release_year: Optional[int],
        genre_id: Optional[str],
    ) -> str:
        cur.execute(
            """
            SELECT id, genre_id
            FROM vinyl.album
            WHERE artist_id = %s
              AND lower(title) = lower(%s)
              AND (
                (release_year IS NULL AND %s IS NULL)
                OR release_year = %s
              )
            LIMIT 1;
            """,
            (artist_id, title, release_year, release_year),
        )
        row = cur.fetchone()
        if row:
            album_id = str(row["id"])
            if genre_id and not row.get("genre_id"):
                cur.execute("UPDATE vinyl.album SET genre_id = %s WHERE id = %s;", (genre_id, album_id))
            return album_id

        cur.execute(
            """
            INSERT INTO vinyl.album (artist_id, title, release_year, genre_id)
            VALUES (%s, %s, %s, %s)
            RETURNING id;
            """,
            (artist_id, title, release_year, genre_id),
        )
        return str(cur.fetchone()["id"])

    def add_record(
        self,
        *,
        user_id: str,
        artist: str,
        album: str,
        year: Optional[int],
        genre: Optional[str],
        rating: int,
        notes: str,
    ) -> str:
        artist = (artist or "").strip() or "Unknown Artist"
        album = (album or "").strip() or "Untitled Album"
        genre = (genre or "").strip() or None
        notes = (notes or "").strip()
        rating = int(max(0, min(5, int(rating))))

        with self._connect(user_id=user_id) as conn:
            with conn.cursor() as cur:
                artist_id = self._get_or_create_artist_id(cur, name=artist)
                genre_id = self._get_or_create_genre_id(cur, name=genre)
                album_id = self._get_or_create_album_id(
                    cur,
                    artist_id=artist_id,
                    title=album,
                    release_year=year,
                    genre_id=genre_id,
                )
                cur.execute(
                    """
                    INSERT INTO vinyl.collection_item (user_id, album_id, rating, notes)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id;
                    """,
                    (user_id, album_id, rating, notes),
                )
                new_id = str(cur.fetchone()["id"])
            conn.commit()
            return new_id

    def bulk_add(
        self,
        *,
        user_id: str,
        items: Sequence[Dict[str, Any]],
        default_rating: int = 4,
    ) -> Tuple[int, List[str]]:
        added = 0
        skipped: List[str] = []
        default_rating = int(max(0, min(5, int(default_rating))))

        with self._connect(user_id=user_id) as conn:
            with conn.cursor() as cur:
                for it in items:
                    try:
                        artist = (it.get("artist") or "").strip()
                        album = (it.get("album") or "").strip()
                        if not artist or not album:
                            skipped.append(f"missing artist/album: {it!r}")
                            continue

                        year = it.get("year")
                        year = int(year) if year not in (None, "", 0) else None
                        genre = (it.get("genre") or "").strip() or None
                        rating = it.get("rating", default_rating)
                        rating = int(max(0, min(5, int(rating))))
                        notes = (it.get("notes") or "").strip()

                        artist_id = self._get_or_create_artist_id(cur, name=artist)
                        genre_id = self._get_or_create_genre_id(cur, name=genre)
                        album_id = self._get_or_create_album_id(
                            cur,
                            artist_id=artist_id,
                            title=album,
                            release_year=year,
                            genre_id=genre_id,
                        )
                        cur.execute(
                            """
                            INSERT INTO vinyl.collection_item (user_id, album_id, rating, notes)
                            VALUES (%s, %s, %s, %s);
                            """,
                            (user_id, album_id, rating, notes),
                        )
                        added += 1
                    except Exception as e:
                        skipped.append(f"{it!r} -> {e}")

            conn.commit()
        return added, skipped

    def delete_records(self, *, user_id: str, record_ids: Sequence[str]) -> int:
        ids = [x for x in (record_ids or []) if x]
        if not ids:
            return 0

        with self._connect(user_id=user_id) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM vinyl.collection_item
                    WHERE user_id = %s
                      AND id = ANY(%s::uuid[]);
                    """,
                    (user_id, ids),
                )
                deleted = int(cur.rowcount or 0)
            conn.commit()
            return deleted

    def _resolve_ids(self, cur, *, user_id: str, match: Dict[str, Any]) -> List[str]:
        if not match:
            return []
        if match.get("id"):
            return [str(match["id"])]

        artist = (match.get("artist") or "").strip()
        album = (match.get("album") or "").strip()
        year = match.get("year", None)

        where = ["user_id = %s"]
        params: List[Any] = [user_id]

        if artist:
            where.append("lower(artist) = lower(%s)")
            params.append(artist)
        if album:
            where.append("lower(album) = lower(%s)")
            params.append(album)
        if year is not None:
            where.append("year = %s")
            params.append(year)

        sql = f"""
        SELECT id
        FROM vinyl.v_record_flat
        WHERE {" AND ".join(where)}
        ORDER BY added_at DESC
        LIMIT 50;
        """
        cur.execute(sql, params)
        return [str(r["id"]) for r in (cur.fetchall() or [])]

    def _fetch_records_by_ids(self, cur, *, user_id: str, ids: Sequence[str]) -> List[Dict[str, Any]]:
        if not ids:
            return []
        cur.execute(
            """
            SELECT id, artist, album, year, genre, rating, notes, album_id
            FROM vinyl.v_record_flat
            WHERE user_id = %s
              AND id = ANY(%s::uuid[]);
            """,
            (user_id, list(ids)),
        )
        return [dict(r) for r in (cur.fetchall() or [])]

    def audit(self, *, user_id: str, action_type: str, payload: Dict[str, Any]) -> str:
        with self._connect(user_id=user_id) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO vinyl.vault_action_audit (user_id, action_type, payload)
                    VALUES (%s, %s, %s::jsonb)
                    RETURNING id;
                    """,
                    (user_id, action_type, json.dumps(payload)),
                )
                audit_id = str(cur.fetchone()["id"])
            conn.commit()
            return audit_id

    def apply_wax_actions(
        self,
        *,
        user_id: str,
        raw_text: str,
        parsed_payload: Dict[str, Any],
        actions: Sequence[Dict[str, Any]],
    ) -> Dict[str, List[str]]:
        results: Dict[str, List[str]] = {"added": [], "updated": [], "deleted": [], "skipped": []}

        def summary_line(r: Dict[str, Any]) -> str:
            y = r.get("year")
            year_txt = f" ({y})" if y else ""
            return f"{r.get('artist','Unknown')} — {r.get('album','Untitled')}{year_txt}"

        with self._connect(user_id=user_id) as conn:
            try:
                with conn.cursor() as cur:
                    for action in actions or []:
                        action_type = (action.get("action") or "").lower().strip()

                        if action_type == "add":
                            rec = action.get("record") or {}
                            new_id = self.add_record(
                                user_id=user_id,
                                artist=rec.get("artist") or "Unknown Artist",
                                album=rec.get("album") or "Untitled Album",
                                year=rec.get("year"),
                                genre=rec.get("genre"),
                                rating=rec.get("rating", 0),
                                notes=rec.get("notes", ""),
                            )
                            cur.execute(
                                """
                                SELECT artist, album, year
                                FROM vinyl.v_record_flat
                                WHERE user_id = %s AND id = %s::uuid
                                LIMIT 1;
                                """,
                                (user_id, new_id),
                            )
                            row = cur.fetchone() or {}
                            results["added"].append(summary_line(row))

                        elif action_type in {"update", "delete"}:
                            match = action.get("match") or {}
                            ids = self._resolve_ids(cur, user_id=user_id, match=match)
                            if not ids:
                                results["skipped"].append(f"{action_type}: no match for {match}")
                                continue

                            if action_type == "delete":
                                cur.execute(
                                    """
                                    DELETE FROM vinyl.collection_item
                                    WHERE user_id = %s AND id = ANY(%s::uuid[]);
                                    """,
                                    (user_id, ids),
                                )
                                for _ in ids:
                                    results["deleted"].append(f"Deleted item matching {match}")
                                continue

                            updates = action.get("updates") or {}
                            current_rows = self._fetch_records_by_ids(cur, user_id=user_id, ids=ids)

                            for r in current_rows:
                                new_artist = (
                                    updates.get("artist") if "artist" in updates else r.get("artist")
                                ) or r.get("artist")
                                new_album = (
                                    updates.get("album") if "album" in updates else r.get("album")
                                ) or r.get("album")
                                new_year = updates.get("year") if "year" in updates else r.get("year")
                                new_genre = (
                                    updates.get("genre") if "genre" in updates else r.get("genre")
                                ) or r.get("genre")

                                new_rating = updates.get("rating") if "rating" in updates else r.get("rating")
                                new_notes = updates.get("notes") if "notes" in updates else r.get("notes")

                                if (
                                    (new_artist or "").strip().lower()
                                    != (r.get("artist") or "").strip().lower()
                                    or (new_album or "").strip().lower()
                                    != (r.get("album") or "").strip().lower()
                                    or (new_year != r.get("year"))
                                    or (new_genre or "").strip().lower()
                                    != (r.get("genre") or "").strip().lower()
                                ):
                                    artist_id = self._get_or_create_artist_id(
                                        cur, name=(new_artist or "Unknown Artist").strip()
                                    )
                                    genre_id = self._get_or_create_genre_id(
                                        cur, name=(new_genre or "").strip() or None
                                    )
                                    album_id = self._get_or_create_album_id(
                                        cur,
                                        artist_id=artist_id,
                                        title=(new_album or "Untitled Album").strip(),
                                        release_year=int(new_year) if new_year not in (None, "", 0) else None,
                                        genre_id=genre_id,
                                    )
                                    cur.execute(
                                        """
                                        UPDATE vinyl.collection_item
                                        SET album_id = %s
                                        WHERE user_id = %s AND id = %s::uuid;
                                        """,
                                        (album_id, user_id, str(r["id"])),
                                    )

                                if "rating" in updates or "notes" in updates:
                                    rr = int(max(0, min(5, int(new_rating or 0))))
                                    nn = (new_notes or "").strip()
                                    cur.execute(
                                        """
                                        UPDATE vinyl.collection_item
                                        SET rating = %s,
                                            notes = %s
                                        WHERE user_id = %s AND id = %s::uuid;
                                        """,
                                        (rr, nn, user_id, str(r["id"])),
                                    )

                                results["updated"].append(
                                    summary_line({"artist": new_artist, "album": new_album, "year": new_year})
                                )

                        else:
                            results["skipped"].append(f"Unknown action: {action}")

                    cur.execute(
                        """
                        INSERT INTO vinyl.vault_action_audit (user_id, action_type, payload)
                        VALUES (%s, %s, %s::jsonb);
                        """,
                        (
                            user_id,
                            "wax_wizard_apply",
                            json.dumps(
                                {
                                    "raw_text": raw_text,
                                    "parsed_payload": parsed_payload,
                                    "actions": actions,
                                    "results": results,
                                }
                            ),
                        ),
                    )

                conn.commit()
                return results
            except Exception as e:
                conn.rollback()
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO vinyl.vault_action_audit (user_id, action_type, payload)
                            VALUES (%s, %s, %s::jsonb);
                            """,
                            (
                                user_id,
                                "wax_wizard_error",
                                json.dumps(
                                    {
                                        "error": str(e),
                                        "raw_text": raw_text,
                                        "parsed_payload": parsed_payload,
                                        "actions": actions,
                                    }
                                ),
                            ),
                        )
                    conn.commit()
                except Exception:
                    pass
                raise
