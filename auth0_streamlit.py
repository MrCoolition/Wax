from __future__ import annotations

import base64
import hmac
import hashlib
import json
import secrets
import time
import tomllib
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import requests
import streamlit as st
from jose import jwt


@dataclass(frozen=True)
class Auth0Config:
    domain: str
    client_id: str
    client_secret: str
    redirect_uri: str
    logout_redirect_uri: str
    audience: Optional[str] = None


class AuthError(RuntimeError):
    pass


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def _auth0_log(event: str, **context: Any) -> None:
    logs = st.session_state.setdefault("auth0_debug_log", [])
    if not isinstance(logs, list):
        logs = []
        st.session_state["auth0_debug_log"] = logs
    safe_context = {}
    for key, value in context.items():
        if value is None:
            safe_context[key] = None
            continue
        text = str(value)
        if "secret" in key or "token" in key:
            safe_context[key] = f"[redacted:{len(text)}]"
        else:
            safe_context[key] = text
    logs.append({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "event": event, "context": safe_context})


def get_auth0_debug_log() -> list[dict]:
    logs = st.session_state.get("auth0_debug_log", [])
    return logs if isinstance(logs, list) else []


def _pkce_pair() -> Tuple[str, str]:
    verifier = _b64url(secrets.token_bytes(32))
    challenge = _b64url(hashlib.sha256(verifier.encode("utf-8")).digest())
    return verifier, challenge


def _issuer(domain: str) -> str:
    domain = domain.strip()
    if domain.startswith("http://") or domain.startswith("https://"):
        domain = domain.split("://", 1)[1]
    return f"https://{domain}/"


def _authorize_url(cfg: Auth0Config, *, state: str, nonce: str, code_challenge: str) -> str:
    base = f"https://{cfg.domain}/authorize"
    scope = "openid profile email"
    params = {
        "response_type": "code",
        "client_id": cfg.client_id,
        "redirect_uri": cfg.redirect_uri,
        "scope": scope,
        "state": state,
        "nonce": nonce,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }
    if cfg.audience:
        params["audience"] = cfg.audience

    qs = "&".join(f"{k}={requests.utils.quote(str(v), safe='')}" for k, v in params.items())
    return f"{base}?{qs}"


def _token_exchange(cfg: Auth0Config, *, code: str, code_verifier: str) -> Dict[str, Any]:
    url = f"https://{cfg.domain}/oauth/token"
    payload = {
        "grant_type": "authorization_code",
        "client_id": cfg.client_id,
        "code": code,
        "redirect_uri": cfg.redirect_uri,
        "code_verifier": code_verifier,
    }
    if cfg.client_secret:
        payload["client_secret"] = cfg.client_secret

    resp = requests.post(url, json=payload, timeout=20)
    if resp.status_code >= 400:
        _auth0_log("token_exchange_failed", status=resp.status_code, body=resp.text[:400])
        raise AuthError(f"Token exchange failed: {resp.status_code} {resp.text}")
    return resp.json()


@st.cache_data(show_spinner=False, ttl=3600)
def _jwks(domain: str) -> Dict[str, Any]:
    url = f"https://{domain}/.well-known/jwks.json"
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    return resp.json()


def _verify_id_token(cfg: Auth0Config, id_token: str, *, expected_nonce: str) -> Dict[str, Any]:
    if not id_token:
        raise AuthError("Missing id_token")

    header = jwt.get_unverified_header(id_token)
    kid = header.get("kid")
    if not kid:
        raise AuthError("id_token missing kid")

    jwks = _jwks(cfg.domain)
    keys = jwks.get("keys", [])
    key = next((k for k in keys if k.get("kid") == kid), None)
    if not key:
        raise AuthError("Unable to find matching JWKS key")

    claims = jwt.decode(
        id_token,
        key,
        algorithms=["RS256"],
        audience=cfg.client_id,
        issuer=_issuer(cfg.domain),
        options={"verify_at_hash": False},
    )

    nonce = claims.get("nonce")
    if expected_nonce and nonce != expected_nonce:
        raise AuthError("Nonce mismatch")

    now = int(time.time())
    exp = int(claims.get("exp", 0) or 0)
    if exp and now > exp:
        raise AuthError("Token expired")

    return claims


def _get_query_params() -> Dict[str, Any]:
    try:
        return dict(st.query_params)
    except Exception:
        return st.experimental_get_query_params()


def _clear_query_params() -> None:
    try:
        st.query_params.clear()
    except Exception:
        st.experimental_set_query_params()


def _parse_auth0_blob(blob: str) -> Dict[str, str]:
    cleaned = (blob or "").strip()
    if not cleaned:
        return {}
    cleaned = cleaned.replace("\\n", "\n")
    try:
        parsed = tomllib.loads(cleaned)
    except tomllib.TOMLDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(k): str(v) for k, v in parsed.items() if v is not None}


def _get_auth0_secret(key: str) -> str:
    raw = st.secrets.get(key, "")
    if isinstance(raw, str):
        return raw.strip()
    return str(raw or "").strip()


def _state_secret(cfg: Auth0Config) -> str:
    return cfg.client_secret or cfg.client_id


def _encode_state(payload: Dict[str, Any], secret: str) -> str:
    data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    b64 = _b64url(data)
    sig = hmac.new(secret.encode("utf-8"), b64.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{b64}.{sig}"


def _decode_state(state: str, secret: str) -> Dict[str, Any]:
    if not state or "." not in state:
        return {}
    b64, sig = state.split(".", 1)
    expected = hmac.new(secret.encode("utf-8"), b64.encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected):
        _auth0_log("state_signature_mismatch")
        return {}
    try:
        padding = "=" * (-len(b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(b64 + padding).decode("utf-8"))
    except Exception:
        _auth0_log("state_decode_failed")
        return {}
    return payload if isinstance(payload, dict) else {}


def get_auth0_config() -> Auth0Config:
    domain = _get_auth0_secret("auth0_domain")
    client_id = _get_auth0_secret("auth0_client_id")
    client_secret = _get_auth0_secret("auth0_client_secret")
    redirect_uri = _get_auth0_secret("auth0_redirect_uri")
    logout_redirect_uri = _get_auth0_secret("auth0_logout_redirect_uri")
    audience = _get_auth0_secret("auth0_audience") or None

    if not (domain and client_id and redirect_uri and logout_redirect_uri):
        blob = _get_auth0_secret("auth0_secrets") or _get_auth0_secret("auth0_domain")
        parsed = _parse_auth0_blob(blob)
        if parsed:
            domain = domain or parsed.get("auth0_domain", "")
            client_id = client_id or parsed.get("auth0_client_id", "")
            client_secret = client_secret or parsed.get("auth0_client_secret", "")
            redirect_uri = redirect_uri or parsed.get("auth0_redirect_uri", "")
            logout_redirect_uri = logout_redirect_uri or parsed.get("auth0_logout_redirect_uri", "")
            audience = audience or parsed.get("auth0_audience") or None

    if not (domain and client_id and redirect_uri and logout_redirect_uri):
        _auth0_log(
            "auth0_secrets_missing",
            has_domain=bool(domain),
            has_client_id=bool(client_id),
            has_redirect_uri=bool(redirect_uri),
            has_logout_redirect_uri=bool(logout_redirect_uri),
            has_client_secret=bool(client_secret),
        )
        raise AuthError(
            "Auth0 secrets missing. Required: auth0_domain, auth0_client_id, "
            "auth0_redirect_uri, auth0_logout_redirect_uri. "
            "Also recommended: auth0_client_secret."
        )

    return Auth0Config(
        domain=domain,
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        logout_redirect_uri=logout_redirect_uri,
        audience=audience,
    )


def logout_url(cfg: Auth0Config) -> str:
    base = f"https://{cfg.domain}/v2/logout"
    params = {
        "client_id": cfg.client_id,
        "returnTo": cfg.logout_redirect_uri,
    }
    qs = "&".join(f"{k}={requests.utils.quote(str(v), safe='')}" for k, v in params.items())
    return f"{base}?{qs}"


def _start_login_flow(cfg: Auth0Config, *, message: str = "Log in to access your Vinyl Vault.") -> None:
    verifier, challenge = _pkce_pair()
    login_state = secrets.token_urlsafe(24)
    nonce = secrets.token_urlsafe(24)

    st.session_state["auth0_code_verifier"] = verifier
    st.session_state["auth0_state"] = login_state
    st.session_state["auth0_nonce"] = nonce

    state_payload = {
        "v": verifier,
        "n": nonce,
        "r": login_state,
        "t": int(time.time()),
    }
    encoded_state = _encode_state(state_payload, _state_secret(cfg))

    st.session_state["auth0_state"] = encoded_state

    url = _authorize_url(cfg, state=encoded_state, nonce=nonce, code_challenge=challenge)

    _auth0_log("login_flow_started", has_client_secret=bool(cfg.client_secret), redirect_uri=cfg.redirect_uri)
    st.info(message)
    if hasattr(st, "link_button"):
        st.link_button("Log in with Auth0", url, use_container_width=True)
    else:
        st.markdown(f"[Log in with Auth0]({url})")
    st.stop()


def require_auth0_login() -> Dict[str, Any]:
    cfg = get_auth0_config()

    if "auth0_claims" in st.session_state and st.session_state["auth0_claims"]:
        _auth0_log("auth0_claims_cached")
        return st.session_state["auth0_claims"]

    qp = _get_query_params()

    if "error" in qp:
        err = qp.get("error")
        desc = qp.get("error_description", "")
        _auth0_log("auth0_error", error=err, description=desc)
        raise AuthError(f"Auth0 error: {err} {desc}")

    code = qp.get("code")
    state = qp.get("state")

    if not code:
        _auth0_log("auth0_no_code", query_params=list(qp.keys()))
        _start_login_flow(cfg)

    expected_state = st.session_state.get("auth0_state")
    if not expected_state:
        _auth0_log("auth0_missing_state", has_code=bool(code), has_state=bool(state))
        decoded = _decode_state(str(state or ""), _state_secret(cfg))
        if decoded:
            st.session_state["auth0_code_verifier"] = decoded.get("v", "")
            st.session_state["auth0_nonce"] = decoded.get("n", "")
            st.session_state["auth0_state"] = state
            expected_state = state
        else:
            _clear_query_params()
            st.session_state.pop("auth0_code_verifier", None)
            st.session_state.pop("auth0_state", None)
            st.session_state.pop("auth0_nonce", None)
            _start_login_flow(cfg)
    if state != expected_state:
        _clear_query_params()
        st.session_state.pop("auth0_code_verifier", None)
        st.session_state.pop("auth0_state", None)
        st.session_state.pop("auth0_nonce", None)
        _start_login_flow(cfg, message="Your login session expired. Please sign in again.")

    verifier = st.session_state.get("auth0_code_verifier", "")
    nonce = st.session_state.get("auth0_nonce", "")

    _auth0_log("auth0_token_exchange_start", has_verifier=bool(verifier))
    tokens = _token_exchange(cfg, code=code, code_verifier=verifier)
    id_token = tokens.get("id_token", "")
    claims = _verify_id_token(cfg, id_token, expected_nonce=nonce)
    _auth0_log("auth0_token_verified", subject=claims.get("sub", ""))

    st.session_state["auth0_tokens"] = tokens
    st.session_state["auth0_claims"] = claims

    _clear_query_params()
    st.rerun()

    return claims
