from __future__ import annotations

import base64
import hashlib
import secrets
import time
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


def get_auth0_config() -> Auth0Config:
    domain = (st.secrets.get("auth0_domain", "") or "").strip()
    client_id = (st.secrets.get("auth0_client_id", "") or "").strip()
    client_secret = (st.secrets.get("auth0_client_secret", "") or "").strip()
    redirect_uri = (st.secrets.get("auth0_redirect_uri", "") or "").strip()
    logout_redirect_uri = (st.secrets.get("auth0_logout_redirect_uri", "") or "").strip()
    audience = (st.secrets.get("auth0_audience", "") or "").strip() or None

    if not (domain and client_id and redirect_uri and logout_redirect_uri):
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


def require_auth0_login() -> Dict[str, Any]:
    cfg = get_auth0_config()

    if "auth0_claims" in st.session_state and st.session_state["auth0_claims"]:
        return st.session_state["auth0_claims"]

    qp = _get_query_params()

    if "error" in qp:
        err = qp.get("error")
        desc = qp.get("error_description", "")
        raise AuthError(f"Auth0 error: {err} {desc}")

    code = qp.get("code")
    state = qp.get("state")

    if not code:
        verifier, challenge = _pkce_pair()
        login_state = secrets.token_urlsafe(24)
        nonce = secrets.token_urlsafe(24)

        st.session_state["auth0_code_verifier"] = verifier
        st.session_state["auth0_state"] = login_state
        st.session_state["auth0_nonce"] = nonce

        url = _authorize_url(cfg, state=login_state, nonce=nonce, code_challenge=challenge)

        st.info("Log in to access your Vinyl Vault.")
        st.link_button("Log in with Auth0", url, use_container_width=True)
        st.stop()

    expected_state = st.session_state.get("auth0_state")
    if not expected_state or state != expected_state:
        _clear_query_params()
        raise AuthError("State mismatch. Please try logging in again.")

    verifier = st.session_state.get("auth0_code_verifier", "")
    nonce = st.session_state.get("auth0_nonce", "")

    tokens = _token_exchange(cfg, code=code, code_verifier=verifier)
    id_token = tokens.get("id_token", "")
    claims = _verify_id_token(cfg, id_token, expected_nonce=nonce)

    st.session_state["auth0_tokens"] = tokens
    st.session_state["auth0_claims"] = claims

    _clear_query_params()
    st.rerun()

    return claims
