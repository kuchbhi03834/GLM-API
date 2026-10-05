"""
Authentication — Playwright login & session capture for ChatGLM / Zhipu AI.

Supports two modes:
1. Official API Key mode: If `ZHIPU_API_KEY` is set, uses it directly (no browser needed).
2. Web Chat mode: Launches Playwright to capture the session from https://chatglm.cn/
   and saves it under `session/session.json`. Later runs refresh headlessly.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROFILE_DIR = Path(os.getenv("GLM_PROFILE_DIR", ROOT / "session" / "profile"))
DEFAULT_SESSION_FILE = ROOT / "session" / "session.json"

CHAT_URL = "https://chatglm.cn/"
SIGNIN_URL = "https://chatglm.cn/login"
LAUNCH_ARGS = ["--disable-blink-features=AutomationControlled"]
SESSION_MAX_AGE = 12 * 60 * 60  # 12 hours


class LoginRequired(RuntimeError):
    DEFAULT = (
        "No active GLM session or ZHIPU_API_KEY found. Log in first by running:\n"
        "    python -m glm.auth\n"
        "or set ZHIPU_API_KEY in your .env file."
    )

    def __init__(self, message: str = DEFAULT):
        super().__init__(message)


@dataclass
class Session:
    token: str
    cookies: Dict[str, str]
    user_agent: str
    captured_at: float
    is_api_key: bool = False

    @property
    def age(self) -> float:
        return time.time() - self.captured_at

    def save(self, path: Path = DEFAULT_SESSION_FILE) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path = DEFAULT_SESSION_FILE) -> Optional["Session"]:
        if not path.exists():
            return None
        try:
            return cls(**json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            return None


_READ_TOKEN_JS = """
() => {
  try {
    for (const key of ['chatglm_token', 'token', 'access_token', 'Authorization']) {
      const val = window.localStorage.getItem(key);
      if (val) {
        try {
          const parsed = JSON.parse(val);
          if (parsed && typeof parsed === 'object') {
            return parsed.token || parsed.access_token || parsed.value || val;
          }
        } catch (e) {}
        return val;
      }
    }
    return null;
  } catch (e) { return null; }
}
"""

_auth_lock = threading.Lock()


def _safe_evaluate(page, js: str):
    try:
        return page.evaluate(js)
    except Exception as e:
        msg = str(e).lower()
        if "execution context was destroyed" in msg or "navigation" in msg:
            return None
        raise


def _capture_from_context(context, page) -> Optional[Session]:
    token = _safe_evaluate(page, _READ_TOKEN_JS)
    cookies = {c["name"]: c["value"] for c in context.cookies()}
    if not token:
        # Fallback to cookie tokens
        token = cookies.get("chatglm_token") or cookies.get("chatglm_refresh_token") or cookies.get("access_token")
    if not token:
        return None
    ua = _safe_evaluate(page, "() => navigator.userAgent") or ""
    return Session(token=token, cookies=cookies, user_agent=ua, captured_at=time.time())


def _safe_goto(page, url: str) -> None:
    try:
        page.goto(url, wait_until="commit", timeout=60000)
    except Exception as e:
        print(f"[auth] Navigation notice ({type(e).__name__}): continuing...")
    page.wait_for_timeout(2000)


def login(
    profile_dir: Path = DEFAULT_PROFILE_DIR,
    headless: bool = False,
) -> Session:
    """Interactive login for chatglm.cn. Opens browser window once to capture session."""
    from playwright.sync_api import sync_playwright

    with _auth_lock:
        profile_dir.mkdir(parents=True, exist_ok=True)
        with sync_playwright() as p:
            try:
                context = p.chromium.launch_persistent_context(
                    str(profile_dir), headless=headless, channel="chrome", args=LAUNCH_ARGS,
                )
            except Exception:
                context = p.chromium.launch_persistent_context(
                    str(profile_dir), headless=headless, args=LAUNCH_ARGS,
                )
            page = context.pages[0] if context.pages else context.new_page()

            _safe_goto(page, SIGNIN_URL)
            print("[auth] Please sign in in the browser window. Waiting for session token...")

            deadline = time.time() + 300
            session = None
            while time.time() < deadline:
                session = _capture_from_context(context, page)
                if session and session.token:
                    break
                page.wait_for_timeout(1000)

            context.close()
            if not session or not session.token:
                raise RuntimeError("Login timed out or no token was captured.")

            session.save()
            print(f"[auth] Successfully captured GLM session token!")
            return session


def refresh_session(
    profile_dir: Path = DEFAULT_PROFILE_DIR,
    session_file: Path = DEFAULT_SESSION_FILE,
) -> Optional[Session]:
    """Attempt thread-safe headless background refresh using cached profile."""
    # Check if API key is in environment
    api_key = os.getenv("ZHIPU_API_KEY", "").strip() or os.getenv("GLM_API_KEY", "").strip()
    if api_key:
        s = Session(token=api_key, cookies={}, user_agent="GLM-API/1.0", captured_at=time.time(), is_api_key=True)
        s.save(session_file)
        return s

    with _auth_lock:
        cached = Session.load(session_file)
        if cached and cached.age < 30.0:
            return cached

        from playwright.sync_api import sync_playwright

        try:
            profile_dir.mkdir(parents=True, exist_ok=True)
            with sync_playwright() as p:
                try:
                    context = p.chromium.launch_persistent_context(
                        str(profile_dir), headless=True, channel="chrome", args=LAUNCH_ARGS,
                    )
                except Exception:
                    context = p.chromium.launch_persistent_context(
                        str(profile_dir), headless=True, args=LAUNCH_ARGS,
                    )
                page = context.pages[0] if context.pages else context.new_page()
                try:
                    _safe_goto(page, CHAT_URL)
                    session = _capture_from_context(context, page)
                finally:
                    context.close()

            if session is not None:
                session.save(session_file)
                return session
        except Exception as e:
            print(f"[auth] Headless refresh error: {e}")
        return None


def get_session(
    profile_dir: Path = DEFAULT_PROFILE_DIR,
    session_file: Path = DEFAULT_SESSION_FILE,
    max_age: int = SESSION_MAX_AGE,
    allow_interactive: bool = True,
) -> Session:
    """Return a usable session: env API key -> fresh cached session -> headless refresh -> interactive."""
    # 1. Environment API Key
    api_key = os.getenv("ZHIPU_API_KEY", "").strip() or os.getenv("GLM_API_KEY", "").strip()
    if api_key:
        return Session(token=api_key, cookies={}, user_agent="GLM-API/1.0", captured_at=time.time(), is_api_key=True)

    # 2. Cached session
    cached = Session.load(session_file)
    if cached and cached.age < max_age:
        return cached

    # 3. Headless refresh
    session = refresh_session(profile_dir, session_file)
    if session is not None:
        return session

    # 4. Interactive login
    if not allow_interactive:
        raise LoginRequired()

    print("[auth] No valid session or API key found. Launching browser to sign in...")
    return login(profile_dir=profile_dir)


if __name__ == "__main__":
    s = login()
    print(f"[auth] Captured token: {s.token[:12]}...")
