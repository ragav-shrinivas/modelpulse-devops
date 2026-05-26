"""
Multi-Strategy Chat Share-Link Scraper
======================================

Why this module exists:
    Claude/ChatGPT share pages are SPAs with anti-bot fingerprinting,
    stale class names, and streaming content. The original implementation
    used Playwright + 4 stale CSS selectors and failed silently 90% of
    the time (falling back to randomised demo data the user mistook
    for a real analysis).

Strategy cascade (each falls back to the next):
    1. requests + __NEXT_DATA__ JSON parse        ← fast, no browser
    2. requests + regex over inline HTML payload  ← server-rendered shells
    3. Playwright with stealth tweaks + multiple
       selector families + 12 s settle wait      ← live SPA scrape
    4. PASTE-MODE — caller passes in the messages
       directly via /analyze-chat-text            ← always works

Each strategy returns:
    {"messages": [...], "strategy": <str>, "ok": bool, "error": <str|None>}
"""

from __future__ import annotations
import json
import re
import logging
from typing import Dict, List, Optional

log = logging.getLogger(__name__)


# ---------- strategy 1: requests + __NEXT_DATA__ ----------

def _try_requests_nextdata(url: str) -> Optional[List[str]]:
    try:
        import requests
    except ImportError:
        return None
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/605.1.15 (KHTML, like Gecko) "
                "Version/17.0 Safari/605.1.15"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code != 200:
            return None
        # Look for __NEXT_DATA__ JSON blob
        m = re.search(
            r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
            r.text, re.DOTALL
        )
        if not m:
            return None
        data = json.loads(m.group(1))
        return _walk_for_messages(data)
    except Exception as e:
        log.warning("requests+__NEXT_DATA__ failed: %s", e)
        return None


def _walk_for_messages(obj, found=None) -> List[str]:
    """
    Recursively walk a JSON tree pulled from share pages and collect
    plausible assistant messages. Both Claude and ChatGPT use varying
    shapes; we look for objects with role/sender == 'assistant' and
    a 'text', 'content', 'message', or 'parts' field.
    """
    if found is None:
        found = []
    if isinstance(obj, dict):
        role = (obj.get("role") or obj.get("sender") or
                obj.get("author", {}).get("role") if isinstance(obj.get("author"), dict) else None)
        if role == "assistant":
            text = _extract_text_field(obj)
            if text and len(text.strip()) > 20:
                found.append(text.strip())
        for v in obj.values():
            _walk_for_messages(v, found)
    elif isinstance(obj, list):
        for v in obj:
            _walk_for_messages(v, found)
    return found


def _extract_text_field(obj: Dict) -> str:
    for k in ("text", "content", "message"):
        v = obj.get(k)
        if isinstance(v, str):
            return v
        if isinstance(v, dict) and "parts" in v and isinstance(v["parts"], list):
            return " ".join(p for p in v["parts"] if isinstance(p, str))
        if isinstance(v, list):
            parts = []
            for item in v:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    for k2 in ("text", "content"):
                        if isinstance(item.get(k2), str):
                            parts.append(item[k2])
            if parts:
                return " ".join(parts)
    return ""


# ---------- strategy 2: regex over inline HTML ----------

_INLINE_MSG_PAT = re.compile(
    r'"(?:text|content|message_text)"\s*:\s*"((?:[^"\\]|\\.){50,})"'
)


def _try_requests_regex(url: str) -> Optional[List[str]]:
    try:
        import requests
    except ImportError:
        return None
    try:
        r = requests.get(url, timeout=15, headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
        })
        if r.status_code != 200:
            return None
        raw_matches = _INLINE_MSG_PAT.findall(r.text)
        # unescape \n and similar
        cleaned = []
        for m in raw_matches:
            try:
                cleaned.append(json.loads(f'"{m}"'))
            except Exception:
                cleaned.append(m)
        # dedupe while preserving order
        seen = set()
        out = []
        for m in cleaned:
            if m not in seen and len(m) > 30:
                seen.add(m)
                out.append(m)
        return out or None
    except Exception as e:
        log.warning("requests+regex failed: %s", e)
        return None


# ---------- strategy 3: Playwright with stealth ----------

_CLAUDE_SELECTORS = [
    # current (2024-2025)
    "[data-testid='message-content']",
    "div.font-claude-message",
    "div.font-claude-response",
    # older
    "[data-testid='assistant-message'] .prose",
    ".assistant-message .prose",
    "[class*='assistant'] [class*='prose']",
    "[class*='AssistantMessage']",
    # generic fallback
    "main article",
]

_CHATGPT_SELECTORS = [
    "[data-message-author-role='assistant']",
    "[data-message-author-role='assistant'] .markdown",
    ".agent-turn .markdown",
    "[class*='assistant'] .markdown",
    "div[data-testid^='conversation-turn']",
    "main article",
]


def _try_playwright(url: str, platform: str) -> Optional[List[str]]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                ],
            )
            ctx = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/605.1.15 (KHTML, like Gecko) "
                    "Version/17.0 Safari/605.1.15"
                ),
                viewport={"width": 1440, "height": 900},
                locale="en-US",
            )
            # Stealth: spoof navigator.webdriver
            ctx.add_init_script(
                "Object.defineProperty(navigator, 'webdriver', "
                "{ get: () => undefined });"
            )
            page = ctx.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            # Wait for either networkidle or 12 s, whichever comes first
            try:
                page.wait_for_load_state("networkidle", timeout=12000)
            except Exception:
                pass
            page.wait_for_timeout(2500)

            selectors = (_CLAUDE_SELECTORS if platform == "claude"
                         else _CHATGPT_SELECTORS)
            messages: List[str] = []
            for sel in selectors:
                els = page.query_selector_all(sel)
                if els:
                    candidates = [
                        el.inner_text().strip()
                        for el in els
                        if el.inner_text().strip()
                    ]
                    # Keep the longest plausible set
                    if candidates and any(len(c) > 30 for c in candidates):
                        messages = candidates
                        break
            browser.close()
            return messages or None
    except Exception as e:
        log.warning("playwright failed: %s", e)
        return None


# ---------- orchestrator ----------

def detect_platform(url: str) -> Optional[str]:
    if "claude.ai/share" in url:
        return "claude"
    if "chatgpt.com/share" in url or "chat.openai.com/share" in url:
        return "chatgpt"
    return None


def scrape(url: str) -> Dict:
    """
    Main entry. Walks the strategy cascade. Always returns a dict; the
    'strategy' field tells the caller which path succeeded.
    """
    platform = detect_platform(url)
    if not platform:
        return {"ok": False, "error": "Unsupported URL (need claude.ai/share/ or chatgpt.com/share/)",
                "messages": [], "platform": None, "strategy": None}

    # Strategy 1
    msgs = _try_requests_nextdata(url)
    if msgs:
        return {"ok": True, "error": None, "messages": msgs,
                "platform": platform, "strategy": "requests+nextdata"}

    # Strategy 2
    msgs = _try_requests_regex(url)
    if msgs:
        return {"ok": True, "error": None, "messages": msgs,
                "platform": platform, "strategy": "requests+regex"}

    # Strategy 3
    msgs = _try_playwright(url, platform)
    if msgs:
        return {"ok": True, "error": None, "messages": msgs,
                "platform": platform, "strategy": "playwright+stealth"}

    return {
        "ok": False,
        "error": ("All scrape strategies failed. "
                  "Use /analyze-chat-text to paste the conversation directly."),
        "messages": [],
        "platform": platform,
        "strategy": None,
    }
