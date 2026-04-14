"""Shared data pipeline utilities: JSONL I/O, Wikipedia fetch, OpenRouter summarization.

Used by both the CLI scripts (scripts/wikipedia_fetch_and_chunk.py,
scripts/summarize_and_store.py) and the Jupyter notebooks so that all
data-ingestion logic lives in one place.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import ssl
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Generator, Iterable

import backoff
import requests
from requests.adapters import HTTPAdapter

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = "summarizer-uncertainty-ml/0.1 (research; contact: local-dev)"
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

PROMPT_TEMPLATES: dict[str, str] = {
    "shorten": (
        "Shorten the paragraph below into 2-3 concise sentences that preserve all key facts.\n"
        "Return only the shortened text and nothing else.\n\nParagraph:\n{paragraph}"
    ),
    "professional": (
        "Write a professional 2-3 sentence summary of the paragraph below, "
        "suitable for an encyclopedia or formal report.\n"
        "Return only the summary text and nothing else.\n\nParagraph:\n{paragraph}"
    ),
    "informal": (
        "Summarize the paragraph below in 2-3 casual, conversational sentences "
        "as if explaining it to a friend.\n"
        "Return only the summary text and nothing else.\n\nParagraph:\n{paragraph}"
    ),
}

SUMMARY_STYLES: tuple[str, ...] = tuple(PROMPT_TEMPLATES)


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------


def read_jsonl(path: str) -> Generator[dict, None, None]:
    """Yield parsed JSON objects from a plain or gzip-compressed JSONL file."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str, obj: dict) -> None:
    """Append one JSON object as a line to a plain or gzip-compressed JSONL file."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "at", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# HTTP session
# ---------------------------------------------------------------------------


class _SSLContextAdapter(HTTPAdapter):
    def __init__(self, ssl_context: ssl.SSLContext) -> None:
        self.ssl_context = ssl_context
        super().__init__()

    def init_poolmanager(self, *args, **kwargs):
        kwargs["ssl_context"] = self.ssl_context
        return super().init_poolmanager(*args, **kwargs)

    def proxy_manager_for(self, *args, **kwargs):
        kwargs["ssl_context"] = self.ssl_context
        return super().proxy_manager_for(*args, **kwargs)


def build_session(cert_path: str | None = None) -> requests.Session:
    """Return a requests Session with a standard User-Agent and optional CA bundle."""
    session = requests.Session()
    session.headers.update({"User-Agent": DEFAULT_USER_AGENT})
    if cert_path:
        ctx = ssl.create_default_context()
        if hasattr(ssl, "VERIFY_X509_STRICT"):
            ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
        ctx.load_verify_locations(cafile=cert_path)
        session.mount("https://", _SSLContextAdapter(ctx))
    return session


# ---------------------------------------------------------------------------
# Wikipedia
# ---------------------------------------------------------------------------


def fetch_wikipedia_extract(
    title: str,
    session: requests.Session | None = None,
    retries: int = 3,
    wait: float = 1.0,
) -> tuple[str, str | None]:
    """Return (extract_text, article_url) for a Wikipedia article title.

    Returns ("", None) if the page has no extract.
    """
    session = session or build_session()
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "format": "json",
        "titles": title,
        "redirects": 1,
        "formatversion": 2,
    }
    for attempt in range(retries):
        try:
            r = session.get(WIKIPEDIA_API_URL, params=params, timeout=30)
            r.raise_for_status()
            pages = r.json().get("query", {}).get("pages", [])
            if not pages:
                return "", None
            page = pages[0]
            pageid = page.get("pageid")
            url = f"https://en.wikipedia.org/?curid={pageid}" if pageid else None
            return page.get("extract", "") or "", url
        except Exception:
            if attempt + 1 == retries:
                raise
            time.sleep(wait * (2 ** attempt))
    return "", None


def chunk_text(text: str, approx_words: int = 250) -> list[str]:
    """Split *text* into sentence-boundary-aligned chunks of ~approx_words words.

    Requires the NLTK punkt tokenizer to be available.
    """
    from nltk import sent_tokenize

    if not text or not text.strip():
        return []
    sents = sent_tokenize(text)
    chunks: list[str] = []
    cur: list[str] = []
    cur_words = 0
    for s in sents:
        w = len(s.split())
        if cur and cur_words + w > approx_words:
            chunks.append(" ".join(cur))
            cur, cur_words = [s], w
        else:
            cur.append(s)
            cur_words += w
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def fetch_and_chunk_titles(
    titles: Iterable[str],
    approx_words: int = 250,
    session: requests.Session | None = None,
) -> Generator[dict, None, None]:
    """Fetch and chunk Wikipedia articles for a list of titles.

    Yields one dict per chunk with provenance metadata, ready to write to JSONL.
    """
    session = session or build_session()
    for title in titles:
        try:
            text, url = fetch_wikipedia_extract(title, session=session)
        except Exception as e:
            logger.warning("Failed to fetch %r: %s", title, e)
            continue
        if not text:
            logger.warning("Empty extract for %r", title)
            continue
        for i, chunk in enumerate(chunk_text(text, approx_words=approx_words)):
            yield {
                "id": f"wikipedia|{title.replace(' ', '_')}|chunk_{i:04d}",
                "page_title": title,
                "source_url": url,
                "paragraph_idx": i,
                "paragraph_text": chunk,
                "paragraph_word_count": len(chunk.split()),
                "fetched_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }


# ---------------------------------------------------------------------------
# OpenRouter summarization
# ---------------------------------------------------------------------------


@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def call_openrouter(
    prompt: str,
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 256,
    api_key: str | None = None,
    session: requests.Session | None = None,
) -> str:
    """Call the OpenRouter chat completions endpoint and return the reply text."""
    key = (api_key or os.environ.get("OPENROUTER_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "n": 1,
    }
    req_session = session or build_session()
    r = req_session.post(OPENROUTER_ENDPOINT, json=payload, headers=headers, timeout=60)
    if r.status_code >= 400:
        raise RuntimeError(f"OpenRouter {r.status_code}: {r.text[:500]}")
    choices = r.json().get("choices", [])
    return choices[0]["message"]["content"].strip() if choices else ""


def summarize_chunk(
    chunk_obj: dict,
    style: str,
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 200,
    api_key: str | None = None,
    session: requests.Session | None = None,
) -> dict:
    """Generate one style variant summary for a chunk dict.

    Returns a result dict on success, or a dict with an "error" key on failure.
    """
    if style not in PROMPT_TEMPLATES:
        raise ValueError(f"Unknown style {style!r}. Choose from {list(PROMPT_TEMPLATES)}.")
    prompt = PROMPT_TEMPLATES[style].format(paragraph=chunk_obj["paragraph_text"])
    try:
        text = call_openrouter(
            prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            session=session,
        )
    except Exception as e:
        return {"error": str(e), "id": chunk_obj["id"], "summary_style": style}
    return {
        "id": f"{chunk_obj['id']}|{style}",
        "page_title": chunk_obj.get("page_title"),
        "paragraph_text": chunk_obj["paragraph_text"],
        "summary": text,
        "summary_style": style,
        "model": model,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def summarize_chunks(
    chunks: list[dict],
    model: str,
    styles: Iterable[str] = SUMMARY_STYLES,
    temperature: float = 0.7,
    max_tokens: int = 200,
    api_key: str | None = None,
    workers: int = 4,
    session: requests.Session | None = None,
) -> Generator[dict, None, None]:
    """Summarize a list of chunk dicts in all requested styles, concurrently.

    Yields result dicts as they complete. Error dicts (containing "error" key)
    are also yielded so callers can log or skip them.
    """
    styles = list(styles)
    tasks = [(chunk, style) for chunk in chunks for style in styles]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(summarize_chunk, chunk, style, model, temperature,
                        max_tokens, api_key, session): (chunk["id"], style)
            for chunk, style in tasks
        }
        for future in as_completed(futures):
            yield future.result()
