#!/usr/bin/env python3
"""
summarize_and_store.py
- Read chunk JSONL produced by fetch_and_chunk.py
- Call the OpenRouter chat completions API
- For each chunk, produce three summaries: shorten, professional, informal
- Store sentence-level summaries to JSONL

Usage:
    echo "OPENROUTER_API_KEY=..." > .env
    python summarize_and_store.py --infile chunks.jsonl.gz --outfile summaries.jsonl.gz --model openai/gpt-4o-mini

Requirements:
    pip install requests tqdm backoff python-dotenv
Notes:
    - Chat endpoints typically do NOT return token logprobs. If you need token logprobs, use completion endpoints that expose them.
    - The script loads OPENROUTER_API_KEY from a .env file in the project root.
"""

import argparse
import gzip
import json
import os
from pathlib import Path
import ssl
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import backoff
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")
openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
DEFAULT_USER_AGENT = "summarizer-uncertainty-ml/0.1 (research script; contact: local-dev)"
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_SITE_URL = os.environ.get("OPENROUTER_SITE_URL")
OPENROUTER_APP_NAME = os.environ.get("OPENROUTER_APP_NAME", "summarizer-uncertainty-ml")

SUMMARY_STYLES = ("shorten", "professional", "informal")

PROMPT_TEMPLATES = {
    "shorten": (
        "Shorten the paragraph below into 2-3 concise sentences that preserve all key facts.\n"
        "Return only the shortened text and nothing else.\n\n"
        "Paragraph:\n{paragraph}"
    ),
    "professional": (
        "Write a professional 2-3 sentence summary of the paragraph below, "
        "suitable for an encyclopedia or formal report.\n"
        "Return only the summary text and nothing else.\n\n"
        "Paragraph:\n{paragraph}"
    ),
    "informal": (
        "Summarize the paragraph below in 2-3 casual, conversational sentences "
        "as if explaining it to a friend.\n"
        "Return only the summary text and nothing else.\n\n"
        "Paragraph:\n{paragraph}"
    ),
}


class SSLContextAdapter(HTTPAdapter):
    """Requests adapter that uses a caller-provided SSL context."""

    def __init__(self, ssl_context: ssl.SSLContext) -> None:
        self.ssl_context = ssl_context
        super().__init__()

    def init_poolmanager(self, *args, **kwargs):
        kwargs["ssl_context"] = self.ssl_context
        return super().init_poolmanager(*args, **kwargs)

    def proxy_manager_for(self, *args, **kwargs):
        kwargs["ssl_context"] = self.ssl_context
        return super().proxy_manager_for(*args, **kwargs)


# ----------------------------
# Helpers
# ----------------------------
def build_session(cert_path=None):
    """Build a requests session, optionally using a custom CA bundle."""

    session = requests.Session()
    session.headers.update({"User-Agent": DEFAULT_USER_AGENT})
    if cert_path:
        ssl_context = ssl.create_default_context()
        if hasattr(ssl, "VERIFY_X509_STRICT"):
            ssl_context.verify_flags &= ~ssl.VERIFY_X509_STRICT
        ssl_context.load_verify_locations(cafile=cert_path)
        adapter = SSLContextAdapter(ssl_context)
        session.mount("https://", adapter)
    return session


def read_jsonl(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def write_jsonl(path, obj):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "at", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ----------------------------
# API caller (with backoff)
# ----------------------------
@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def call_openrouter_chat(prompt, model, temperature, max_tokens=256, api_key=None, session=None):
    key = (api_key or os.environ.get("OPENROUTER_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    request_session = session or build_session()
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    if OPENROUTER_SITE_URL:
        headers["HTTP-Referer"] = OPENROUTER_SITE_URL
    if OPENROUTER_APP_NAME:
        headers["X-Title"] = OPENROUTER_APP_NAME
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "n": 1,
    }
    r = request_session.post(OPENROUTER_ENDPOINT, json=payload, headers=headers, timeout=60)
    if r.status_code >= 400:
        raise RuntimeError(f"OpenRouter error {r.status_code}: {r.text[:1000]}")
    return r.json()


# ----------------------------
# Worker
# ----------------------------
def summarize_chunk(chunk_obj, style, model="openai/gpt-4o-mini", temperature=0.7, api_key=None, max_tokens=200, session=None):
    paragraph = chunk_obj["paragraph_text"]
    prompt = PROMPT_TEMPLATES[style].format(paragraph=paragraph)
    try:
        raw = call_openrouter_chat(
            prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            session=session,
        )
        choices = raw.get("choices") or []
        text = choices[0].get("message", {}).get("content", "").strip() if choices else ""
    except Exception as e:
        return {"error": str(e), "id": chunk_obj["id"], "summary_style": style, "chunk_meta": chunk_obj}

    return {
        "id": f"{chunk_obj['id']}|{style}",
        "page_title": chunk_obj.get("page_title"),
        "paragraph_text": paragraph,
        "summary": text,
        "summary_style": style,
        "model": model,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# ----------------------------
# Main
# ----------------------------
def main(args):
    session = build_session(cert_path=args.ssl_cert)
    pool = ThreadPoolExecutor(max_workers=args.workers)
    futures = []

    for chunk_count, chunk in enumerate(read_jsonl(args.infile)):
        if args.n_max is not None and chunk_count >= args.n_max:
            break

        for style in SUMMARY_STYLES:
            future = pool.submit(
                summarize_chunk,
                chunk,
                style,
                args.model,
                args.temperature,
                openrouter_api_key,
                args.max_tokens,
                session,
            )
            futures.append(future)

            if len(futures) > args.max_queue:
                for f in as_completed(futures, timeout=None):
                    res = f.result()
                    write_jsonl(args.outfile, res)
                    futures.remove(f)
                    break

    for f in tqdm(futures, desc="summarizing", total=len(futures)):
        try:
            res = f.result()
        except Exception as e:
            res = {"error": str(e)}
        write_jsonl(args.outfile, res)

    pool.shutdown(wait=True)
    print("Finished. Wrote results to", args.outfile)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--infile", required=True, help="input chunks JSONL (gz ok)")
    p.add_argument("--outfile", required=True, help="output JSONL (gz ok)")
    p.add_argument("--model", default="openai/gpt-4o-mini")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--workers", type=int, default=4, help="concurrent worker threads")
    p.add_argument("--max_queue", type=int, default=200, help="max pending futures before throttling")
    p.add_argument("--max_tokens", type=int, default=200)
    p.add_argument("--n-max", type=int, help="maximum number of input paragraphs to process")
    p.add_argument("--ssl-cert", help="path to a CA bundle to use for HTTPS verification")
    args = p.parse_args()
    main(args)
