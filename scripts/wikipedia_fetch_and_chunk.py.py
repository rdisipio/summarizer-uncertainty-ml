#!/usr/bin/env python3
"""
fetch_and_chunk.py
- Fetch Wikipedia pages via API (or read a text file list)
- Clean text, split into sentences, and pack into ~250-word chunks preserving sentence boundaries
- Output: JSONL file, one object per chunk with provenance metadata

Usage:
    python fetch_and_chunk.py --titles-file titles.txt --out chunks.jsonl.gz
Requirements:
    pip install requests nltk tqdm
"""

import argparse
import gzip
import json
import ssl
import time

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from nltk import sent_tokenize, download
download('punkt')  # ensure punkt is available

DEFAULT_WORD_TARGET = 250
DEFAULT_USER_AGENT = "summarizer-uncertainty-ml/0.1 (research script; contact: local-dev)"


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

def fetch_wikipedia_extract(title, session=None, retries=3, wait=1.0):
    session = session or build_session()
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "format": "json",
        "titles": title,
        "redirects": 1,
        "formatversion": 2
    }
    for attempt in range(retries):
        try:
            r = session.get("https://en.wikipedia.org/w/api.php", params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            pages = data.get("query", {}).get("pages", [])
            if not pages:
                return "", None
            page = pages[0]
            extract = page.get("extract", "") or ""
            pageid = page.get("pageid")
            url = f"https://en.wikipedia.org/?curid={pageid}" if pageid else None
            return extract, url
        except Exception as e:
            if attempt + 1 == retries:
                raise
            time.sleep(wait * (2 ** attempt))
    return "", None

def chunk_text_preserve_sentences(text, approx_words=DEFAULT_WORD_TARGET):
    if not text or not text.strip():
        return []
    sents = sent_tokenize(text)
    chunks = []
    cur = []
    cur_words = 0
    for s in sents:
        w = len(s.split())
        if cur and (cur_words + w > approx_words):
            chunks.append(" ".join(cur))
            cur = [s]
            cur_words = w
        else:
            cur.append(s)
            cur_words += w
    if cur:
        chunks.append(" ".join(cur))
    return chunks

def read_titles_file(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t:
                yield t

def main(args):
    session = build_session(cert_path=args.ssl_cert)
    out_f = gzip.open(args.out, "wt", encoding="utf-8") if args.out.endswith(".gz") else open(args.out, "w", encoding="utf-8")
    titles = list(read_titles_file(args.titles_file))
    for title in tqdm(titles, desc="titles"):
        try:
            text, url = fetch_wikipedia_extract(title, session=session)
        except Exception as e:
            print(f"[WARN] failed to fetch {title}: {e}")
            continue
        if not text:
            continue
        chunks = chunk_text_preserve_sentences(text, approx_words=args.approx_words)
        for i, chunk in enumerate(chunks):
            obj = {
                "id": f"wikipedia|{title.replace(' ','_')}|chunk_{i:04d}",
                "page_title": title,
                "source_url": url,
                "paragraph_idx": i,
                "paragraph_text": chunk,
                "paragraph_word_count": len(chunk.split()),
                "fetched_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
            out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    out_f.close()
    print("Done. Written to", args.out)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--titles-file", required=True, help="one Wikipedia title per line (e.g. 'Paris', 'Machine learning')")
    p.add_argument("--out", required=True, help="output JSONL path (use .gz to compress)")
    p.add_argument("--approx-words", type=int, default=DEFAULT_WORD_TARGET, help="target words per chunk")
    p.add_argument("--ssl-cert", help="path to a CA bundle to use for HTTPS verification")
    args = p.parse_args()
    main(args)
