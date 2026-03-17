#!/usr/bin/env python3
"""
summarize_and_store.py
- Read chunk JSONL produced by fetch_and_chunk.py
- Call the OpenRouter chat completions API
- Store structured results (summary sentences, raw response, prompt, metadata) to JSONL

Usage:
    echo "OPENROUTER_API_KEY=..." > .env
    python summarize_and_store.py --infile chunks.jsonl.gz --outfile summaries.jsonl.gz --model openai/gpt-4o-mini --samples 3 --temperature 0.2

Requirements:
    pip install requests tqdm backoff nltk python-dotenv
Notes:
    - Chat endpoints typically do NOT return token logprobs. If you need token logprobs, use completion endpoints that expose them.
    - The script loads OPENROUTER_API_KEY from a .env file in the project root.
"""

import argparse
import gzip
import json
import os
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import backoff
import requests
from dotenv import load_dotenv
from nltk import sent_tokenize
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")
openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")

# ----------------------------
# Helpers
# ----------------------------
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
# API callers (with backoff)
# ----------------------------
@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def call_openrouter_chat(prompt, model, temperature, max_tokens=256, api_key=None):
    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    url = "https://api.openrouter.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "n": 1
    }
    r = requests.post(url, json=payload, headers=headers, timeout=60)
    r.raise_for_status()
    return r.json()

# ----------------------------
# Prompting helpers
# ----------------------------
PROMPT_TEMPLATE = """Summarize the paragraph below in 2-3 concise factual sentences.
Return JSON exactly in this format:
{{"sentences": ["S1", "S2", ...]}}

Paragraph:
{paragraph}
"""
# ----------------------------
# Worker
# ----------------------------
def summarize_chunk(chunk_obj, model="openai/gpt-4o-mini", temperature=0.0, api_key=None, max_tokens=200):
    paragraph = chunk_obj["paragraph_text"]
    prompt = PROMPT_TEMPLATE.format(paragraph=paragraph)
    start = time.time()
    raw = None
    try:
        raw = call_openrouter_chat(prompt, model=model, temperature=temperature, max_tokens=max_tokens, api_key=api_key)
        choices = raw.get("choices") or []
        text = choices[0].get("message", {}).get("content", "").strip() if choices else ""
        provider_meta = {"provider": "openrouter", "model": model}
    except Exception as e:
        return {"error": str(e), "id": chunk_obj["id"], "chunk_meta": chunk_obj}

    latency_ms = int((time.time() - start) * 1000)

    # Try to parse JSON from model output
    sentences = []
    parse_error = None
    try:
        parsed = json.loads(text)
        sentences = parsed.get("sentences", [])
    except Exception:
        # fallback: simple sentence splitting heuristics
        parse_error = "json_parse_failed"
        # Very simple: split by newline then by sentence tokenizer
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if lines:
            # If the assistant returned a JSON-like or numbered list, try to extract quotes
            # Otherwise, fallback to splitting the full text into sentences
            joined = " ".join(lines)
            sentences = sent_tokenize(joined)

    out = {
        "id": chunk_obj["id"],
        "page_title": chunk_obj.get("page_title"),
        "source_url": chunk_obj.get("source_url"),
        "paragraph_idx": chunk_obj.get("paragraph_idx"),
        "paragraph_word_count": chunk_obj.get("paragraph_word_count"),
        "prompt": prompt,
        "response_text_raw": text,
        "response_sentences": sentences,
        "provider_meta": provider_meta,
        "latency_ms": latency_ms,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "parse_error": parse_error,
        "raw_api_response": raw
    }
    return out

# ----------------------------
# Main
# ----------------------------
def main(args):
    # Threaded pool for moderate concurrency
    pool = ThreadPoolExecutor(max_workers=args.workers)
    futures = []
    # iterate chunks
    for chunk in read_jsonl(args.infile):
        # If you want multiple samples per chunk, call multiple times with different temps
        temps = [args.temperature] if args.samples == 1 else [args.temperature] + [args.temperature + 0.2 * i for i in range(1, args.samples)]
        for temp in temps:
            future = pool.submit(
                summarize_chunk,
                chunk,
                args.model,
                float(temp),
                openrouter_api_key,
                args.max_tokens,
            )
            futures.append(future)
            # optional: throttle launching to respect rate limits
            if len(futures) > args.max_queue:
                for f in as_completed(futures, timeout=None):
                    res = f.result()
                    write_jsonl(args.outfile, res)
                    futures.remove(f)
                    break

    # collect remaining
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
    p.add_argument("--samples", type=int, default=1, help="number of sampled responses per chunk (varying temps by +0.2 increments)")
    p.add_argument("--workers", type=int, default=4, help="concurrent worker threads")
    p.add_argument("--max_queue", type=int, default=200, help="max pending futures before throttling")
    p.add_argument("--max_tokens", type=int, default=200)
    args = p.parse_args()
    main(args)
