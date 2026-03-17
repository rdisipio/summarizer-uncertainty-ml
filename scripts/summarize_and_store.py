#!/usr/bin/env python3
"""
summarize_and_store.py
- Read chunk JSONL produced by fetch_and_chunk.py
- Call summarizer API (OpenAI chat or OpenRouter)
- Store structured results (summary sentences, raw response, prompt, metadata) to JSONL

Usage:
    export OPENAI_API_KEY=...
    python summarize_and_store.py --in chunks.jsonl.gz --out summaries.jsonl.gz --provider openai --model gpt-4o-mini --samples 3 --temperature 0.2

Requirements:
    pip install openai requests tqdm ratelimit backoff
Notes:
    - Chat endpoints typically do NOT return token logprobs. If you need token logprobs, use completion endpoints that expose them.
    - OpenRouter usage via requests is shown as a fallback; set OPENROUTER_API_KEY if using provider=openrouter.
"""

import argparse
import os
import json
import gzip
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import backoff

# Optional: import openai if provider=openai
try:
    import openai
except Exception:
    openai = None

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
def call_openai_chat(prompt_messages, model, temperature, max_tokens=256, api_key=None):
    if openai is None:
        raise RuntimeError("openai package not installed")
    openai.api_key = api_key or os.environ.get("OPENAI_API_KEY")
    resp = openai.ChatCompletion.create(
        model=model,
        messages=prompt_messages,
        temperature=temperature,
        max_tokens=max_tokens,
        n=1
        # note: ChatCompletion usually doesn't return token logprobs
    )
    return resp

@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def call_openrouter_chat(prompt, model, temperature, api_key=None):
    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    url = "https://api.openrouter.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
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

def make_messages_from_prompt(prompt):
    # Chat API compatible messages (user-only)
    return [{"role": "user", "content": prompt}]

# ----------------------------
# Worker
# ----------------------------
def summarize_chunk(chunk_obj, provider="openai", model="gpt-4o-mini", temperature=0.0, api_keys=None, max_tokens=200):
    paragraph = chunk_obj["paragraph_text"]
    prompt = PROMPT_TEMPLATE.format(paragraph=paragraph)
    start = time.time()
    raw = None
    try:
        if provider == "openai":
            messages = make_messages_from_prompt(prompt)
            raw = call_openai_chat(messages, model=model, temperature=temperature, max_tokens=max_tokens, api_key=api_keys.get("openai"))
            # parse assistant text
            text = raw["choices"][0]["message"]["content"].strip()
            provider_meta = {"provider": "openai", "model": model}
        else:
            raw = call_openrouter_chat(prompt, model=model, temperature=temperature, api_key=api_keys.get("openrouter"))
            # OpenRouter response format might differ
            # try to access the assistant message
            choices = raw.get("choices") or []
            text = choices[0].get("message", {}).get("content") if choices else ""
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
            from nltk import sent_tokenize
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
    api_keys = {"openai": os.environ.get("OPENAI_API_KEY"), "openrouter": os.environ.get("OPENROUTER_API_KEY")}
    # Threaded pool for moderate concurrency
    pool = ThreadPoolExecutor(max_workers=args.workers)
    futures = []
    count = 0
    # iterate chunks
    for chunk in read_jsonl(args.infile):
        # If you want multiple samples per chunk, call multiple times with different temps
        temps = [args.temperature] if args.samples == 1 else [args.temperature] + [args.temperature + 0.2 * i for i in range(1, args.samples)]
        for temp in temps:
            future = pool.submit(summarize_chunk, chunk, args.provider, args.model, float(temp), api_keys, args.max_tokens)
            futures.append(future)
            count += 1
            # optional: throttle launching to respect rate limits
            if len(futures) > args.max_queue:
                # wait for some to finish
                done, not_done = [], []
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
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--infile", required=True, help="input chunks JSONL (gz ok)")
    p.add_argument("--outfile", required=True, help="output JSONL (gz ok)")
    p.add_argument("--provider", choices=["openai", "openrouter"], default="openai")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--samples", type=int, default=1, help="number of sampled responses per chunk (varying temps by +0.2 increments)")
    p.add_argument("--workers", type=int, default=4, help="concurrent worker threads")
    p.add_argument("--max_queue", type=int, default=200, help="max pending futures before throttling")
    p.add_argument("--max_tokens", type=int, default=200)
    args = p.parse_args()
    main(args)
