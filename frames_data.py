"""
FRAMES Data

Loads the FRAMES benchmark (Krishna et al., "Fact, Fetch, and Reason";
google/frames-benchmark on Hugging Face) and, for a given question, fetches
its gold Wikipedia articles via the real Wikipedia API. This is the "real
gold documents, capped in length" piece of IntentKV's documented FRAMES
adaptation -- see trace_agent.py's web_search docstring for what's
deliberately simplified relative to that paper's full adaptation
(no distractor documents, no forced minimum tool-call counts).

FRAMES: 824 multi-hop questions, each requiring 2-15 Wikipedia articles to
answer, with a gold answer and a reasoning_types label. The dataset itself
only ships Wikipedia URLs, not article text -- that's fetched here.
"""

import ast
import json
import os
import urllib.parse

import requests
from datasets import load_dataset

WIKI_CACHE_DIR = "wiki_cache"
MAX_CHARS_PER_DOC = 3000  # same cap IntentKV's FRAMES adaptation used
_USER_AGENT = "foresite-kv-cache-research/1.0 (educational project; contact via github)"


def load_frames_questions(n=None, seed=42):
    """
    returns a list of dicts: {prompt, answer, reasoning_types, wiki_links}
    wiki_links is a real python list of Wikipedia URLs (non-null only)
    """
    ds = load_dataset("google/frames-benchmark", split="test")
    if n is not None:
        ds = ds.shuffle(seed=seed).select(range(n))

    questions = []
    for row in ds:
        links = ast.literal_eval(row["wiki_links"]) if isinstance(row["wiki_links"], str) else row["wiki_links"]
        questions.append(
            {
                "prompt": row["Prompt"],
                "answer": row["Answer"],
                "reasoning_types": row["reasoning_types"],
                "wiki_links": [u for u in links if u],
            }
        )
    return questions


def _title_from_url(url: str) -> str:
    path = url.rstrip("/").rsplit("/", 1)[-1]
    return urllib.parse.unquote(path)


def _safe_filename(title: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in title)[:150]


def fetch_wikipedia_extract(url: str, max_chars: int = MAX_CHARS_PER_DOC):
    """
    returns (real_title, text) for one Wikipedia URL, via the real Wikipedia
    API (action=query&prop=extracts), cached to disk so re-running the same
    question doesn't re-fetch the same articles
    """
    title = _title_from_url(url)
    cache_path = os.path.join(WIKI_CACHE_DIR, f"{_safe_filename(title)}.json")

    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        return cached["title"], cached["text"][:max_chars]

    resp = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "titles": title,
            "format": "json",
            "redirects": 1,
        },
        headers={"User-Agent": _USER_AGENT},
        timeout=15,
    )
    resp.raise_for_status()
    pages = resp.json()["query"]["pages"]
    page = next(iter(pages.values()))
    real_title = page.get("title", title)
    text = page.get("extract") or f"[No Wikipedia extract found for {title}]"

    os.makedirs(WIKI_CACHE_DIR, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump({"title": real_title, "text": text}, f)

    return real_title, text[:max_chars]


def build_corpus(question: dict, max_chars: int = MAX_CHARS_PER_DOC) -> list[dict]:
    """
    fetches every gold Wikipedia article for one FRAMES question, returns
    a list of {title, url, text}. skips (and warns on) any URL that fails
    to fetch rather than failing the whole question.
    """
    corpus = []
    for url in question["wiki_links"]:
        try:
            title, text = fetch_wikipedia_extract(url, max_chars=max_chars)
            corpus.append({"title": title, "url": url, "text": text})
        except (requests.RequestException, KeyError, IndexError, StopIteration) as e:
            print(f"  [warning: failed to fetch {url}: {e}]")
    return corpus


if __name__ == "__main__":
    questions = load_frames_questions(n=3)
    print(f"loaded {len(questions)} FRAMES questions")
    for q in questions:
        print(f"\nprompt: {q['prompt']}")
        print(f"gold answer: {q['answer']}")
        print(f"reasoning_types: {q['reasoning_types']}")
        print(f"gold wiki_links ({len(q['wiki_links'])}): {q['wiki_links']}")
        corpus = build_corpus(q)
        print(f"fetched {len(corpus)}/{len(q['wiki_links'])} gold documents")
        for doc in corpus:
            print(f"  - {doc['title']}: {len(doc['text'])} chars")
