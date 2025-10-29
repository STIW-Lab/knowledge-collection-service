import time
import re
import json
from pathlib import Path
from typing import List, Dict, Any

from config import REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT

# --- Heuristics - Regex---
STEP_PAT = re.compile(r"^(\s*[-*\d\.)]+\s+|\b(try|consider|start|first|next|then|finally|should)\b)", re.I)

def advice_like(text: str) -> bool:
    if not text: return False
    t = text.strip()
    if len(t) < 30:  # too short to be useful advice
        return False
    # contains bullets, numbers, or directive verbs
    return bool(STEP_PAT.search(t))

#Not implemented - should be discussed!!!
def extract_steps(text: str) -> List[str]:
    # crude split by lines that look like bullet/numbered steps
    steps = []
    for line in text.splitlines():
        lt = line.strip()
        if len(lt) >= 4 and (lt.startswith("-") or lt.startswith("*") or re.match(r"^\d+\.|^\d+\)", lt)):
            steps.append(lt.lstrip("-* ").strip())
    # fallback: look for sentences with directive verbs
    if not steps:
        sents = re.split(r"(?<=[.!?])\s+", text)
        for s in sents:
            if advice_like(s):
                steps.append(s.strip())
    # keep unique-ish
    seen = set()
    uniq = []
    for s in steps:
        k = s.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(s)
    return uniq[:10]

def score_comment(score: int, num_replies: int, awards: int, length: int) -> float:
    # Normalize-ish by simple log scaling and weights
    import math
    s = math.log1p(max(score, 0)) * 0.6
    r = math.log1p(max(num_replies, 0)) * 0.3
    a = math.log1p(max(awards, 0)) * 0.1
    L = 0.0
    if 60 <= length <= 1200:
        # reward reasonable length
        L = 0.2
    return s + r + a + L

def score_submission(score: int, num_comments: int, upvote_ratio: float) -> float:
    import math
    s = math.log1p(max(score, 0)) * 0.5
    c = math.log1p(max(num_comments, 0)) * 0.3
    u = (upvote_ratio or 0.5) * 0.2
    return s + c + u

def run(domains_path: str, out_path: str, per_query: int = 10, per_comments: int = 150):
    # Lazy import PRAW to fail early if creds are missing
    import praw
    from praw.models import MoreComments

    if not (REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET and REDDIT_USER_AGENT):
        raise SystemExit("Missing Reddit creds. Fill .env first (REDDIT_CLIENT_ID/SECRET/USER_AGENT).")

    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
    )

    domains = json.load(open(domains_path, "r", encoding="utf-8"))
    out = open(out_path, "w", encoding="utf-8")

    out.write("[\n")
    first_item = True

    for domain, cfg in domains.items():
        subs = cfg.get("subreddits", [])
        queries = cfg.get("queries", [])
        print(f"[domain] {domain}: {subs} with {len(queries)} queries")

        for sub in subs:
            subreddit = reddit.subreddit(sub)
            for q in queries:
                print(f"  [search] r/{sub} :: {q}")
                results = subreddit.search(q, sort="relevance", time_filter="year", limit=per_query)
                for submission in results:
                    try:
                        submission_data = {
                            "domain": domain,
                            "source": "reddit",
                            "collected_at": int(time.time()),
                            "question": {
                                "id": submission.id,
                                "title": submission.title or "",
                                "url": f"https://www.reddit.com{submission.permalink}",
                                "selftext": submission.selftext or "",
                            },
                            "question_meta": {
                                "subreddit": sub,
                                "score": int(getattr(submission, "score", 0) or 0),
                                "num_comments": int(getattr(submission, "num_comments", 0) or 0),
                                "upvote_ratio": float(getattr(submission, "upvote_ratio", 0.0) or 0.0),
                                "created_utc": int(getattr(submission, "created_utc", 0) or 0),
                                "relevance_score": score_submission(
                                    int(getattr(submission, "score", 0) or 0),
                                    int(getattr(submission, "num_comments", 0) or 0),
                                    float(getattr(submission, "upvote_ratio", 0.0) or 0.0),
                                ),
                            },
                            "answers": [],
                        }

                        submission.comments.replace_more(limit=0)
                        comments = submission.comments.list()
                        comments = comments[:per_comments]

                        for c in comments:
                            if isinstance(c, MoreComments):
                                continue
                            body = (c.body or "").strip()
                            # awards
                            total_awards = 0
                            try:
                                if hasattr(c, "total_awards_received"):
                                    total_awards = int(c.total_awards_received or 0)
                            except Exception:
                                total_awards = 0
                            # replies count
                            replies_count = 0
                            try:
                                replies_count = len(c.replies) if hasattr(c, "replies") else 0
                            except Exception:
                                replies_count = 0

                            sc = score_comment(
                                int(getattr(c, "score", 0) or 0),
                                int(replies_count or 0),
                                int(total_awards or 0),
                                len(body),
                            )

                            item = {
                                "id": c.id,
                                "parent_id": c.parent_id,
                                "is_op": bool(getattr(c, "is_submitter", False)),
                                "author": str(getattr(c, "author", "")) if getattr(c, "author", None) else "[deleted]",
                                "score": int(getattr(c, "score", 0) or 0),
                                "replies_count": int(replies_count or 0),
                                "total_awards_received": int(total_awards or 0),
                                "created_utc": int(getattr(c, "created_utc", 0) or 0),
                                "text": body,
                                "extracted_steps": extract_steps(body),
                                "relevance_score": sc,
                            }
                            submission_data["answers"].append(item)

                        submission_data["answers"].sort(key=lambda x: x["relevance_score"], reverse=True)

                        # out.write(json.dumps(submission_data, ensure_ascii=False) + "\\n")
                        # out.flush()

                        if not first_item:
                            out.write(",\n")
                        else:
                            first_item = False

                        # pretty-print the object
                        pretty = json.dumps(submission_data, ensure_ascii=False, indent=2)
                        # indent nicely under the array 
                        indented = "\n".join(("  " + line) for line in pretty.splitlines())
                        out.write(indented)
                        out.flush()

                    except Exception as e:
                        print(f"    [warn] error on submission: {e}")
                        continue

    out.write("\n]\n")
    out.close()
    print(f"[done] wrote {out_path}")

if __name__ == "__main__":
    run(domains_path="domains.json", out_path="reddit_dataset.jsonl", per_query=5, per_comments=120)
