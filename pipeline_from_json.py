#!/usr/bin/env python3
"""
End-to-end loader from domains.json -> Reddit -> embeddings -> Postgres (pgvector).

Requirements (pip):
  pip install praw python-dotenv psycopg2-binary SQLAlchemy openai tenacity tqdm

Env vars expected:
  REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT
  OPENAI_API_KEY
Optional:
  DATABASE_URL (default: postgresql+psycopg2://postgres:dummy@localhost:5432/knowledgebase)

Schema expected (already in your DB from earlier):
  domain(domain_id uuid PK, domain_tag text, domain_name text)
  submissions(
    submission_id text PK,
    domain_id uuid NOT NULL REFERENCES domain(domain_id) ON DELETE CASCADE,
    subreddit text, title text, selftext text,
    score int, upvote_ratio double precision, comment_count int,
    permalink text, created_utc timestamp without time zone,
    embedding vector(1536)
  )
  comments(
    comment_id text PK,
    submission_id text NOT NULL REFERENCES submissions(submission_id) ON DELETE CASCADE,
    author text, body text, score int, created_utc timestamp without time zone
  )
"""

import os
import json
import time
import uuid
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Iterable, Optional, Tuple

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm import tqdm

import praw
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, Connection
from openai import OpenAI

# ---------- Config ----------
load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:dummy@localhost:5432/knowledgebase"
)

REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT") or "knowledge-collection-service/0.1"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = "text-embedding-3-small"  # 1536 dims → matches vector(1536)


# How many posts to fetch per (subreddit, query)
PER_QUERY_LIMIT = int(os.getenv("PER_QUERY_LIMIT", "50"))
# How many top-level comments to pull per submission
COMMENTS_LIMIT = int(os.getenv("COMMENTS_LIMIT", "20"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("pipeline_from_json")


# ---------- Helpers ----------
def load_domains_from_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Support either array-of-domains OR { "Domain Name": {subreddits, queries} } style
    if isinstance(data, list):
        domains = data
    elif isinstance(data, dict):
        domains = []
        for i, (name, payload) in enumerate(data.items(), start=1):
            dom = {
                "domain_id": payload.get("domain_id"),
                "domain_tag": payload.get("domain_tag") or "".join([w[0] for w in name.split()]).upper()[:4],
                "domain_name": name,
                "subreddits": payload.get("subreddits", []),
                "queries": payload.get("queries", []),
            }
            domains.append(dom)
    else:
        raise ValueError("domains.json must be a list or an object mapping")

    # Fill missing/null UUIDs in-memory (we don't rewrite the file)
    for d in domains:
        if not d.get("domain_id"):
            d["domain_id"] = str(uuid.uuid4())
    return domains


def connect_engine() -> Engine:
    return create_engine(DATABASE_URL, pool_pre_ping=True, future=True)


def reddit_client() -> praw.Reddit:
    if not (REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET and REDDIT_USER_AGENT):
        raise SystemExit("Missing Reddit creds. Set REDDIT_CLIENT_ID/SECRET/USER_AGENT.")
    return praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
        check_for_async=False,
    )


def openai_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise SystemExit("Missing OPENAI_API_KEY.")
    return OpenAI(api_key=OPENAI_API_KEY)


@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    retry=retry_if_exception_type(Exception),
)
def embed_text(client: OpenAI, text_blob: str) -> List[float]:
    # OpenAI SDK v1
    resp = client.embeddings.create(model=EMBED_MODEL, input=text_blob)
    return resp.data[0].embedding


def to_pgvector_literal(vec: List[float]) -> str:
    # pgvector accepts a string like: [0.1, -0.2, ...]
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


def utc_from_timestamp(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None)  # naive UTC (matches your schema)


# ---------- DB Upserts ----------
def upsert_domain(conn: Connection, domain_id: str, domain_tag: str, domain_name: str) -> None:
    sql = text("""
    INSERT INTO public.domain (domain_id, domain_tag, domain_name)
    VALUES (:domain_id, :domain_tag, :domain_name)
    ON CONFLICT (domain_id) DO UPDATE
    SET domain_tag = EXCLUDED.domain_tag,
        domain_name = EXCLUDED.domain_name
""")

    conn.execute(sql, {"domain_id": domain_id, "domain_tag": domain_tag, "domain_name": domain_name})


def upsert_submission(
    conn: Connection,
    submission_id: str,
    domain_id: str,
    subreddit: str,
    title: str,
    selftext: str,
    score: Optional[int],
    upvote_ratio: Optional[float],
    comment_count: Optional[int],
    permalink: str,
    created_utc: Optional[datetime],
    embedding_vec: Optional[List[float]],
) -> None:
    # Insert first without embedding (or with) — we’ll pass embedding when available
    if embedding_vec is not None:
        sql = text("""
            INSERT INTO public.submissions (
                submission_id, domain_id, subreddit, title, selftext,
                score, upvote_ratio, comment_count, permalink, created_utc, embedding
            )
            VALUES (
                :submission_id, :domain_id, :subreddit, :title, :selftext,
                :score, :upvote_ratio, :comment_count, :permalink, :created_utc, :embedding
            )
            ON CONFLICT (submission_id) DO UPDATE
            SET domain_id = EXCLUDED.domain_id,
                subreddit = EXCLUDED.subreddit,
                title = EXCLUDED.title,
                selftext = EXCLUDED.selftext,
                score = EXCLUDED.score,
                upvote_ratio = EXCLUDED.upvote_ratio,
                comment_count = EXCLUDED.comment_count,
                permalink = EXCLUDED.permalink,
                created_utc = EXCLUDED.created_utc,
                embedding = EXCLUDED.embedding
        """)
        conn.execute(sql, {
            "submission_id": submission_id,
            "domain_id": domain_id,
            "subreddit": subreddit,
            "title": title,
            "selftext": selftext,
            "score": score,
            "upvote_ratio": upvote_ratio,
            "comment_count": comment_count,
            "permalink": permalink,
            "created_utc": created_utc,
            "embedding": to_pgvector_literal(embedding_vec)
        })
    else:
        sql = text("""
            INSERT INTO public.submissions (
                submission_id, domain_id, subreddit, title, selftext,
                score, upvote_ratio, comment_count, permalink, created_utc
            )
            VALUES (
                :submission_id, :domain_id::uuid, :subreddit, :title, :selftext,
                :score, :upvote_ratio, :comment_count, :permalink, :created_utc
            )
            ON CONFLICT (submission_id) DO NOTHING
        """)
        conn.execute(sql, {
            "submission_id": submission_id,
            "domain_id": domain_id,
            "subreddit": subreddit,
            "title": title,
            "selftext": selftext,
            "score": score,
            "upvote_ratio": upvote_ratio,
            "comment_count": comment_count,
            "permalink": permalink,
            "created_utc": created_utc
        })


def upsert_comment(
    conn: Connection,
    comment_id: str,
    submission_id: str,
    author: Optional[str],
    body: str,
    score: Optional[int],
    created_utc: Optional[datetime],
) -> None:
    sql = text("""
        INSERT INTO public.comments (
            comment_id, submission_id, author, body, score, created_utc
        )
        VALUES (
            :comment_id, :submission_id, :author, :body, :score, :created_utc
        )
        ON CONFLICT (comment_id) DO UPDATE
        SET submission_id = EXCLUDED.submission_id,
            author = EXCLUDED.author,
            body = EXCLUDED.body,
            score = EXCLUDED.score,
            created_utc = EXCLUDED.created_utc
    """)
    conn.execute(sql, {
        "comment_id": comment_id,
        "submission_id": submission_id,
        "author": author,
        "body": body,
        "score": score,
        "created_utc": created_utc
    })


# ---------- Reddit collection ----------
def search_posts(
    r: praw.Reddit,
    subreddit: str,
    query: str,
    limit: int
) -> Iterable[praw.models.Submission]:
    # Use Reddit search (not Pushshift). Limit is small to stay polite.
    try:
        for s in r.subreddit(subreddit).search(query=query, sort="relevance", time_filter="year", limit=limit):
            yield s
    except Exception as e:
        log.warning(f"Search failed for r/{subreddit} '{query}': {e}")


def fetch_comments(s: praw.models.Submission, limit: int) -> List[praw.models.Comment]:
    # Refresh comments and flatten a bit, limit to top-level
    try:
        s.comments.replace_more(limit=0)
        out = []
        for c in s.comments[:limit]:
            if isinstance(c, praw.models.Comment):
                out.append(c)
        return out
    except Exception as e:
        log.warning(f"Comments fetch failed for {s.id}: {e}")
        return []


# ---------- Main pipeline ----------
def run(domains_json_path: str) -> None:
    engine = connect_engine()
    r = reddit_client()
    ai = openai_client()

    domains = load_domains_from_json(domains_json_path)

    total_inserted = 0
    total_embedded = 0
    total_comments = 0

    with engine.begin() as conn:
        # Upsert domains
        for d in domains:
            upsert_domain(conn, d["domain_id"], d["domain_tag"], d["domain_name"])

    for d in domains:
        dom_id = d["domain_id"]
        subreddits = d.get("subreddits", [])
        queries = d.get("queries", [])

        if not subreddits or not queries:
            log.info(f"Skipping domain with no subreddits/queries: {d['domain_name']}")
            continue

        log.info(f"Domain: {d['domain_name']} ({d['domain_tag']}) — {len(subreddits)} subs x {len(queries)} queries")

        for sr in subreddits:
            for q in queries:
                log.info(f"Searching r/{sr} for '{q}' (limit={PER_QUERY_LIMIT})")
                # Iterate results, insert + embed
                for s in tqdm(list(search_posts(r, sr, q, PER_QUERY_LIMIT)), desc=f"{sr}:{q}", leave=False):
                    # Basic fields
                    submission_id = s.id  # string
                    subreddit = s.subreddit.display_name if hasattr(s.subreddit, "display_name") else sr
                    title = s.title or ""
                    selftext = s.selftext or ""
                    score = int(getattr(s, "score", 0) or 0)
                    upvote_ratio = float(getattr(s, "upvote_ratio", 0) or 0)
                    comment_count = int(getattr(s, "num_comments", 0) or 0)
                    permalink = f"https://reddit.com{s.permalink}" if getattr(s, "permalink", None) else ""
                    created = utc_from_timestamp(getattr(s, "created_utc", time.time()))

                    # Build embedding text (title + body, truncated defensively)
                    blob = (title + "\n\n" + selftext).strip()
                    if len(blob) > 8000:
                        blob = blob[:8000]

                    # Embed with retries
                    try:
                        emb = embed_text(ai, blob)
                    except Exception as e:
                        log.warning(f"Embedding failed for {submission_id}: {e}")
                        emb = None

                    with engine.begin() as conn:
                        upsert_submission(
                            conn=conn,
                            submission_id=submission_id,
                            domain_id=dom_id,
                            subreddit=subreddit,
                            title=title,
                            selftext=selftext,
                            score=score,
                            upvote_ratio=upvote_ratio,
                            comment_count=comment_count,
                            permalink=permalink,
                            created_utc=created,
                            embedding_vec=emb,
                        )

                    total_inserted += 1
                    if emb is not None:
                        total_embedded += 1

                    # Fetch a slice of comments
                    comments = fetch_comments(s, COMMENTS_LIMIT)
                    if comments:
                        with engine.begin() as conn:
                            for c in comments:
                                cid = c.id
                                author = None
                                try:
                                    author = str(c.author) if c.author else None
                                except Exception:
                                    author = None
                                body = c.body or ""
                                cscore = int(getattr(c, "score", 0) or 0)
                                ccreated = utc_from_timestamp(getattr(c, "created_utc", time.time()))
                                upsert_comment(
                                    conn=conn,
                                    comment_id=cid,
                                    submission_id=submission_id,
                                    author=author,
                                    body=body,
                                    score=cscore,
                                    created_utc=ccreated,
                                )
                                total_comments += 1

    log.info(f"Done. Submissions inserted: {total_inserted}, embedded: {total_embedded}, comments inserted: {total_comments}")
    log.info("Tip: \n  SELECT COUNT(*) FROM submissions; \n  SELECT COUNT(*) FROM comments; \n  SELECT * FROM domain;")


# ---------- CLI ----------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load Reddit data per domains.json into Postgres (with embeddings).")
    parser.add_argument("--domains", "-d", required=True, help="Path to domains.json")
    args = parser.parse_args()

    run(args.domains)
