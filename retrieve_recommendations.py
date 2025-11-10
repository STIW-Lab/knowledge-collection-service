


#!/usr/bin/env python3
"""
Local hybrid retrieval (same logic as your web workflow), but from your KB:

- HNSW pgvector over submissions
- HNSW pgvector over comments (best per submission)
- Postgres FTS over title+selftext+aggregated comments
- LLM plan (must/nice/avoid/themes)
- Pseudo-relevance feedback (PRF) from nearest neighbors
- De-dup + ranking that honors must/avoid + social signals
- Summarize 1 actionable step per thread

Tables:
  submissions(
    submission_id text primary key,
    domain_id text,
    subreddit text,
    title text,
    selftext text,
    score int,
    upvote_ratio float,
    comment_count int,
    permalink text,
    created_utc timestamp,
    embedding vector(1536)
  )
  comments(
    comment_id text primary key,
    submission_id text references submissions(submission_id),
    author text,
    body text,
    score int,
    created_utc timestamp,
    embedding vector(1536)  -- optional but recommended
  )
  domain(
    domain_id text primary key,
    domain_tag text,
    domain_name text
  )
"""

import os, re, json, math
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

import psycopg2
import psycopg2.extras as extras
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# ---------------- Config ----------------
load_dotenv()

DB_DSN = os.getenv("DATABASE_URL", "postgresql://postgres:dummy@127.0.0.1:5432/knowledgebase").replace(
    "postgresql+psycopg2://", "postgresql://"
)
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY")
EMBED_MODEL        = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL         = os.getenv("CHAT_MODEL", "gpt-4o-mini")
SUMMARIZER_MODEL   = os.getenv("SUMMARIZER_MODEL", "gpt-4o-mini")

TOP_K_SUB          = int(os.getenv("TOP_K_SUB", "120"))
TOP_K_COM          = int(os.getenv("TOP_K_COM", "200"))
TOP_K_FTS          = int(os.getenv("TOP_K_FTS", "120"))
FINAL_N            = int(os.getenv("FINAL_N", "5"))
COMMENT_SIP        = int(os.getenv("COMMENT_SIP", "80"))
BACKFILL_COMMENTS  = int(os.getenv("BACKFILL_COMMENTS", "0"))  # 0 disables

HELP_PAT = re.compile(
    r"(this worked|worked for me|helped me|thanks|thank you|game changer|fixed it|saved me|improved|it helped|life saver)",
    re.I
)

STOP = {
    "the","a","an","to","of","and","or","for","in","on","with","by","from","at","as","is","are","was","were",
    "this","that","these","those","it","its","be","been","being","do","did","done","does","can","could","should",
    "would","will","may","might","you","your","yours","i","me","my","we","our","they","their","them"
}

@dataclass
class Hit:
    submission_id: str
    domain_id: Optional[str]
    subreddit: str
    title: str
    selftext: str
    score: int
    upvote_ratio: float
    comment_count: int
    permalink: str
    # signals
    dist: float                  # submission dist
    sim: float                   # 1/(1+dist)
    best_comment_dist: float     # best matching comment dist
    best_comment_sim: float
    fts_rank: float
    # meta
    comments: List[Tuple[str,int,str]]  # (author, score, body)
    domain_tag: Optional[str]
    domain_name: Optional[str]
    must_hits: int               # count of must tokens seen in blob

# ---------------- Infra ----------------
def open_db():
    return psycopg2.connect(DB_DSN)

def open_ai() -> OpenAI:
    if not OPENAI_API_KEY:
        raise SystemExit("Missing OPENAI_API_KEY.")
    return OpenAI(api_key=OPENAI_API_KEY)

def embed_many(client: OpenAI, texts: List[str]) -> List[List[float]]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def embed(client: OpenAI, text: str) -> List[float]:
    return embed_many(client, [text])[0]

def vec_to_pg(vec: List[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"

# ---------------- Setup: pgvector HNSW ----------------
def ensure_pgvector_hnsw(conn):
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("""
          DO $$
          BEGIN
            IF NOT EXISTS (
              SELECT 1 FROM pg_indexes WHERE schemaname='public' AND indexname='idx_submissions_embedding_hnsw'
            ) THEN
              EXECUTE 'CREATE INDEX idx_submissions_embedding_hnsw ON public.submissions USING hnsw (embedding vector_l2_ops) WITH (m=16, ef_construction=200)';
            END IF;
          END$$;
        """)
        cur.execute("""
          DO $$
          BEGIN
            IF EXISTS (
              SELECT 1 FROM information_schema.columns
              WHERE table_schema='public' AND table_name='comments' AND column_name='embedding'
            ) AND NOT EXISTS (
              SELECT 1 FROM pg_indexes WHERE schemaname='public' AND indexname='idx_comments_embedding_hnsw'
            ) THEN
              EXECUTE 'CREATE INDEX idx_comments_embedding_hnsw ON public.comments USING hnsw (embedding vector_l2_ops) WITH (m=16, ef_construction=200)';
            END IF;
          END$$;
        """)
    conn.commit()

# ---------------- Optional: backfill comment embeddings ----------------
def backfill_comment_embeddings(conn, client: OpenAI, limit_rows: int):
    if limit_rows <= 0:
        return
    with conn.cursor() as cur:
        cur.execute("""
          SELECT COUNT(*) FROM information_schema.columns
          WHERE table_schema='public' AND table_name='comments' AND column_name='embedding';
        """)
        has_col = cur.fetchone()[0] == 1
    if not has_col:
        with conn.cursor() as cur:
            cur.execute("ALTER TABLE public.comments ADD COLUMN embedding vector(1536);")
        conn.commit()

    with conn.cursor(cursor_factory=extras.DictCursor) as cur:
        cur.execute("""
          SELECT comment_id, body
          FROM public.comments
          WHERE embedding IS NULL
          ORDER BY score DESC NULLS LAST
          LIMIT %s;
        """, (limit_rows,))
        rows = cur.fetchall()
    if not rows:
        return
    BATCH = 64
    for i in tqdm(range(0, len(rows), BATCH), desc="Backfilling comment embeddings"):
        chunk = rows[i:i+BATCH]
        texts = [(r["body"] or "")[:8000] for r in chunk]
        embs = embed_many(client, texts)
        with conn.cursor() as cur:
            for r, e in zip(chunk, embs):
                cur.execute("UPDATE public.comments SET embedding=%s::vector WHERE comment_id=%s;",
                            (vec_to_pg(e), r["comment_id"]))
        conn.commit()

# ---------------- Token & PRF ----------------
def _tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    toks = [t for t in text.split() if t and t not in STOP and len(t) > 1]
    return toks[:400]

def mine_kb_terms(conn, qvec: List[float], k: int = 150) -> List[str]:
    vec = vec_to_pg(qvec)
    sql = """
      SELECT s.title, s.selftext
      FROM public.submissions s
      WHERE s.embedding IS NOT NULL
      ORDER BY s.embedding <-> %s::vector
      LIMIT %s;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (vec, k))
        rows = cur.fetchall()
    freq: Dict[str,int] = {}
    for title, selftext in rows:
        blob = ((title or "") + " " + (selftext or ""))[:16000]
        toks = _tokenize(blob)
        for t in toks: freq[t] = freq.get(t, 0) + 1
        for i in range(len(toks) - 1):
            a, b = toks[i], toks[i+1]
            if a in STOP or b in STOP: continue
            bg = a + " " + b
            if 3 <= len(bg) <= 30:
                freq[bg] = freq.get(bg, 0) + 1
    return [w for w,_ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)][:180]

# ---------------- LLM planner (mirrors your “narratives”) ----------------
def llm_plan(client: OpenAI, query: str) -> Dict[str, Any]:
    sys = (
        "Parse the user's goal into compact search directives for a Reddit knowledge base.\n"
        "Return STRICT JSON with keys: must_keywords[], nice_keywords[], avoid_keywords[], themes[].\n"
        "Rules: lowercase; 1–3 words; prefer domain terms (e.g., diabetes, a1c); <=20 tokens total."
    )
    user = f"Goal:\n{query}\n\nReturn JSON only."
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role":"system","content":sys},{"role":"user","content":user}],
        temperature=0,
        response_format={"type":"json_object"},
    )
    try:
        data = json.loads(resp.choices[0].message.content)
    except Exception:
        data = {}
    def norm(lst):
        out=[]; seen=set()
        for w in (lst or []):
            if isinstance(w,str):
                w=w.strip().lower()
                if 0 < len(w) <= 40 and w not in seen:
                    seen.add(w); out.append(w)
        return out[:12]
    return {
        "must_keywords": norm(data.get("must_keywords")),
        "nice_keywords": norm(data.get("nice_keywords")),
        "avoid_keywords": norm(data.get("avoid_keywords")),
        "themes":        norm(data.get("themes")),
    }

def select_expansions_with_llm(client: OpenAI, user_query: str, candidates: List[str]) -> List[str]:
    pool = "\n".join(f"- {c}" for c in candidates[:180])
    sys = (
        "From the candidate pool, select 8–12 domain-specific tokens/phrases most relevant to the user's goal.\n"
        'Return STRICT JSON: {"expansions": []}. Lowercase, <=3 words, no duplicates.'
    )
    user = f"USER GOAL:\n{user_query}\n\nCANDIDATES:\n{pool}\n\nReturn JSON only."
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role":"system","content":sys},{"role":"user","content":user}],
        temperature=0,
        response_format={"type":"json_object"},
    )
    try:
        data = json.loads(resp.choices[0].message.content)
        exps = data.get("expansions", []) or []
    except Exception:
        exps = []
    out=[]; seen=set()
    for w in exps:
        if isinstance(w,str):
            w=w.strip().lower()
            if w and len(w)<=40 and w not in seen:
                seen.add(w); out.append(w)
    return out[:12]

# ---------------- Retrieval channels ----------------
def candidates_from_submissions(conn, qvec: List[float], k: int) -> Dict[str, Hit]:
    vec = vec_to_pg(qvec)
    sql = """
      SELECT s.submission_id, s.domain_id, s.subreddit, s.title, s.selftext,
             COALESCE(s.score,0) AS score, COALESCE(s.upvote_ratio,0.0) AS upvote_ratio,
             COALESCE(s.comment_count,0) AS comment_count, s.permalink,
             (s.embedding <-> %s::vector) AS dist,
             d.domain_tag, d.domain_name
      FROM public.submissions s
      LEFT JOIN public.domain d ON d.domain_id = s.domain_id
      WHERE s.embedding IS NOT NULL
      ORDER BY s.embedding <-> %s::vector
      LIMIT %s;
    """
    with conn.cursor(cursor_factory=extras.DictCursor) as cur:
        cur.execute(sql, (vec, vec, k))
        rows = cur.fetchall()
    out: Dict[str, Hit] = {}
    for r in rows:
        dist = float(r["dist"]); sim = 1.0/(1.0+dist)
        out[r["submission_id"]] = Hit(
            submission_id=r["submission_id"],
            domain_id=r["domain_id"],
            subreddit=r["subreddit"],
            title=r["title"] or "",
            selftext=r["selftext"] or "",
            score=int(r["score"] or 0),
            upvote_ratio=float(r["upvote_ratio"] or 0.0),
            comment_count=int(r["comment_count"] or 0),
            permalink=r["permalink"] or "",
            dist=dist, sim=sim,
            best_comment_dist=1e9, best_comment_sim=0.0,
            fts_rank=0.0,
            comments=[],
            domain_tag=r.get("domain_tag"),
            domain_name=r.get("domain_name"),
            must_hits=0,
        )
    return out

def candidates_from_comments(conn, qvec: List[float], k: int) -> Dict[str, Tuple[str,float,float]]:
    # return submission_id -> (comment_id, best_dist, best_sim)
    with conn.cursor() as cur:
        cur.execute("""
          SELECT COUNT(*) FROM information_schema.columns
          WHERE table_schema='public' AND table_name='comments' AND column_name='embedding';
        """)
        has_col = cur.fetchone()[0] == 1
    if not has_col:
        return {}
    vec = vec_to_pg(qvec)
    sql = """
      SELECT c.comment_id, c.submission_id, (c.embedding <-> %s::vector) AS dist
      FROM public.comments c
      WHERE c.embedding IS NOT NULL
      ORDER BY c.embedding <-> %s::vector
      LIMIT %s;
    """
    with conn.cursor(cursor_factory=extras.DictCursor) as cur:
        cur.execute(sql, (vec, vec, k))
        rows = cur.fetchall()
    best: Dict[str, Tuple[str,float,float]] = {}
    for r in rows:
        dist=float(r["dist"]); sim=1.0/(1.0+dist)
        sid=r["submission_id"]
        prev=best.get(sid)
        if prev is None or dist < prev[1]:
            best[sid] = (r["comment_id"], dist, sim)
    return best

def fts_candidates(conn, query_text: str, must: List[str], nice: List[str], k: int) -> Dict[str, float]:
    parts = []
    parts += [w for w in (must or []) if w]
    parts += [w for w in (nice or []) if w]
    # include original query to keep nuance (matches websearch_to_tsquery behavior)
    web_q = " ".join(dict.fromkeys(parts)) or query_text

    sql = """
    WITH cm AS (
      SELECT c.submission_id, string_agg(COALESCE(c.body,''), ' ') AS corpus
      FROM public.comments c GROUP BY c.submission_id
    )
    SELECT s.submission_id,
           ts_rank_cd(
             to_tsvector('english', COALESCE(s.title,'') || ' ' || COALESCE(s.selftext,'') || ' ' || COALESCE(cm.corpus,'')),
             websearch_to_tsquery('english', %s)
           ) AS fts_rank
    FROM public.submissions s
    LEFT JOIN cm ON cm.submission_id = s.submission_id
    WHERE websearch_to_tsquery('english', %s) @@
          to_tsvector('english', COALESCE(s.title,'') || ' ' || COALESCE(s.selftext,'') || ' ' || COALESCE(cm.corpus,''))
    ORDER BY fts_rank DESC
    LIMIT %s;
    """
    with conn.cursor(cursor_factory=extras.DictCursor) as cur:
        cur.execute(sql, (web_q, web_q, k))
        rows = cur.fetchall()
    return {r["submission_id"]: float(r["fts_rank"] or 0.0) for r in rows}

def fetch_comments(conn, submission_id: str, limit: int) -> List[Tuple[str,int,str]]:
    sql = """
      SELECT COALESCE(author,''), COALESCE(score,0), COALESCE(body,'')
      FROM public.comments
      WHERE submission_id=%s
      ORDER BY score DESC
      LIMIT %s;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (submission_id, limit))
        return cur.fetchall()

# ---------------- Heuristics ----------------
def helpful_votes(h: Hit) -> int:
    affirm = 0
    for (_a, _s, b) in h.comments:
        if HELP_PAT.search(b or ""):
            affirm += 1
    bonus = max(0, int(round((h.upvote_ratio * max(h.score, 0)) / 10.0)))
    return affirm + bonus

def social_engagement(h: Hit) -> float:
    eng = math.log1p(max(h.score,0) + max(h.comment_count,0))
    return min(1.0, eng / 10.0)

def count_must_hits(blob: str, must: List[str]) -> int:
    if not must: return 0
    low = (blob or "").lower()
    return sum(1 for m in must if m and m.lower() in low)

# ---------------- Merge + rank ----------------
def merge_and_rank(conn,
                   sub_hits: Dict[str, Hit],
                   com_best: Dict[str, Tuple[str,float,float]],
                   fts_rank: Dict[str, float],
                   must: List[str],
                   avoid: List[str],
                   final_n: int) -> List[Hit]:

    # add best comment sim
    for sid, (_cid, cdist, csim) in com_best.items():
        if sid in sub_hits:
            h = sub_hits[sid]
            h.best_comment_dist = min(h.best_comment_dist, cdist)
            h.best_comment_sim  = max(h.best_comment_sim, csim)
        else:
            # bring missing submission in
            with conn.cursor(cursor_factory=extras.DictCursor) as cur:
                cur.execute("""
                  SELECT s.submission_id, s.domain_id, s.subreddit, s.title, s.selftext,
                         COALESCE(s.score,0) AS score, COALESCE(s.upvote_ratio,0.0) AS upvote_ratio,
                         COALESCE(s.comment_count,0) AS comment_count, s.permalink,
                         d.domain_tag, d.domain_name
                  FROM public.submissions s
                  LEFT JOIN public.domain d ON d.domain_id = s.domain_id
                  WHERE s.submission_id=%s;
                """, (sid,))
                r = cur.fetchone()
            if r:
                sub_hits[sid] = Hit(
                    submission_id=r["submission_id"],
                    domain_id=r["domain_id"],
                    subreddit=r["subreddit"],
                    title=r["title"] or "",
                    selftext=r["selftext"] or "",
                    score=int(r["score"] or 0),
                    upvote_ratio=float(r["upvote_ratio"] or 0.0),
                    comment_count=int(r["comment_count"] or 0),
                    permalink=r["permalink"] or "",
                    dist=1.0, sim=0.0,
                    best_comment_dist=cdist, best_comment_sim=csim,
                    fts_rank=0.0,
                    comments=[],
                    domain_tag=r.get("domain_tag"),
                    domain_name=r.get("domain_name"),
                    must_hits=0,
                )

    # apply FTS score
    for sid, fr in fts_rank.items():
        if sid in sub_hits:
            sub_hits[sid].fts_rank = max(sub_hits[sid].fts_rank, fr)
        else:
            with conn.cursor(cursor_factory=extras.DictCursor) as cur:
                cur.execute("""
                  SELECT s.submission_id, s.domain_id, s.subreddit, s.title, s.selftext,
                         COALESCE(s.score,0) AS score, COALESCE(s.upvote_ratio,0.0) AS upvote_ratio,
                         COALESCE(s.comment_count,0) AS comment_count, s.permalink,
                         d.domain_tag, d.domain_name
                  FROM public.submissions s
                  LEFT JOIN public.domain d ON d.domain_id = s.domain_id
                  WHERE s.submission_id=%s;
                """, (sid,))
                r = cur.fetchone()
            if r:
                sub_hits[sid] = Hit(
                    submission_id=r["submission_id"],
                    domain_id=r["domain_id"],
                    subreddit=r["subreddit"],
                    title=r["title"] or "",
                    selftext=r["selftext"] or "",
                    score=int(r["score"] or 0),
                    upvote_ratio=float(r["upvote_ratio"] or 0.0),
                    comment_count=int(r["comment_count"] or 0),
                    permalink=r["permalink"] or "",
                    dist=1.0, sim=0.0,
                    best_comment_dist=1e9, best_comment_sim=0.0,
                    fts_rank=fr,
                    comments=[],
                    domain_tag=r.get("domain_tag"),
                    domain_name=r.get("domain_name"),
                    must_hits=0,
                )

    hits = list(sub_hits.values())

    # hydrate comments & compute must_hits + avoid veto
    av = [a.lower() for a in (avoid or [])]
    filt = []
    for h in hits:
        h.comments = fetch_comments(conn, h.submission_id, COMMENT_SIP)
        blob = f"{h.title}\n{h.selftext}\n" + "\n".join((b or "") for (_a,_s,b) in h.comments[:12])
        if av and any(a in blob.lower() for a in av):
            continue  # veto
        h.must_hits = count_must_hits(blob, must)
        filt.append(h)
    hits = filt

    if not hits:
        return []

    # normalize
    max_sim  = max([h.sim for h in hits] + [1e-6])
    max_csim = max([h.best_comment_sim for h in hits] + [1e-6])
    max_fts  = max([h.fts_rank for h in hits] + [1e-6])

    # weights (close to your web logic but favor must-hit)
    W_SUB   = 0.40
    W_COM   = 0.22
    W_FTS   = 0.23
    W_SOC   = 0.10
    W_MUST  = 0.05  # soft boost if blob mentions must tokens

    def rank(h: Hit) -> float:
        s_sub = (h.sim / max_sim) if max_sim > 0 else 0.0
        s_com = (h.best_comment_sim / max_csim) if max_csim > 0 else 0.0
        s_fts = (h.fts_rank / max_fts) if max_fts > 0 else 0.0
        s_soc = social_engagement(h)
        s_mus = min(1.0, h.must_hits / max(1, len(must))) if must else 0.0
        return W_SUB*s_sub + W_COM*s_com + W_FTS*s_fts + W_SOC*s_soc + W_MUST*s_mus

    hits.sort(key=rank, reverse=True)

    # de-dup by (subreddit + title[:256]) to avoid near clones
    seen = set(); dedup=[]
    for h in hits:
        key = (h.subreddit, (h.title or "")[:256])
        if key in seen: 
            continue
        seen.add(key); dedup.append(h)
    return dedup[:final_n]

# ---------------- Summarization ----------------
def build_llm_prompt(query: str, h: Hit, max_comments: int = 8) -> str:
    cs = h.comments[:max_comments]
    ctext = "\n\n".join([f"- ({sc}) {b.strip()}"[:1000] for (_a, sc, b) in cs if (b or "").strip()])
    return f"""
You are an assistant turning a Reddit thread into **1 concrete, practical step** toward the user's goal.

USER GOAL:
{query.strip()}

SOURCE THREAD:
Subreddit: r/{h.subreddit}
Title: {h.title.strip()}

Post:
{h.selftext.strip()[:2000]}

Top comments (score in parens):
{ctext}

TASK:
- Propose one actionable step (start with a verb; keep it crisp).
- Add a brief "How to do it" (1–3 sentences, specific).
- If there's a caveat or precondition mentioned, include it in one short line ("Watch out for …").
- Stay faithful to the text. Do NOT invent facts.

Return strict JSON with keys exactly:
{{
  "step": "...",
  "how": "...",
  "caveat": "..."  // may be empty
}}
""".strip()

def llm_summarize_step(client: OpenAI, prompt: str) -> dict:
    resp = client.chat.completions.create(
        model=SUMMARIZER_MODEL,
        messages=[{"role":"user","content":prompt}],
        temperature=0.2,
        response_format={"type":"json_object"},
    )
    try:
        data = json.loads(resp.choices[0].message.content)
        return {
            "step":   (data.get("step") or "").strip(),
            "how":    (data.get("how") or "").strip(),
            "caveat": (data.get("caveat") or "").strip(),
        }
    except Exception:
        return {"step":"","how":"","caveat":""}

# ---------------- Orchestrator ----------------
def retrieve(query: str) -> List[dict]:
    client = open_ai()
    qvec = embed(client, query)
    with open_db() as conn:
        ensure_pgvector_hnsw(conn)
        if BACKFILL_COMMENTS > 0:
            backfill_comment_embeddings(conn, client, BACKFILL_COMMENTS)

        # plan + PRF (mirrors your notebook routing)
        plan = llm_plan(client, query)
        prf_candidates = mine_kb_terms(conn, qvec, k=150)
        expansions = select_expansions_with_llm(client, query, prf_candidates)

        must  = plan.get("must_keywords") or []
        nice  = list(dict.fromkeys((plan.get("nice_keywords") or []) + expansions))[:20]
        avoid = plan.get("avoid_keywords") or []

        # channels
        sub_hits = candidates_from_submissions(conn, qvec, TOP_K_SUB)
        com_best = candidates_from_comments(conn, qvec, TOP_K_COM)
        fts_rank = fts_candidates(conn, query, must, nice, TOP_K_FTS)

        # merge + rank with must/avoid shaping (keeps “diabetes” on-topic)
        top_hits = merge_and_rank(conn, sub_hits, com_best, fts_rank, must, avoid, FINAL_N)

    # summarize
    out = []
    for h in tqdm(top_hits, desc="Summarizing"):
        summary = llm_summarize_step(client, build_llm_prompt(query, h))
        out.append({
            "step": summary.get("step") or (h.title.strip()[:140] or "Do this first"),
            "how": summary.get("how"),
            "caveat": summary.get("caveat"),
            "people_found_helpful": helpful_votes(h),
            "subreddit": h.subreddit,
            "permalink": h.permalink,
            "domain_id": h.domain_id,
            "domain_tag": h.domain_tag,
            "domain_name": h.domain_name,
            "score": h.score,
            "upvote_ratio": h.upvote_ratio,
        })
    return out

# ---------------- CLI ----------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Retrieve actionable steps from local Reddit KB (HNSW + FTS), matching web logic.")
    p.add_argument("--query", "-q", required=True, help="User goal / constraints (plain text).")
    args = p.parse_args()

    recs = retrieve(args.query)
    for i, r in enumerate(recs, 1):
        print(f"\n#{i}. {r['step']}")
        print(f"   How: {r['how']}")
        if r.get("caveat"):
            print(f"   Caveat: {r['caveat']}")
        print(f"   ~ {r['people_found_helpful']} people found this helpful")
        dom = f" | Domain: {r['domain_tag'] or ''} {r['domain_name'] or ''}".strip()
        print(f"   Source: r/{r['subreddit']}  {r['permalink']}{dom}")
