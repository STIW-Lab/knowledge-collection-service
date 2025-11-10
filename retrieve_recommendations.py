# #!/usr/bin/env python3
# """
# Query your pgvector knowledge base for actionable, ranked steps.

# Inputs:
#   - User query (goal / constraints as plain text)

# Outputs (top 5):
#   - step (short imperative)
#   - how (1–3 lines)
#   - "N people found this helpful" (heuristic)
#   - source subreddit + Reddit permalink

# Requirements (pip):
#   pip install python-dotenv psycopg2-binary openai tqdm

# Env vars:
#   DATABASE_URL = postgresql://postgres:dummy@127.0.0.1:5432/knowledgebase
#   OPENAI_API_KEY = ...
# Optional:
#   EMBED_MODEL (default: text-embedding-3-small)
#   SUMMARIZER_MODEL (default: gpt-4o-mini)
#   CHAT_MODEL (default: gpt-4o-mini)
#   TOP_K (default: 30)
#   FINAL_N (default: 5)
# """

# import os
# import re
# import math
# import json
# from dataclasses import dataclass
# from typing import List, Tuple, Optional, Dict, Any

# import psycopg2
# import psycopg2.extras as extras
# from dotenv import load_dotenv
# from tqdm import tqdm
# from openai import OpenAI

# # ---------- Config ----------
# load_dotenv()

# DB_DSN = os.getenv(
#     "DATABASE_URL",
#     "postgresql://postgres:dummy@127.0.0.1:5432/knowledgebase"
# ).replace("postgresql+psycopg2://", "postgresql://")

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
# SUMMARIZER_MODEL = os.getenv("SUMMARIZER_MODEL", "gpt-4o-mini")
# CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# TOP_K = int(os.getenv("TOP_K", "30"))      # first-stage nearest neighbors
# FINAL_N = int(os.getenv("FINAL_N", "5"))   # how many to return
# COMMENT_SIP = 50                           # comments fetched per hit for signals/summarization

# HELP_PAT = re.compile(
#     r"(this worked|worked for me|helped me|thanks|thank you|game changer|fixed it|saved me|improved|it helped|life saver)",
#     re.I
# )

# STOP = {
#     "the","a","an","to","of","and","or","for","in","on","with","by","from","at","as","is","are","was","were",
#     "this","that","these","those","it","its","be","been","being","do","did","done","does","can","could","should",
#     "would","will","may","might","you","your","yours","i","me","my","we","our","they","their","them"
# }

# @dataclass
# class Hit:
#     submission_id: str
#     domain_id: str
#     subreddit: str
#     title: str
#     selftext: str
#     score: int
#     upvote_ratio: float
#     comment_count: int
#     permalink: str
#     dist: float
#     sim: float
#     comments: List[Tuple[str, int, str]]


# # ---------- Infra ----------
# def open_db():
#     return psycopg2.connect(DB_DSN)


# def open_ai() -> OpenAI:
#     if not OPENAI_API_KEY:
#         raise SystemExit("Missing OPENAI_API_KEY.")
#     return OpenAI(api_key=OPENAI_API_KEY)


# def embed(client: OpenAI, text: str) -> List[float]:
#     return client.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding


# def vec_to_pg(vec: List[float]) -> str:
#     return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


# # ---------- Basic nearest neighbors ----------
# def nearest_submissions(conn, qvec: List[float], k: int) -> List[Hit]:
#     """
#     Uses pgvector L2 distance. Similarity proxy: sim = 1 / (1 + dist)
#     """
#     vec = vec_to_pg(qvec)
#     sql = """
#       SELECT
#         s.submission_id, s.domain_id, s.subreddit, s.title, s.selftext,
#         COALESCE(s.score,0) AS score,
#         COALESCE(s.upvote_ratio,0.0) AS upvote_ratio,
#         COALESCE(s.comment_count,0) AS comment_count,
#         s.permalink,
#         (s.embedding <-> %s::vector) AS dist
#       FROM submissions s
#       WHERE s.embedding IS NOT NULL
#       ORDER BY s.embedding <-> %s::vector
#       LIMIT %s;
#     """
#     with conn.cursor(cursor_factory=extras.DictCursor) as cur:
#         cur.execute(sql, (vec, vec, k))
#         rows = cur.fetchall()

#     hits: List[Hit] = []
#     for r in rows:
#         dist = float(r["dist"])
#         sim = 1.0 / (1.0 + dist)
#         hits.append(Hit(
#             submission_id=r["submission_id"],
#             domain_id=r["domain_id"],
#             subreddit=r["subreddit"],
#             title=r["title"] or "",
#             selftext=r["selftext"] or "",
#             score=int(r["score"] or 0),
#             upvote_ratio=float(r["upvote_ratio"] or 0.0),
#             comment_count=int(r["comment_count"] or 0),
#             permalink=r["permalink"] or "",
#             dist=dist,
#             sim=sim,
#             comments=[]
#         ))
#     return hits


# # ---------- Comments ----------
# def fetch_comments_for(conn, submission_id: str, limit: int = 50) -> List[Tuple[str, int, str]]:
#     sql = """
#       SELECT COALESCE(author,''), COALESCE(score,0), COALESCE(body,'')
#       FROM comments
#       WHERE submission_id = %s
#       ORDER BY score DESC
#       LIMIT %s;
#     """
#     with conn.cursor() as cur:
#         cur.execute(sql, (submission_id, limit))
#         return cur.fetchall()


# def estimate_helpful_count(hit: Hit) -> int:
#     """
#     Heuristic:
#       - count comments matching HELP_PAT
#       - add bonus from upvotes: round(max(0, upvote_ratio * score) / 10)
#     """
#     affirm = 0
#     for (_, cscore, body) in hit.comments:
#         if HELP_PAT.search(body or ""):
#             affirm += 1
#     bonus = max(0, int(round((hit.upvote_ratio * max(hit.score, 0)) / 10.0)))
#     return affirm + bonus


# def engagement_score(hit: Hit) -> float:
#     """
#     Combine semantic similarity with social signal.
#       rank = 0.6 * sim + 0.4 * norm(log(1 + score + comment_count))
#     """
#     eng = math.log1p(max(hit.score, 0) + max(hit.comment_count, 0))
#     norm_eng = min(1.0, eng / 10.0)
#     return 0.6 * hit.sim + 0.4 * norm_eng


# # ---------- Token utilities (for PRF) ----------
# def _tokenize(text: str) -> List[str]:
#     text = (text or "").lower()
#     text = re.sub(r"[^a-z0-9\s]", " ", text)
#     toks = [t for t in text.split() if t and t not in STOP and len(t) > 1]
#     return toks[:400]


# # ---------- LLM planner (no hardcoded synonyms) ----------
# def llm_plan(client: OpenAI, query: str) -> Dict[str, Any]:
#     """
#     Extract narrow routing info from a user goal for Reddit KB search.
#     Returns JSON: {must_keywords[], nice_keywords[], avoid_keywords[], themes[]}
#     """
#     sys = (
#         "You parse a user goal into compact search directives for a Reddit knowledge base.\n"
#         "Return STRICT JSON with keys: must_keywords[], nice_keywords[], avoid_keywords[], themes[].\n"
#         "Rules:\n"
#         "- Lowercase tokens, 1–3 words each.\n"
#         "- Prefer domain terms (e.g., 'diabetes', 'a1c', 'resume').\n"
#         "- Avoid generic verbs and pronouns.\n"
#         "- Keep total tokens small (<= 20 across all lists)."
#     )
#     user = f"Goal:\n{query}\n\nReturn JSON only."
#     resp = client.chat.completions.create(
#         model=CHAT_MODEL,
#         messages=[{"role": "system", "content": sys},
#                   {"role": "user", "content": user}],
#         temperature=0,
#         response_format={"type": "json_object"},
#     )
#     try:
#         data = json.loads(resp.choices[0].message.content)
#     except Exception:
#         data = {}
#     def norm(lst):
#         out = []
#         for w in (lst or []):
#             if not isinstance(w, str):
#                 continue
#             w = w.strip().lower()
#             if 0 < len(w) <= 40:
#                 out.append(w)
#         seen = set(); keep=[]
#         for w in out:
#             if w not in seen:
#                 seen.add(w); keep.append(w)
#         return keep[:12]
#     return {
#         "must_keywords": norm(data.get("must_keywords")),
#         "nice_keywords": norm(data.get("nice_keywords")),
#         "avoid_keywords": norm(data.get("avoid_keywords")),
#         "themes":        norm(data.get("themes")),
#     }


# # ---------- Pseudo-relevance feedback from your KB ----------
# def mine_kb_terms(conn, qvec: List[float], k: int = 80) -> List[str]:
#     """
#     Take top-k vector neighbors and mine frequent unigrams + bigrams.
#     """
#     vec = vec_to_pg(qvec)
#     sql = """
#       SELECT title, selftext
#       FROM submissions
#       WHERE embedding IS NOT NULL
#       ORDER BY embedding <-> %s::vector
#       LIMIT %s;
#     """
#     with conn.cursor() as cur:
#         cur.execute(sql, (vec, k))
#         rows = cur.fetchall()

#     freq: Dict[str, int] = {}
#     for title, selftext in rows:
#         blob = ((title or "") + " " + (selftext or ""))[:12000]
#         toks = _tokenize(blob)
#         for t in toks:  # unigrams
#             freq[t] = freq.get(t, 0) + 1
#         for i in range(len(toks) - 1):  # bigrams
#             a, b = toks[i], toks[i+1]
#             if a in STOP or b in STOP:
#                 continue
#             bg = a + " " + b
#             if 3 <= len(bg) <= 30:
#                 freq[bg] = freq.get(bg, 0) + 1

#     candidates = [w for w, _ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)]
#     return candidates[:150]


# def select_expansions_with_llm(client: OpenAI, user_query: str, candidates: List[str]) -> List[str]:
#     """
#     Ask LLM to pick 8–12 expansions from mined candidate terms that are
#     specifically relevant to the user_query. No manual lists.
#     """
#     pool = "\n".join(f"- {c}" for c in candidates[:150])
#     sys = (
#         "Select the most relevant domain-specific tokens/phrases to help search a Reddit knowledge base.\n"
#         'Return STRICT JSON: {"expansions": []}. 8–12 items, lowercase, short (<=3 words), no duplicates.'
#     )
#     user = f"USER GOAL:\n{user_query}\n\nCANDIDATE TERMS (from similar KB threads):\n{pool}\n\nReturn JSON only."
#     resp = client.chat.completions.create(
#         model=CHAT_MODEL,
#         messages=[{"role": "system", "content": sys},
#                   {"role": "user", "content": user}],
#         temperature=0,
#         response_format={"type": "json_object"},
#     )
#     try:
#         data = json.loads(resp.choices[0].message.content)
#         exps = data.get("expansions", []) or []
#     except Exception:
#         exps = []
#     out, seen = [], set()
#     for w in exps:
#         if not isinstance(w, str):
#             continue
#         w = w.strip().lower()
#         if w and len(w) <= 40 and w not in seen:
#             seen.add(w); out.append(w)
#     return out[:12]


# # ---------- Subreddit discovery / filtering ----------
# def fetch_known_subreddits(conn) -> List[str]:
#     sql = """SELECT DISTINCT subreddit FROM submissions LIMIT 2000;"""
#     with conn.cursor() as cur:
#         cur.execute(sql)
#         return [r[0] for r in cur.fetchall() if r and r[0]]


# def select_candidate_subreddits(conn, known: List[str], plan: Dict[str, Any]) -> List[str]:
#     """
#     Light heuristic: if themes contain tokens that look like they name communities,
#     choose known subs whose name contains those tokens. Otherwise return empty -> no filter.
#     """
#     themes = (plan.get("themes") or []) + (plan.get("must_keywords") or [])
#     themes = [t for t in themes if 2 <= len(t) <= 18 and t.isascii()]
#     hits = []
#     for sub in known:
#         s = sub.lower()
#         for t in themes:
#             if t in s and len(t) >= 3:
#                 hits.append(sub)
#                 break
#     # de-dupe; if too few, return [] meaning no subreddit filter
#     uniq = list(dict.fromkeys(hits))
#     return uniq if len(uniq) >= 3 else []


# # ---------- Filtered retrieval (keywords + optional subs) ----------
# def nearest_submissions_filtered(conn,
#                                  qvec: List[float],
#                                  k: int,
#                                  must_keywords: List[str],
#                                  subreddits: List[str]) -> List[Hit]:
#     """
#     Apply keyword AND filters (ILIKE) and optional subreddit IN filter,
#     still ranking by vector distance. Returns up to k.
#     """
#     vec = vec_to_pg(qvec)

#     # de-dupe + sanitize inputs
#     must_keywords = [w for w in dict.fromkeys((must_keywords or [])) if isinstance(w, str) and w.strip()]
#     subreddits = [s for s in dict.fromkeys((subreddits or [])) if isinstance(s, str) and s.strip()]

#     where_clauses = ["s.embedding IS NOT NULL"]
#     where_params: List[Any] = []

#     # dynamic keyword AND filters (title+selftext)
#     text_expr = "(COALESCE(s.title,'') || ' ' || COALESCE(s.selftext,''))"
#     for kw in must_keywords:
#         where_clauses.append(f"{text_expr} ILIKE %s")
#         where_params.append(f"%{kw}%")

#     # optional subreddit filter via IN (%s) with tuple param
#     if subreddits:
#         where_clauses.append("s.subreddit IN %s")
#         where_params.append(tuple(subreddits))

#     where_sql = " AND ".join(where_clauses)

#     sql = f"""
#       SELECT
#         s.submission_id, s.domain_id, s.subreddit, s.title, s.selftext,
#         COALESCE(s.score,0) AS score,
#         COALESCE(s.upvote_ratio,0.0) AS upvote_ratio,
#         COALESCE(s.comment_count,0) AS comment_count,
#         s.permalink,
#         (s.embedding <-> %s::vector) AS dist
#       FROM submissions s
#       WHERE {where_sql}
#       ORDER BY s.embedding <-> %s::vector
#       LIMIT %s;
#     """

#     # IMPORTANT: placeholder order = [SELECT vec] + [WHERE params] + [ORDER BY vec, LIMIT]
#     final_params = [vec] + where_params + [vec, k]

#     with conn.cursor(cursor_factory=extras.DictCursor) as cur:
#         cur.execute(sql, final_params)
#         rows = cur.fetchall()

#     out: List[Hit] = []
#     for r in rows:
#         dist = float(r["dist"])
#         sim = 1.0 / (1.0 + dist)
#         out.append(Hit(
#             submission_id=r["submission_id"],
#             domain_id=r["domain_id"],
#             subreddit=r["subreddit"],
#             title=r["title"] or "",
#             selftext=r["selftext"] or "",
#             score=int(r["score"] or 0),
#             upvote_ratio=float(r["upvote_ratio"] or 0.0),
#             comment_count=int(r["comment_count"] or 0),
#             permalink=r["permalink"] or "",
#             dist=dist,
#             sim=sim,
#             comments=[]
#         ))
#     return out



# # ---------- Summarization ----------
# def build_llm_prompt(query: str, hit: Hit, max_comments: int = 6) -> str:
#     top_comments = hit.comments[:max_comments]
#     comments_text = "\n\n".join(
#         [f"- ({sc}) {b.strip()}"[:1000] for (_a, sc, b) in top_comments if (b or "").strip()]
#     )
#     prompt = f"""
# You are an assistant turning Reddit discussions into **1 concrete, practical step** toward a user's goal.

# USER GOAL:
# {query.strip()}

# SOURCE THREAD:
# Subreddit: r/{hit.subreddit}
# Title: {hit.title.strip()}

# Post:
# {hit.selftext.strip()[:2000]}

# Top comments (score in parens):
# {comments_text}

# TASK:
# - Propose one actionable step (start with a verb; keep it crisp).
# - Add a brief "How to do it" (1–3 sentences, specific).
# - If there's a caveat or precondition from the thread, include it in one short line ("Watch out for …").
# - Do NOT invent facts or numbers. Stay faithful to the text.

# Return strict JSON with keys:
# {{
#   "step": "...",
#   "how": "...",
#   "caveat": "..."  // may be empty
# }}
# """
#     return prompt


# def llm_summarize_step(client: OpenAI, prompt: str) -> dict:
#     resp = client.chat.completions.create(
#         model=SUMMARIZER_MODEL,
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.2,
#         response_format={"type": "json_object"},
#     )
#     try:
#         data = json.loads(resp.choices[0].message.content)
#         return {
#             "step": (data.get("step") or "").strip(),
#             "how": (data.get("how") or "").strip(),
#             "caveat": (data.get("caveat") or "").strip(),
#         }
#     except Exception:
#         return {"step": "", "how": "", "caveat": ""}


# # ---------- Orchestrator ----------
# def retrieve(query: str) -> List[dict]:
#     client = open_ai()
#     qvec = embed(client, query)

#     with open_db() as conn:
#         # plan from query (no hardcoded synonyms)
#         plan = llm_plan(client, query)

#         # pseudo-relevance feedback from KB -> candidate expansions
#         candidates = mine_kb_terms(conn, qvec, k=80)
#         expansions = select_expansions_with_llm(client, query, candidates)

#         # merge expansions into "nice" bucket
#         nice_plus = list(dict.fromkeys((plan.get("nice_keywords") or []) + expansions))[:20]
#         must = plan.get("must_keywords") or []
#         avoid = plan.get("avoid_keywords") or []  # used as veto later

#         known_subs = fetch_known_subreddits(conn)
#         subs = select_candidate_subreddits(conn, known_subs, plan)

#         # Progressive relaxation attempts
#         attempts = [
#             {"must": must, "subs": subs},
#             {"must": must, "subs": []},
#             {"must": (must + nice_plus), "subs": subs},
#             {"must": (must + nice_plus), "subs": []},
#             {"must": [], "subs": subs},
#             {"must": [], "subs": []},
#         ]

#         hits: List[Hit] = []
#         for a in attempts:
#             hits = nearest_submissions_filtered(conn, qvec, TOP_K, a["must"], a["subs"])
#             if hits:
#                 break

#         # fetch comments for heuristics + summarization context
#         for h in hits:
#             h.comments = fetch_comments_for(conn, h.submission_id, limit=COMMENT_SIP)

#     # simple avoid-keywords veto (post-filter)
#     if avoid:
#         av = [x.lower() for x in avoid]
#         filt = []
#         for h in hits:
#             blob = f"{h.title} {h.selftext}".lower()
#             if not any(a in blob for a in av):
#                 filt.append(h)
#         hits = filt or hits  # if all filtered, fall back

#     # rank
#     hits.sort(key=engagement_score, reverse=True)
#     top = hits[:FINAL_N]

#     # Summarize into steps
#     results = []
#     for h in tqdm(top, desc="Summarizing"):
#         prompt = build_llm_prompt(query, h)
#         summary = llm_summarize_step(client, prompt)
#         helpful = estimate_helpful_count(h)
#         results.append({
#             "step": summary.get("step") or h.title.strip()[:140] or "Do this first",
#             "how": summary.get("how"),
#             "caveat": summary.get("caveat"),
#             "people_found_helpful": helpful,
#             "subreddit": h.subreddit,
#             "permalink": h.permalink,
#             "score": h.score,
#             "upvote_ratio": h.upvote_ratio,
#         })
#     return results


# # ---------- CLI ----------
# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description="Retrieve actionable steps from the Reddit KB (no hardcoded synonyms).")
#     parser.add_argument("--query", "-q", required=True, help="User goal / constraints (plain text).")
#     args = parser.parse_args()

#     recs = retrieve(args.query)

#     for i, r in enumerate(recs, 1):
#         print(f"\n#{i}. {r['step']}")
#         print(f"   How: {r['how']}")
#         if r.get("caveat"):
#             print(f"   Caveat: {r['caveat']}")
#         print(f"   ~ {r['people_found_helpful']} people found this helpful")
#         print(f"   Source: r/{r['subreddit']}  {r['permalink']}")



#!/usr/bin/env python3
"""
Hybrid KB retrieval (pgvector + Postgres FTS over posts+comments) with LLM planning,
PRF expansions, and step summarization from Reddit threads already loaded in your DB.

Tables expected:
  submissions(submission_id text pk, subreddit text, title text, selftext text,
              score int, upvote_ratio float, comment_count int,
              permalink text, created_utc timestamp, embedding vector(1536))
  comments(comment_id text pk, submission_id text fk, author text, body text, score int, created_utc timestamp)

Env:
  DATABASE_URL = postgresql://postgres:dummy@127.0.0.1:5432/knowledgebase
  OPENAI_API_KEY = ...
  EMBED_MODEL = text-embedding-3-small (default)
  CHAT_MODEL = gpt-4o-mini (default)
  SUMMARIZER_MODEL = gpt-4o-mini (default)

Run:
  python retrieve_recommendations.py --query "Goal: I am tested with diabetes. I want to reverse it."
"""

# import os
# import re
# import json
# import math
# from typing import List, Tuple, Dict, Any
# from dataclasses import dataclass

# import psycopg2
# import psycopg2.extras as extras
# from dotenv import load_dotenv
# from openai import OpenAI
# from tqdm import tqdm

# # -------- Config --------
# load_dotenv()

# DB_DSN = os.getenv(
#     "DATABASE_URL",
#     "postgresql://postgres:dummy@127.0.0.1:5432/knowledgebase"
# ).replace("postgresql+psycopg2://", "postgresql://")

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
# CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
# SUMMARIZER_MODEL = os.getenv("SUMMARIZER_MODEL", "gpt-4o-mini")

# TOP_K = int(os.getenv("TOP_K", "100"))     # candidates from each channel
# FINAL_N = int(os.getenv("FINAL_N", "5"))   # final results
# COMMENT_SIP = int(os.getenv("COMMENT_SIP", "60"))

# HELP_PAT = re.compile(
#     r"(this worked|worked for me|helped me|thanks|thank you|game changer|fixed it|saved me|improved|it helped|life saver)",
#     re.I
# )

# STOP = {
#     "the","a","an","to","of","and","or","for","in","on","with","by","from","at","as","is","are","was","were",
#     "this","that","these","those","it","its","be","been","being","do","did","done","does","can","could","should",
#     "would","will","may","might","you","your","yours","i","me","my","we","our","they","their","them"
# }

# @dataclass
# class Hit:
#     submission_id: str
#     subreddit: str
#     title: str
#     selftext: str
#     score: int
#     upvote_ratio: float
#     comment_count: int
#     permalink: str
#     dist: float          # L2 distance (lower is better)
#     sim: float           # 1/(1+dist)
#     fts_rank: float      # ts_rank-ish score (higher is better)
#     comments: List[Tuple[str,int,str]]  # (author, score, body)

# # -------- Infra --------
# def open_db():
#     return psycopg2.connect(DB_DSN)

# def open_ai() -> OpenAI:
#     if not OPENAI_API_KEY:
#         raise SystemExit("Missing OPENAI_API_KEY.")
#     return OpenAI(api_key=OPENAI_API_KEY)

# def embed(client: OpenAI, text: str) -> List[float]:
#     return client.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding

# def vec_to_pg(vec: List[float]) -> str:
#     return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"

# # -------- Token utils (for PRF) --------
# def _tokenize(text: str) -> List[str]:
#     import re as _re
#     text = (text or "").lower()
#     text = _re.sub(r"[^a-z0-9\s]", " ", text)
#     toks = [t for t in text.split() if t and t not in STOP and len(t) > 1]
#     return toks[:400]

# # -------- LLM Planning (no hardcoded synonyms) --------
# def llm_plan(client: OpenAI, query: str) -> Dict[str, Any]:
#     sys = (
#         "Parse the user's goal into compact search directives for a Reddit knowledge base.\n"
#         "Return STRICT JSON with keys: must_keywords[], nice_keywords[], avoid_keywords[], themes[].\n"
#         "Rules: lowercase, 1–3 words each, domain terms preferred, no fluff, <=20 tokens total."
#     )
#     user = f"Goal:\n{query}\n\nReturn JSON only."
#     resp = client.chat.completions.create(
#         model=CHAT_MODEL,
#         messages=[{"role": "system", "content": sys},
#                   {"role": "user", "content": user}],
#         temperature=0,
#         response_format={"type": "json_object"},
#     )
#     try:
#         data = json.loads(resp.choices[0].message.content)
#     except Exception:
#         data = {}
#     def norm(lst):
#         out = []
#         for w in (lst or []):
#             if isinstance(w, str):
#                 w = w.strip().lower()
#                 if 0 < len(w) <= 40:
#                     out.append(w)
#         seen=set(); keep=[]
#         for w in out:
#             if w not in seen:
#                 seen.add(w); keep.append(w)
#         return keep[:12]
#     return {
#         "must_keywords": norm(data.get("must_keywords")),
#         "nice_keywords": norm(data.get("nice_keywords")),
#         "avoid_keywords": norm(data.get("avoid_keywords")),
#         "themes":        norm(data.get("themes")),
#     }

# # -------- PRF mining from your KB --------
# def mine_kb_terms(conn, qvec: List[float], k: int = 120) -> List[str]:
#     vec = vec_to_pg(qvec)
#     sql = """
#       SELECT s.title, s.selftext
#       FROM submissions s
#       WHERE s.embedding IS NOT NULL
#       ORDER BY s.embedding <-> %s::vector
#       LIMIT %s;
#     """
#     with conn.cursor() as cur:
#         cur.execute(sql, (vec, k))
#         rows = cur.fetchall()
#     freq: Dict[str,int] = {}
#     for title, selftext in rows:
#         blob = ((title or "") + " " + (selftext or ""))[:16000]
#         toks = _tokenize(blob)
#         for t in toks:  # unigrams
#             freq[t] = freq.get(t, 0) + 1
#         for i in range(len(toks)-1):  # bigrams
#             a,b = toks[i], toks[i+1]
#             if a in STOP or b in STOP: 
#                 continue
#             bg = a + " " + b
#             if 3 <= len(bg) <= 30:
#                 freq[bg] = freq.get(bg, 0) + 1
#     return [w for w,_ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)][:150]

# def select_expansions_with_llm(client: OpenAI, user_query: str, candidates: List[str]) -> List[str]:
#     pool = "\n".join(f"- {c}" for c in candidates[:150])
#     sys = (
#         "From the candidate pool, select 8–12 domain-specific tokens/phrases most relevant to the user goal.\n"
#         "Return STRICT JSON: {\"expansions\": []}. lowercase, <=3 words, no duplicates."
#     )
#     user = f"USER GOAL:\n{user_query}\n\nCANDIDATES:\n{pool}\n\nReturn JSON only."
#     resp = client.chat.completions.create(
#         model=CHAT_MODEL,
#         messages=[{"role": "system", "content": sys},
#                   {"role": "user", "content": user}],
#         temperature=0,
#         response_format={"type": "json_object"},
#     )
#     try:
#         data = json.loads(resp.choices[0].message.content)
#         exps = data.get("expansions", []) or []
#     except Exception:
#         exps = []
#     out, seen = [], set()
#     for w in exps:
#         if not isinstance(w, str): 
#             continue
#         w = w.strip().lower()
#         if w and len(w) <= 40 and w not in seen:
#             seen.add(w); out.append(w)
#     return out[:12]

# # -------- Comments & helpfulness --------
# def fetch_comments_for(conn, submission_id: str, limit: int = 60) -> List[Tuple[str,int,str]]:
#     sql = """
#       SELECT COALESCE(author,''), COALESCE(score,0), COALESCE(body,'')
#       FROM comments
#       WHERE submission_id = %s
#       ORDER BY score DESC
#       LIMIT %s;
#     """
#     with conn.cursor() as cur:
#         cur.execute(sql, (submission_id, limit))
#         return cur.fetchall()

# def estimate_helpful_count(hit: Hit) -> int:
#     affirm = 0
#     for (_a, _s, b) in hit.comments:
#         if HELP_PAT.search(b or ""):
#             affirm += 1
#     bonus = max(0, int(round((hit.upvote_ratio * max(hit.score, 0)) / 10.0)))
#     return affirm + bonus

# def engagement_score(hit: Hit) -> float:
#     eng = math.log1p(max(hit.score, 0) + max(hit.comment_count, 0))
#     norm_eng = min(1.0, eng / 10.0)
#     return norm_eng

# # -------- Core retrieval (hybrid) --------
# def semantic_candidates(conn, qvec: List[float], k: int) -> List[Hit]:
#     vec = vec_to_pg(qvec)
#     sql = """
#       SELECT
#         s.submission_id, s.subreddit, s.title, s.selftext,
#         COALESCE(s.score,0) AS score,
#         COALESCE(s.upvote_ratio,0.0) AS upvote_ratio,
#         COALESCE(s.comment_count,0) AS comment_count,
#         s.permalink,
#         (s.embedding <-> %s::vector) AS dist
#       FROM submissions s
#       WHERE s.embedding IS NOT NULL
#       ORDER BY s.embedding <-> %s::vector
#       LIMIT %s;
#     """
#     with conn.cursor(cursor_factory=extras.DictCursor) as cur:
#         cur.execute(sql, (vec, vec, k))
#         rows = cur.fetchall()
#     out=[]
#     for r in rows:
#         dist=float(r["dist"]); sim=1.0/(1.0+dist)
#         out.append(Hit(
#             submission_id=r["submission_id"],
#             subreddit=r["subreddit"],
#             title=r["title"] or "",
#             selftext=r["selftext"] or "",
#             score=int(r["score"] or 0),
#             upvote_ratio=float(r["upvote_ratio"] or 0.0),
#             comment_count=int(r["comment_count"] or 0),
#             permalink=r["permalink"] or "",
#             dist=dist, sim=sim, fts_rank=0.0, comments=[]
#         ))
#     return out

# def fts_candidates(conn, query_text: str, must: List[str], nice: List[str], k: int) -> List[Hit]:
#     """
#     Full-text over title+selftext+aggregated comments.
#     We use websearch_to_tsquery for natural parsing.
#     """
#     # Build a compact websearch string from must + nice (fallback to raw query_text)
#     parts = []
#     for w in (must or []):
#         parts.append(w)
#     for w in (nice or []):
#         parts.append(w)
#     web_q = " ".join(dict.fromkeys([p for p in parts if p])) or query_text

#     sql = """
#     WITH cm AS (
#       SELECT c.submission_id, string_agg(COALESCE(c.body,''), ' ') AS corpus
#       FROM comments c
#       GROUP BY c.submission_id
#     )
#     SELECT
#       s.submission_id, s.subreddit, s.title, s.selftext,
#       COALESCE(s.score,0) AS score,
#       COALESCE(s.upvote_ratio,0.0) AS upvote_ratio,
#       COALESCE(s.comment_count,0) AS comment_count,
#       s.permalink,
#       ts_rank_cd(
#         to_tsvector('english', COALESCE(s.title,'') || ' ' || COALESCE(s.selftext,'') || ' ' || COALESCE(cm.corpus,'')),
#         websearch_to_tsquery('english', %s)
#       ) AS fts_rank
#     FROM submissions s
#     LEFT JOIN cm ON cm.submission_id = s.submission_id
#     WHERE websearch_to_tsquery('english', %s) @@
#           to_tsvector('english', COALESCE(s.title,'') || ' ' || COALESCE(s.selftext,'') || ' ' || COALESCE(cm.corpus,''))
#     ORDER BY fts_rank DESC
#     LIMIT %s;
#     """
#     with conn.cursor(cursor_factory=extras.DictCursor) as cur:
#         cur.execute(sql, (web_q, web_q, k))
#         rows = cur.fetchall()
#     out=[]
#     for r in rows:
#         out.append(Hit(
#             submission_id=r["submission_id"],
#             subreddit=r["subreddit"],
#             title=r["title"] or "",
#             selftext=r["selftext"] or "",
#             score=int(r["score"] or 0),
#             upvote_ratio=float(r["upvote_ratio"] or 0.0),
#             comment_count=int(r["comment_count"] or 0),
#             permalink=r["permalink"] or "",
#             dist=0.0, sim=0.0, fts_rank=float(r["fts_rank"] or 0.0),
#             comments=[]
#         ))
#     return out

# def merge_and_rerank(conn, sem: List[Hit], fts: List[Hit], final_n: int) -> List[Hit]:
#     # Index by submission_id; merge channels, keep best fields
#     by_id: Dict[str, Hit] = {}
#     for h in sem + fts:
#         if h.submission_id not in by_id:
#             by_id[h.submission_id] = h
#         else:
#             x = by_id[h.submission_id]
#             # keep better semantic / fts signals
#             x.sim = max(x.sim, h.sim)
#             x.dist = min(x.dist if x.dist>0 else 1e9, h.dist if h.dist>0 else 1e9)
#             x.fts_rank = max(x.fts_rank, h.fts_rank)
#             # keep metadata if missing
#             if not x.title and h.title: x.title = h.title
#             if not x.selftext and h.selftext: x.selftext = h.selftext
#             if not x.permalink and h.permalink: x.permalink = h.permalink
#             x.score = max(x.score, h.score)
#             x.comment_count = max(x.comment_count, h.comment_count)
#             x.upvote_ratio = max(x.upvote_ratio, h.upvote_ratio)

#     hits = list(by_id.values())

#     # fetch comments for heuristics + summarization
#     for h in hits:
#         h.comments = fetch_comments_for(conn, h.submission_id, limit=COMMENT_SIP)

#     # Normalize signals
#     max_sim = max([h.sim for h in hits] + [1e-6])
#     max_fts = max([h.fts_rank for h in hits] + [1e-6])
#     # Ranking blend: tuneable weights (feels close to web flow)
#     A, B, C = 0.55, 0.25, 0.20  # semantic, text, social
#     def rank(h: Hit) -> float:
#         sem_s = h.sim / max_sim if max_sim>0 else 0.0
#         txt_s = h.fts_rank / max_fts if max_fts>0 else 0.0
#         soc_s = engagement_score(h)
#         return A*sem_s + B*txt_s + C*soc_s

#     hits.sort(key=rank, reverse=True)
#     return hits[:final_n]

# # -------- Summarization to single step --------
# def build_llm_prompt(query: str, h: Hit, max_comments: int = 8) -> str:
#     cs = h.comments[:max_comments]
#     ctext = "\n\n".join([f"- ({sc}) {b.strip()}"[:1000] for (_a, sc, b) in cs if (b or "").strip()])
#     return f"""
# You are an assistant turning a Reddit thread into **1 concrete, practical step** toward the user's goal.

# USER GOAL:
# {query.strip()}

# SOURCE THREAD:
# Subreddit: r/{h.subreddit}
# Title: {h.title.strip()}

# Post:
# {h.selftext.strip()[:2000]}

# Top comments (score in parens):
# {ctext}

# TASK:
# - Propose one actionable step (start with a verb; keep it crisp).
# - Add a brief "How to do it" (1–3 sentences, specific).
# - If there's a caveat or precondition mentioned, include it in one short line ("Watch out for …").
# - Stay faithful to the text. Do NOT invent facts.

# Return strict JSON with keys exactly:
# {{
#   "step": "...",
#   "how": "...",
#   "caveat": "..."  // may be empty
# }}
# """.strip()

# def llm_summarize_step(client: OpenAI, prompt: str) -> dict:
#     resp = client.chat.completions.create(
#         model=SUMMARIZER_MODEL,
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.2,
#         response_format={"type": "json_object"},
#     )
#     try:
#         data = json.loads(resp.choices[0].message.content)
#         return {
#             "step": (data.get("step") or "").strip(),
#             "how": (data.get("how") or "").strip(),
#             "caveat": (data.get("caveat") or "").strip(),
#         }
#     except Exception:
#         return {"step": "", "how": "", "caveat": ""}

# # -------- Orchestrator --------
# def retrieve(query: str) -> List[dict]:
#     client = open_ai()
#     qvec = embed(client, query)

#     with open_db() as conn:
#         # plan + PRF (mirrors your web flow but local to KB)
#         plan = llm_plan(client, query)
#         candidates = mine_kb_terms(conn, qvec, k=120)
#         expansions = select_expansions_with_llm(client, query, candidates)

#         must = plan.get("must_keywords") or []
#         nice = list(dict.fromkeys((plan.get("nice_keywords") or []) + expansions))[:20]

#         # channel 1: semantic
#         sem = semantic_candidates(conn, qvec, TOP_K)

#         # channel 2: FTS over posts+comments (BM25-ish rank)
#         fts = fts_candidates(conn, query, must, nice, TOP_K)

#         # merge + rerank
#         top_hits = merge_and_rerank(conn, sem, fts, FINAL_N)

#     # Summarize into steps
#     out = []
#     for h in tqdm(top_hits, desc="Summarizing"):
#         prompt = build_llm_prompt(query, h)
#         summary = llm_summarize_step(client, prompt)
#         helpful = estimate_helpful_count(h)
#         out.append({
#             "step": summary.get("step") or (h.title.strip()[:140] or "Do this first"),
#             "how": summary.get("how"),
#             "caveat": summary.get("caveat"),
#             "people_found_helpful": helpful,
#             "subreddit": h.subreddit,
#             "permalink": h.permalink,
#             "score": h.score,
#             "upvote_ratio": h.upvote_ratio,
#         })
#     return out

# # -------- CLI --------
# if __name__ == "__main__":
#     import argparse
#     p = argparse.ArgumentParser(description="Retrieve actionable steps from the Reddit KB (hybrid pgvector + FTS).")
#     p.add_argument("--query", "-q", required=True, help="User goal / constraints (plain text).")
#     args = p.parse_args()

#     recs = retrieve(args.query)

#     for i, r in enumerate(recs, 1):
#         print(f"\n#{i}. {r['step']}")
#         print(f"   How: {r['how']}")
#         if r.get("caveat"):
#             print(f"   Caveat: {r['caveat']}")
#         print(f"   ~ {r['people_found_helpful']} people found this helpful")
#         print(f"   Source: r/{r['subreddit']}  {r['permalink']}")

#!/usr/bin/env python3
"""
Hybrid KB retrieval (pgvector HNSW over submissions + comments) + FTS + LLM planning/PRF.

Tables expected:
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
    embedding vector(1536)              -- REQUIRED
  )

  comments(
    comment_id text primary key,
    submission_id text references submissions(submission_id),
    author text,
    body text,
    score int,
    created_utc timestamp,
    embedding vector(1536)              -- OPTIONAL but recommended (script can backfill)
  )

  domain(
    domain_id text primary key,
    domain_tag text,
    domain_name text
  )

Environment:
  DATABASE_URL = postgresql://postgres:dummy@127.0.0.1:5432/knowledgebase
  OPENAI_API_KEY = ...
  EMBED_MODEL = text-embedding-3-small (default)
  CHAT_MODEL = gpt-4o-mini (default)
  SUMMARIZER_MODEL = gpt-4o-mini (default)

Run:
  python retrieve_recommendations.py --query "Goal: I am tested with diabetes. I want to reverse it."
"""

# import os
# import re
# import json
# import math
# from typing import List, Tuple, Dict, Any, Optional
# from dataclasses import dataclass

# import psycopg2
# import psycopg2.extras as extras
# from dotenv import load_dotenv
# from openai import OpenAI
# from tqdm import tqdm

# # ---------- Config ----------
# load_dotenv()

# DB_DSN = os.getenv(
#     "DATABASE_URL",
#     "postgresql://postgres:dummy@127.0.0.1:5432/knowledgebase"
# ).replace("postgresql+psycopg2://", "postgresql://")

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
# CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
# SUMMARIZER_MODEL = os.getenv("SUMMARIZER_MODEL", "gpt-4o-mini")

# # Candidate pool sizes
# TOP_K_SUB = int(os.getenv("TOP_K_SUB", "120"))         # semantic subs
# TOP_K_COM = int(os.getenv("TOP_K_COM", "200"))         # semantic comments
# TOP_K_FTS = int(os.getenv("TOP_K_FTS", "120"))         # FTS
# FINAL_N    = int(os.getenv("FINAL_N", "5"))            # final results
# COMMENT_SIP = int(os.getenv("COMMENT_SIP", "80"))      # comments fetched for heuristics/summaries
# BACKFILL_COMMENTS = int(os.getenv("BACKFILL_COMMENTS", "0"))  # 0 disables; else N to backfill embeddings

# #Replace this with a compatible library
# HELP_PAT = re.compile(
#     r"(this worked|worked for me|helped me|thanks|thank you|game changer|fixed it|saved me|improved|it helped|life saver)",
#     re.I
# )
# STOP = {
#     "the","a","an","to","of","and","or","for","in","on","with","by","from","at","as","is","are","was","were",
#     "this","that","these","those","it","its","be","been","being","do","did","done","does","can","could","should",
#     "would","will","may","might","you","your","yours","i","me","my","we","our","they","their","them"
# }

# @dataclass
# class SubHit:
#     submission_id: str
#     domain_id: Optional[str]
#     subreddit: str
#     title: str
#     selftext: str
#     score: int
#     upvote_ratio: float
#     comment_count: int
#     permalink: str
#     dist: float
#     sim: float
#     fts_rank: float
#     best_comment_dist: float
#     best_comment_sim: float
#     comments: List[Tuple[str,int,str]]  # (author, score, body)
#     domain_tag: Optional[str]
#     domain_name: Optional[str]

# # ---------- Infra ----------
# def open_db():
#     return psycopg2.connect(DB_DSN)

# def open_ai() -> OpenAI:
#     if not OPENAI_API_KEY:
#         raise SystemExit("Missing OPENAI_API_KEY.")
#     return OpenAI(api_key=OPENAI_API_KEY)

# def embed(client: OpenAI, text: str) -> List[float]:
#     return client.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding

# def vec_to_pg(vec: List[float]) -> str:
#     # vector literal for pgvector
#     return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"

# # ---------- Setup: pgvector + HNSW ----------
# def ensure_pgvector_hnsw(conn):
#     """
#     Create pgvector extension and HNSW indexes if missing.
#     Note: HNSW requires pgvector >= 0.5.0 and Postgres 12+.
#     """
#     with conn.cursor() as cur:
#         cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

#         # submissions embedding HNSW
#         cur.execute("""
#           DO $$
#           BEGIN
#             IF NOT EXISTS (
#               SELECT 1 FROM pg_indexes WHERE schemaname='public' AND indexname='idx_submissions_embedding_hnsw'
#             ) THEN
#               EXECUTE 'CREATE INDEX idx_submissions_embedding_hnsw ON public.submissions USING hnsw (embedding vector_l2_ops) WITH (m=16, ef_construction=200)';
#             END IF;
#           END$$;
#         """)

#         # comments embedding HNSW (only if column exists)
#         cur.execute("""
#           DO $$
#           BEGIN
#             IF EXISTS (
#               SELECT 1 FROM information_schema.columns
#               WHERE table_schema='public' AND table_name='comments' AND column_name='embedding'
#             ) AND NOT EXISTS (
#               SELECT 1 FROM pg_indexes WHERE schemaname='public' AND indexname='idx_comments_embedding_hnsw'
#             ) THEN
#               EXECUTE 'CREATE INDEX idx_comments_embedding_hnsw ON public.comments USING hnsw (embedding vector_l2_ops) WITH (m=16, ef_construction=200)';
#             END IF;
#           END$$;
#         """)
#     conn.commit()

# # ---------- Optional: backfill comment embeddings ----------
# def backfill_comment_embeddings(conn, client: OpenAI, limit_rows: int):
#     """
#     If comments.embedding is NULL, batch-fill for top-scored comments.
#     Set BACKFILL_COMMENTS>0 to enable (recommend first run to precompute offline).
#     """
#     if limit_rows <= 0:
#         return
#     with conn.cursor() as cur:
#         cur.execute("""
#           SELECT COUNT(*) FROM information_schema.columns
#           WHERE table_schema='public' AND table_name='comments' AND column_name='embedding';
#         """)
#         has_col = cur.fetchone()[0] == 1
#     if not has_col:
#         with conn.cursor() as cur:
#             cur.execute("ALTER TABLE public.comments ADD COLUMN embedding vector(1536);")
#         conn.commit()

#     with conn.cursor(cursor_factory=extras.DictCursor) as cur:
#         cur.execute("""
#           SELECT comment_id, body
#           FROM public.comments
#           WHERE embedding IS NULL
#           ORDER BY score DESC NULLS LAST
#           LIMIT %s;
#         """, (limit_rows,))
#         rows = cur.fetchall()

#     if not rows:
#         return

#     # Embed in small batches to reduce rate limits
#     BATCH = 64
#     for i in tqdm(range(0, len(rows), BATCH), desc="Backfilling comment embeddings"):
#         chunk = rows[i:i+BATCH]
#         texts = [(r["body"] or "")[:8000] for r in chunk]
#         embs = openai_embed_many(client, texts)
#         with conn.cursor() as cur:
#             for r, e in zip(chunk, embs):
#                 cur.execute(
#                     "UPDATE public.comments SET embedding=%s::vector WHERE comment_id=%s;",
#                     (vec_to_pg(e), r["comment_id"])
#                 )
#         conn.commit()

# def openai_embed_many(client: OpenAI, texts: List[str]) -> List[List[float]]:
#     # Multi-input embedding
#     resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
#     return [d.embedding for d in resp.data]

# # ---------- Token & PRF ----------
# def _tokenize(text: str) -> List[str]:
#     import re as _re
#     text = (text or "").lower()
#     text = _re.sub(r"[^a-z0-9\s]", " ", text)
#     toks = [t for t in text.split() if t and t not in STOP and len(t) > 1]
#     return toks[:400]

# def llm_plan(client: OpenAI, query: str) -> Dict[str, Any]:
#     sys = (
#         "Parse the user's goal into compact search directives for a Reddit knowledge base.\n"
#         "Return STRICT JSON: must_keywords[], nice_keywords[], avoid_keywords[], themes[].\n"
#         "Lowercase, 1–3 words, <=20 tokens total, domain terms preferred."
#     )
#     user = f"Goal:\n{query}\n\nReturn JSON only."
#     resp = client.chat.completions.create(
#         model=CHAT_MODEL,
#         messages=[{"role": "system", "content": sys},
#                   {"role": "user", "content": user}],
#         temperature=0,
#         response_format={"type": "json_object"},
#     )
#     try:
#         data = json.loads(resp.choices[0].message.content)
#     except Exception:
#         data = {}
#     def norm(lst):
#         out=[]; seen=set()
#         for w in (lst or []):
#             if isinstance(w,str):
#                 w=w.strip().lower()
#                 if 0 < len(w) <= 40 and w not in seen:
#                     seen.add(w); out.append(w)
#         return out[:12]
#     return {
#         "must_keywords": norm(data.get("must_keywords")),
#         "nice_keywords": norm(data.get("nice_keywords")),
#         "avoid_keywords": norm(data.get("avoid_keywords")),
#         "themes":        norm(data.get("themes")),
#     }

# def mine_kb_terms(conn, qvec: List[float], k: int = 150) -> List[str]:
#     vec = vec_to_pg(qvec)
#     sql = """
#       SELECT s.title, s.selftext
#       FROM public.submissions s
#       WHERE s.embedding IS NOT NULL
#       ORDER BY s.embedding <-> %s::vector
#       LIMIT %s;
#     """
#     with conn.cursor() as cur:
#         cur.execute(sql, (vec, k))
#         rows = cur.fetchall()
#     freq: Dict[str,int] = {}
#     for title, selftext in rows:
#         blob = ((title or "") + " " + (selftext or ""))[:16000]
#         toks = _tokenize(blob)
#         for t in toks:
#             freq[t] = freq.get(t, 0) + 1
#         for i in range(len(toks)-1):
#             a, b = toks[i], toks[i+1]
#             if a in STOP or b in STOP: 
#                 continue
#             bg = a + " " + b
#             if 3 <= len(bg) <= 30:
#                 freq[bg] = freq.get(bg, 0) + 1
#     return [w for w,_ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)][:180]

# def select_expansions_with_llm(client: OpenAI, user_query: str, candidates: List[str]) -> List[str]:
#     pool = "\n".join(f"- {c}" for c in candidates[:180])
#     sys = (
#         "From the candidate pool, select 8–12 domain-specific tokens/phrases most relevant to the user's goal.\n"
#         "Return STRICT JSON: {\"expansions\": []}. Lowercase, <=3 words, no duplicates."
#     )
#     user = f"USER GOAL:\n{user_query}\n\nCANDIDATES:\n{pool}\n\nReturn JSON only."
#     resp = client.chat.completions.create(
#         model=CHAT_MODEL,
#         messages=[{"role": "system", "content": sys},
#                   {"role": "user", "content": user}],
#         temperature=0,
#         response_format={"type": "json_object"},
#     )
#     try:
#         data = json.loads(resp.choices[0].message.content)
#         exps = data.get("expansions", []) or []
#     except Exception:
#         exps = []
#     out=[]; seen=set()
#     for w in exps:
#         if isinstance(w,str):
#             w=w.strip().lower()
#             if w and len(w)<=40 and w not in seen:
#                 seen.add(w); out.append(w)
#     return out[:12]

# # ---------- Heuristics ----------
# def fetch_comments_for(conn, submission_id: str, limit: int) -> List[Tuple[str,int,str]]:
#     sql = """
#       SELECT COALESCE(author,''), COALESCE(score,0), COALESCE(body,'')
#       FROM public.comments
#       WHERE submission_id=%s
#       ORDER BY score DESC
#       LIMIT %s;
#     """
#     with conn.cursor() as cur:
#         cur.execute(sql, (submission_id, limit))
#         return cur.fetchall()

# def estimate_helpful_count(h: SubHit) -> int:
#     affirm=0
#     for (_a, _s, b) in h.comments:
#         if HELP_PAT.search(b or ""):
#             affirm += 1
#     bonus = max(0, int(round((h.upvote_ratio * max(h.score, 0)) / 10.0)))
#     return affirm + bonus

# def social_engagement(h: SubHit) -> float:
#     eng = math.log1p(max(h.score,0) + max(h.comment_count,0))
#     return min(1.0, eng/10.0)

# # ---------- Retrieval Channels ----------
# def candidates_from_submissions(conn, qvec: List[float], k: int) -> Dict[str, SubHit]:
#     vec = vec_to_pg(qvec)
#     sql = """
#       SELECT s.submission_id, s.domain_id, s.subreddit, s.title, s.selftext,
#              COALESCE(s.score,0) AS score, COALESCE(s.upvote_ratio,0.0) AS upvote_ratio,
#              COALESCE(s.comment_count,0) AS comment_count, s.permalink,
#              (s.embedding <-> %s::vector) AS dist,
#              d.domain_tag, d.domain_name
#       FROM public.submissions s
#       LEFT JOIN public.domain d ON d.domain_id = s.domain_id
#       WHERE s.embedding IS NOT NULL
#       ORDER BY s.embedding <-> %s::vector
#       LIMIT %s;
#     """
#     with conn.cursor(cursor_factory=extras.DictCursor) as cur:
#         cur.execute(sql, (vec, vec, k))
#         rows = cur.fetchall()

#     out: Dict[str, SubHit] = {}
#     for r in rows:
#         dist=float(r["dist"]); sim=1.0/(1.0+dist)
#         out[r["submission_id"]] = SubHit(
#             submission_id=r["submission_id"],
#             domain_id=r["domain_id"],
#             subreddit=r["subreddit"],
#             title=r["title"] or "",
#             selftext=r["selftext"] or "",
#             score=int(r["score"] or 0),
#             upvote_ratio=float(r["upvote_ratio"] or 0.0),
#             comment_count=int(r["comment_count"] or 0),
#             permalink=r["permalink"] or "",
#             dist=dist, sim=sim, fts_rank=0.0,
#             best_comment_dist=1e9, best_comment_sim=0.0,
#             comments=[],
#             domain_tag=r.get("domain_tag"),
#             domain_name=r.get("domain_name"),
#         )
#     return out

# def candidates_from_comments(conn, qvec: List[float], k: int) -> Dict[str, Tuple[str,float,float]]:
#     """
#     Return mapping submission_id -> (comment_id, best_dist, best_sim)
#     Uses HNSW index on comments.embedding.
#     If comments.embedding doesn't exist, returns empty.
#     """
#     with conn.cursor() as cur:
#         cur.execute("""
#           SELECT COUNT(*) FROM information_schema.columns
#           WHERE table_schema='public' AND table_name='comments' AND column_name='embedding';
#         """)
#         has_col = cur.fetchone()[0] == 1
#     if not has_col:
#         return {}

#     vec = vec_to_pg(qvec)
#     sql = """
#       SELECT c.comment_id, c.submission_id,
#              (c.embedding <-> %s::vector) AS dist
#       FROM public.comments c
#       WHERE c.embedding IS NOT NULL
#       ORDER BY c.embedding <-> %s::vector
#       LIMIT %s;
#     """
#     with conn.cursor(cursor_factory=extras.DictCursor) as cur:
#         cur.execute(sql, (vec, vec, k))
#         rows = cur.fetchall()

#     best: Dict[str, Tuple[str,float,float]] = {}
#     for r in rows:
#         dist=float(r["dist"]); sim=1.0/(1.0+dist)
#         sid = r["submission_id"]
#         prev = best.get(sid)
#         if (prev is None) or (dist < prev[1]):
#             best[sid] = (r["comment_id"], dist, sim)
#     return best

# def fts_candidates(conn, query_text: str, must: List[str], nice: List[str], k: int) -> Dict[str, float]:
#     """
#     Full-text over title+selftext+aggregated comments, returns submission_id -> fts_rank
#     """
#     parts = []
#     parts += [w for w in (must or []) if w]
#     parts += [w for w in (nice or []) if w]
#     web_q = " ".join(dict.fromkeys(parts)) or query_text

#     sql = """
#     WITH cm AS (
#       SELECT c.submission_id, string_agg(COALESCE(c.body,''), ' ') AS corpus
#       FROM public.comments c
#       GROUP BY c.submission_id
#     )
#     SELECT s.submission_id,
#            ts_rank_cd(
#              to_tsvector('english', COALESCE(s.title,'') || ' ' || COALESCE(s.selftext,'') || ' ' || COALESCE(cm.corpus,'')),
#              websearch_to_tsquery('english', %s)
#            ) AS fts_rank
#     FROM public.submissions s
#     LEFT JOIN cm ON cm.submission_id = s.submission_id
#     WHERE websearch_to_tsquery('english', %s) @@
#           to_tsvector('english', COALESCE(s.title,'') || ' ' || COALESCE(s.selftext,'') || ' ' || COALESCE(cm.corpus,''))
#     ORDER BY fts_rank DESC
#     LIMIT %s;
#     """
#     with conn.cursor(cursor_factory=extras.DictCursor) as cur:
#         cur.execute(sql, (web_q, web_q, k))
#         rows = cur.fetchall()
#     return {r["submission_id"]: float(r["fts_rank"] or 0.0) for r in rows}

# # ---------- Rank merge ----------
# def merge_and_rank(conn,
#                    sub_hits: Dict[str, SubHit],
#                    com_best: Dict[str, Tuple[str,float,float]],
#                    fts_rank: Dict[str, float],
#                    final_n: int) -> List[SubHit]:

#     # apply best comment sim to each sub
#     for sid, (cid, cdist, csim) in com_best.items():
#         if sid in sub_hits:
#             h = sub_hits[sid]
#             h.best_comment_dist = min(h.best_comment_dist, cdist)
#             h.best_comment_sim = max(h.best_comment_sim, csim)
#         else:
#             # comment surfaced a submission we didn't have from sub-semantic channel; fetch its row
#             with conn.cursor(cursor_factory=extras.DictCursor) as cur:
#                 cur.execute("""
#                   SELECT s.submission_id, s.domain_id, s.subreddit, s.title, s.selftext,
#                          COALESCE(s.score,0), COALESCE(s.upvote_ratio,0.0),
#                          COALESCE(s.comment_count,0), s.permalink,
#                          d.domain_tag, d.domain_name
#                   FROM public.submissions s
#                   LEFT JOIN public.domain d ON d.domain_id = s.domain_id
#                   WHERE s.submission_id=%s;
#                 """, (sid,))
#                 r = cur.fetchone()
#             if r:
#                 sub_hits[sid] = SubHit(
#                     submission_id=r["submission_id"],
#                     domain_id=r["domain_id"],
#                     subreddit=r["subreddit"],
#                     title=r["title"] or "",
#                     selftext=r["selftext"] or "",
#                     score=int(r["coalesce"] if "coalesce" in r else r[6] if len(r)>=7 else 0),  # fallback safety
#                     upvote_ratio=float(r["coalesce"] if "coalesce" in r else r[7] if len(r)>=8 else 0.0),
#                     comment_count=int(r["coalesce"] if "coalesce" in r else r[8] if len(r)>=9 else 0),
#                     permalink=r["permalink"] or "",
#                     dist=1.0, sim=0.0, fts_rank=0.0,
#                     best_comment_dist=cdist, best_comment_sim=csim,
#                     comments=[],
#                     domain_tag=r.get("domain_tag"),
#                     domain_name=r.get("domain_name"),
#                 )

#     # apply fts rank
#     for sid, fr in fts_rank.items():
#         if sid in sub_hits:
#             sub_hits[sid].fts_rank = max(sub_hits[sid].fts_rank, fr)
#         else:
#             # enrich from DB minimal if missing
#             with conn.cursor(cursor_factory=extras.DictCursor) as cur:
#                 cur.execute("""
#                   SELECT s.submission_id, s.domain_id, s.subreddit, s.title, s.selftext,
#                          COALESCE(s.score,0), COALESCE(s.upvote_ratio,0.0),
#                          COALESCE(s.comment_count,0), s.permalink,
#                          d.domain_tag, d.domain_name
#                   FROM public.submissions s
#                   LEFT JOIN public.domain d ON d.domain_id = s.domain_id
#                   WHERE s.submission_id=%s;
#                 """, (sid,))
#                 r = cur.fetchone()
#             if r:
#                 sub_hits[sid] = SubHit(
#                     submission_id=r["submission_id"],
#                     domain_id=r["domain_id"],
#                     subreddit=r["subreddit"],
#                     title=r["title"] or "",
#                     selftext=r["selftext"] or "",
#                     score=int(r["coalesce"] if "coalesce" in r else r[6] if len(r)>=7 else 0),
#                     upvote_ratio=float(r["coalesce"] if "coalesce" in r else r[7] if len(r)>=8 else 0.0),
#                     comment_count=int(r["coalesce"] if "coalesce" in r else r[8] if len(r)>=9 else 0),
#                     permalink=r["permalink"] or "",
#                     dist=1.0, sim=0.0, fts_rank=fr,
#                     best_comment_dist=1e9, best_comment_sim=0.0,
#                     comments=[],
#                     domain_tag=r.get("domain_tag"),
#                     domain_name=r.get("domain_name"),
#                 )

#     hits = list(sub_hits.values())

#     # hydrate comments for heuristics & summarization
#     for h in hits:
#         h.comments = fetch_comments_for(conn, h.submission_id, COMMENT_SIP)

#     # normalize signals
#     max_sim  = max([h.sim for h in hits] + [1e-6])
#     max_csim = max([h.best_comment_sim for h in hits] + [1e-6])
#     max_fts  = max([h.fts_rank for h in hits] + [1e-6])

#     # weights: tune freely
#     W_SUB   = 0.45   # submission semantic
#     W_COM   = 0.25   # best comment semantic
#     W_FTS   = 0.20   # BM25-ish
#     W_SOC   = 0.10   # social signal

#     def rank(h: SubHit) -> float:
#         s_sub = (h.sim / max_sim) if max_sim>0 else 0.0
#         s_com = (h.best_comment_sim / max_csim) if max_csim>0 else 0.0
#         s_fts = (h.fts_rank / max_fts) if max_fts>0 else 0.0
#         s_soc = social_engagement(h)
#         return W_SUB*s_sub + W_COM*s_com + W_FTS*s_fts + W_SOC*s_soc

#     hits.sort(key=rank, reverse=True)
#     return hits[:final_n]

# # ---------- Summarization ----------
# def build_llm_prompt(query: str, h: SubHit, max_comments: int = 8) -> str:
#     cs = h.comments[:max_comments]
#     ctext = "\n\n".join([f"- ({sc}) {b.strip()}"[:1000] for (_a, sc, b) in cs if (b or "").strip()])
#     return f"""
# You are an assistant turning a Reddit thread into **1 concrete, practical step** toward the user's goal.

# USER GOAL:
# {query.strip()}

# SOURCE THREAD:
# Subreddit: r/{h.subreddit}
# Title: {h.title.strip()}

# Post:
# {h.selftext.strip()[:2000]}

# Top comments (score in parens):
# {ctext}

# TASK:
# - Propose one actionable step (start with a verb; keep it crisp).
# - Add a brief "How to do it" (1–3 sentences, specific).
# - If there's a caveat or precondition mentioned, include it in one short line ("Watch out for …").
# - Stay faithful to the text. Do NOT invent facts.

# Return strict JSON with keys exactly:
# {{
#   "step": "...",
#   "how": "...",
#   "caveat": "..."  // may be empty
# }}
# """.strip()

# def llm_summarize_step(client: OpenAI, prompt: str) -> dict:
#     resp = client.chat.completions.create(
#         model=SUMMARIZER_MODEL,
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.2,
#         response_format={"type": "json_object"},
#     )
#     try:
#         data = json.loads(resp.choices[0].message.content)
#         return {
#             "step": (data.get("step") or "").strip(),
#             "how": (data.get("how") or "").strip(),
#             "caveat": (data.get("caveat") or "").strip(),
#         }
#     except Exception:
#         return {"step": "", "how": "", "caveat": ""}

# # ---------- Orchestrator ----------
# def retrieve(query: str) -> List[dict]:
#     client = open_ai()
#     qvec = embed(client, query)

#     with open_db() as conn:
#         ensure_pgvector_hnsw(conn)
#         if BACKFILL_COMMENTS > 0:
#             backfill_comment_embeddings(conn, client, BACKFILL_COMMENTS)

#         # plan + PRF
#         plan = llm_plan(client, query)
#         candidates = mine_kb_terms(conn, qvec, k=150)
#         expansions = select_expansions_with_llm(client, query, candidates)

#         must = plan.get("must_keywords") or []
#         nice = list(dict.fromkeys((plan.get("nice_keywords") or []) + expansions))[:20]

#         # Channel 1: submissions via pgvector/HNSW
#         sub_hits = candidates_from_submissions(conn, qvec, TOP_K_SUB)

#         # Channel 2: comments via pgvector/HNSW -> best per submission
#         com_best = candidates_from_comments(conn, qvec, TOP_K_COM)

#         # Channel 3: FTS over post+comments
#         fts_rank = fts_candidates(conn, query, must, nice, TOP_K_FTS)

#         # Merge & rank
#         top_hits = merge_and_rank(conn, sub_hits, com_best, fts_rank, FINAL_N)

#     # Summarize
#     out = []
#     for h in tqdm(top_hits, desc="Summarizing"):
#         summary = llm_summarize_step(client, build_llm_prompt(query, h))
#         out.append({
#             "step": summary.get("step") or (h.title.strip()[:140] or "Do this first"),
#             "how": summary.get("how"),
#             "caveat": summary.get("caveat"),
#             "people_found_helpful": estimate_helpful_count(h),
#             "subreddit": h.subreddit,
#             "permalink": h.permalink,
#             "domain_id": h.domain_id,
#             "domain_tag": h.domain_tag,
#             "domain_name": h.domain_name,
#             "score": h.score,
#             "upvote_ratio": h.upvote_ratio,
#         })
#     return out

# # ---------- CLI ----------
# if __name__ == "__main__":
#     import argparse
#     p = argparse.ArgumentParser(description="Retrieve actionable steps (HNSW: submissions + comments) from Reddit KB.")
#     p.add_argument("--query", "-q", required=True, help="User goal / constraints (plain text).")
#     args = p.parse_args()

#     recs = retrieve(args.query)
#     for i, r in enumerate(recs, 1):
#         print(f"\n#{i}. {r['step']}")
#         print(f"   How: {r['how']}")
#         if r.get("caveat"):
#             print(f"   Caveat: {r['caveat']}")
#         print(f"   ~ {r['people_found_helpful']} people found this helpful")
#         dom = f" | Domain: {r['domain_tag'] or ''} {r['domain_name'] or ''}".strip()
#         print(f"   Source: r/{r['subreddit']}  {r['permalink']}{dom}")



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
