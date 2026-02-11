"""Adaptive Reddit search service for live data fetching."""
import json
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Set, Tuple, Optional
from dataclasses import dataclass

import praw
from openai import AsyncOpenAI
from loguru import logger

from lib.db import Database


@dataclass
class AdaptiveSearchResult:
    """Result from adaptive search."""
    subreddits_suggested: List[str]
    subreddits_validated: List[str]
    submissions_fetched: int
    comments_fetched: int
    submissions_persisted: int
    comments_persisted: int
    submissions: List[Dict[str, Any]]
    comments: List[Dict[str, Any]]


class AdaptiveSearch:
    """Adaptive search service for fetching live Reddit data."""

    def __init__(
        self,
        db: Database,
        openai_client: AsyncOpenAI,
        reddit_client_id: Optional[str] = None,
        reddit_client_secret: Optional[str] = None,
        reddit_user_agent: Optional[str] = None
    ):
        """Initialize adaptive search with database and API clients."""
        self.db = db
        self.openai = openai_client
        
        # Reddit credentials
        self.reddit_client_id = reddit_client_id or os.getenv("REDDIT_CLIENT_ID")
        self.reddit_client_secret = reddit_client_secret or os.getenv("REDDIT_CLIENT_SECRET")
        self.reddit_user_agent = reddit_user_agent or os.getenv("REDDIT_USER_AGENT", "knowledge-collection-service/0.1")
        
        # Thread pool for sync PRAW operations
        self._executor = ThreadPoolExecutor(max_workers=3)
        
        # Load allowlist from domains.json
        self.allowlist = self._load_allowlist()
        logger.info(f"[ADAPTIVE] Loaded allowlist with {len(self.allowlist)} subreddits")

    def _load_allowlist(self) -> Set[str]:
        """Load subreddit allowlist from domains.json."""
        allowlist = set()
        domains_path = os.path.join(os.path.dirname(__file__), "..", "domains.json")
        
        try:
            with open(domains_path, "r") as f:
                domains = json.load(f)
            
            for domain_data in domains.values():
                subreddits = domain_data.get("subreddits", [])
                for sub in subreddits:
                    allowlist.add(sub.lower())
            
            logger.info(f"[ADAPTIVE:ALLOWLIST] Loaded {len(allowlist)} subreddits from domains.json")
        except Exception as e:
            logger.error(f"[ADAPTIVE:ALLOWLIST] Failed to load domains.json: {e}")
            # Fallback to a minimal allowlist
            allowlist = {"excons", "felons", "jobs4felons", "careerguidance", "personalfinance"}
        
        return allowlist

    async def suggest_subreddits(
        self,
        query: str,
        user_story: str,
        max_subreddits: int = 3
    ) -> Tuple[List[str], str]:
        """
        Use LLM to suggest relevant subreddits for the query.
        
        Returns:
            Tuple of (list of subreddit names, LLM reasoning)
        """
        logger.info(f"[ADAPTIVE:LLM] Requesting subreddit suggestions for query: \"{query[:50]}...\"")
        
        prompt = f"""You are helping find the most relevant Reddit communities for someone seeking advice.

User Background:
{user_story}

Their Question:
{query}

Based on this, suggest up to {max_subreddits} subreddit names (without the r/ prefix) that would have the most helpful advice for this person.

Consider subreddits that:
1. Have active communities with supportive members
2. Focus on the specific challenges this person faces
3. Have practical, actionable advice in their posts

Return your response as JSON in this exact format:
{{"subreddits": ["subreddit1", "subreddit2", "subreddit3"], "reasoning": "Brief explanation of why these subreddits"}}

Only return the JSON, no other text."""

        try:
            response = await self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that suggests relevant Reddit communities. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            raw_response = response.choices[0].message.content.strip()
            logger.info(f"[ADAPTIVE:LLM] Raw LLM response: {raw_response}")
            
            # Parse JSON response
            # Handle potential markdown code blocks
            if raw_response.startswith("```"):
                raw_response = raw_response.split("\n", 1)[1] if "\n" in raw_response else raw_response
                if raw_response.endswith("```"):
                    raw_response = raw_response.rsplit("```", 1)[0]
                raw_response = raw_response.strip()
            
            parsed = json.loads(raw_response)
            subreddits = parsed.get("subreddits", [])
            reasoning = parsed.get("reasoning", "")
            
            logger.info(f"[ADAPTIVE:LLM] Suggested subreddits: {subreddits}")
            return subreddits, reasoning
            
        except json.JSONDecodeError as e:
            logger.error(f"[ADAPTIVE:LLM] Failed to parse LLM response as JSON: {e}")
            return [], "Failed to parse LLM response"
        except Exception as e:
            logger.error(f"[ADAPTIVE:LLM] Error during subreddit suggestion: {e}")
            return [], str(e)

    def validate_subreddits(self, suggestions: List[str]) -> Tuple[List[str], List[str]]:
        """
        Validate suggested subreddits against the allowlist.
        
        Returns:
            Tuple of (validated subreddits, rejected subreddits)
        """
        logger.info(f"[ADAPTIVE:VALIDATE] Checking against allowlist ({len(self.allowlist)} subreddits)")
        
        validated = []
        rejected = []
        
        for sub in suggestions:
            sub_lower = sub.lower()
            if sub_lower in self.allowlist:
                validated.append(sub_lower)
            else:
                rejected.append(sub)
        
        logger.info(f"[ADAPTIVE:VALIDATE] Validated subreddits: {validated} ({len(rejected)} rejected: {rejected})")
        return validated, rejected

    def _fetch_reddit_posts_sync(
        self,
        subreddits: List[str],
        posts_per_subreddit: int = 10,
        comments_per_post: int = 5
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Synchronous Reddit fetching using PRAW (runs in thread pool).
        
        Returns:
            Tuple of (submissions, comments)
        """
        submissions = []
        comments = []
        
        try:
            reddit = praw.Reddit(
                client_id=self.reddit_client_id,
                client_secret=self.reddit_client_secret,
                user_agent=self.reddit_user_agent
            )
            
            for sub_name in subreddits:
                logger.info(f"[ADAPTIVE:PRAW] Fetching hot/top posts from r/{sub_name}...")
                
                try:
                    subreddit = reddit.subreddit(sub_name)
                    
                    # Fetch hot posts
                    post_count = 0
                    comment_count = 0
                    
                    for submission in subreddit.hot(limit=posts_per_subreddit):
                        # Skip stickied posts
                        if submission.stickied:
                            continue
                        
                        # Extract submission data
                        sub_data = {
                            "submission_id": submission.id,
                            "title": submission.title,
                            "selftext": submission.selftext or "",
                            "permalink": submission.permalink,
                            "score": submission.score,
                            "subreddit": sub_name,
                            "created_utc": submission.created_utc
                        }
                        submissions.append(sub_data)
                        post_count += 1
                        
                        # Fetch top comments
                        submission.comments.replace_more(limit=0)
                        for comment in submission.comments[:comments_per_post]:
                            if hasattr(comment, 'body') and comment.body and comment.body != "[deleted]":
                                comment_data = {
                                    "comment_id": comment.id,
                                    "submission_id": submission.id,
                                    "body": comment.body,
                                    "score": comment.score,
                                    "permalink": comment.permalink if hasattr(comment, 'permalink') else f"{submission.permalink}{comment.id}",
                                    "created_utc": comment.created_utc if hasattr(comment, 'created_utc') else None
                                }
                                comments.append(comment_data)
                                comment_count += 1
                    
                    logger.info(f"[ADAPTIVE:PRAW] r/{sub_name}: Found {post_count} posts, fetched {comment_count} comments")
                    
                except Exception as e:
                    logger.error(f"[ADAPTIVE:PRAW] Error fetching from r/{sub_name}: {e}")
                    continue
            
            logger.info(f"[ADAPTIVE:PRAW] Total fetched: {len(submissions)} posts, {len(comments)} comments")
            
        except Exception as e:
            logger.error(f"[ADAPTIVE:PRAW] Reddit client error: {e}")
        
        return submissions, comments

    async def fetch_reddit_posts(
        self,
        subreddits: List[str],
        posts_per_subreddit: int = 10,
        comments_per_post: int = 5
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Fetch posts and comments from Reddit asynchronously.
        
        Returns:
            Tuple of (submissions, comments)
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._fetch_reddit_posts_sync,
            subreddits,
            posts_per_subreddit,
            comments_per_post
        )

    async def embed_content(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts using OpenAI."""
        if not texts:
            return []
        
        logger.info(f"[ADAPTIVE:EMBED] Embedding {len(texts)} texts...")
        
        embeddings = []
        # Batch in groups of 100 (OpenAI limit)
        batch_size = 100
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = await self.openai.embeddings.create(
                model="text-embedding-3-small",
                input=batch
            )
            for item in response.data:
                embeddings.append(item.embedding)
        
        logger.info(f"[ADAPTIVE:EMBED] Generated {len(embeddings)} embeddings")
        return embeddings

    async def embed_and_filter(
        self,
        submissions: List[Dict[str, Any]],
        comments: List[Dict[str, Any]],
        query_embedding: List[float],
        threshold: float = 0.45
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Embed fetched content and filter by similarity to query.
        
        Returns:
            Tuple of (filtered submissions, filtered comments)
        """
        from scipy.spatial.distance import cosine
        import numpy as np
        
        # Embed submissions
        if submissions:
            logger.info(f"[ADAPTIVE:EMBED] Embedding {len(submissions)} submissions...")
            sub_texts = [f"{s['title']} {s['selftext']}" for s in submissions]
            sub_embeddings = await self.embed_content(sub_texts)
            
            for i, emb in enumerate(sub_embeddings):
                submissions[i]["embedding"] = emb
                submissions[i]["distance"] = cosine(query_embedding, emb)
        
        # Embed comments
        if comments:
            logger.info(f"[ADAPTIVE:EMBED] Embedding {len(comments)} comments...")
            comment_texts = [c["body"] for c in comments]
            comment_embeddings = await self.embed_content(comment_texts)
            
            for i, emb in enumerate(comment_embeddings):
                comments[i]["embedding"] = emb
                comments[i]["distance"] = cosine(query_embedding, emb)
        
        # Filter by threshold
        logger.info(f"[ADAPTIVE:FILTER] Filtering by similarity (threshold: {threshold})")
        
        filtered_submissions = [s for s in submissions if s.get("distance", 1.0) <= threshold]
        filtered_comments = [c for c in comments if c.get("distance", 1.0) <= threshold]
        
        logger.info(f"[ADAPTIVE:FILTER] Submissions: {len(submissions)} -> {len(filtered_submissions)} passed threshold")
        logger.info(f"[ADAPTIVE:FILTER] Comments: {len(comments)} -> {len(filtered_comments)} passed threshold")
        
        return filtered_submissions, filtered_comments

    async def persist_to_kb(
        self,
        submissions: List[Dict[str, Any]],
        comments: List[Dict[str, Any]],
        domain_id: str = "adaptive"
    ) -> Tuple[int, int, int, int]:
        """
        Persist submissions and comments to the knowledge base.
        
        Returns:
            Tuple of (submissions_inserted, submissions_skipped, comments_inserted, comments_skipped)
        """
        subs_inserted = 0
        subs_skipped = 0
        comments_inserted = 0
        comments_skipped = 0
        
        logger.info(f"[ADAPTIVE:PERSIST] Inserting {len(submissions)} submissions into KB...")
        
        # Insert submissions
        for sub in submissions:
            try:
                # Check if already exists
                existing = await self.db.fetchval(
                    "SELECT 1 FROM submissions WHERE submission_id = $1",
                    sub["submission_id"]
                )
                
                if existing:
                    subs_skipped += 1
                    continue
                
                # Insert new submission
                embedding_json = json.dumps(sub["embedding"])
                await self.db.execute(
                    """
                    INSERT INTO submissions (submission_id, domain_id, title, selftext, permalink, embedding)
                    VALUES ($1, $2, $3, $4, $5, $6::vector)
                    ON CONFLICT (submission_id) DO NOTHING
                    """,
                    sub["submission_id"],
                    domain_id,
                    sub["title"],
                    sub["selftext"],
                    sub["permalink"],
                    embedding_json
                )
                subs_inserted += 1
                
            except Exception as e:
                logger.error(f"[ADAPTIVE:PERSIST] Error inserting submission {sub.get('submission_id')}: {e}")
                subs_skipped += 1
        
        logger.info(f"[ADAPTIVE:PERSIST] Skipped {subs_skipped} duplicates, inserted {subs_inserted} new submissions")
        
        logger.info(f"[ADAPTIVE:PERSIST] Inserting {len(comments)} comments into KB...")
        
        # Insert comments
        for comment in comments:
            try:
                # Check if already exists
                existing = await self.db.fetchval(
                    "SELECT 1 FROM comments WHERE id = $1",
                    comment["comment_id"]
                )
                
                if existing:
                    comments_skipped += 1
                    continue
                
                # Insert new comment
                embedding_json = json.dumps(comment["embedding"])
                await self.db.execute(
                    """
                    INSERT INTO comments (id, submission_id, body, score, permalink, embedding)
                    VALUES ($1, $2, $3, $4, $5, $6::vector)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    comment["comment_id"],
                    comment["submission_id"],
                    comment["body"],
                    comment["score"],
                    comment.get("permalink", ""),
                    embedding_json
                )
                comments_inserted += 1
                
            except Exception as e:
                logger.error(f"[ADAPTIVE:PERSIST] Error inserting comment {comment.get('comment_id')}: {e}")
                comments_skipped += 1
        
        logger.info(f"[ADAPTIVE:PERSIST] Skipped {comments_skipped} duplicates, inserted {comments_inserted} new comments")
        
        return subs_inserted, subs_skipped, comments_inserted, comments_skipped

    async def run_adaptive_search(
        self,
        query: str,
        user_story: str,
        query_embedding: List[float],
        distance_threshold: float = 0.55,
        max_subreddits: int = 3,
        posts_per_subreddit: int = 10,
        comments_per_post: int = 5
    ) -> AdaptiveSearchResult:
        """
        Run the full adaptive search pipeline.
        
        Returns:
            AdaptiveSearchResult with all fetched and persisted data
        """
        logger.info("[ADAPTIVE] Starting adaptive search pipeline")
        
        # Step 1: LLM suggests subreddits
        suggested, reasoning = await self.suggest_subreddits(query, user_story, max_subreddits)
        
        if not suggested:
            logger.warning("[ADAPTIVE] No subreddits suggested by LLM")
            return AdaptiveSearchResult(
                subreddits_suggested=[],
                subreddits_validated=[],
                submissions_fetched=0,
                comments_fetched=0,
                submissions_persisted=0,
                comments_persisted=0,
                submissions=[],
                comments=[]
            )
        
        # Step 2: Validate against allowlist
        validated, rejected = self.validate_subreddits(suggested)
        
        if not validated:
            logger.warning(f"[ADAPTIVE] No subreddits passed validation. Rejected: {rejected}")
            # Fall back to some default subreddits from allowlist
            validated = list(self.allowlist)[:3]
            logger.info(f"[ADAPTIVE] Falling back to default subreddits: {validated}")
        
        # Step 3: Fetch from Reddit
        submissions, comments = await self.fetch_reddit_posts(
            validated,
            posts_per_subreddit,
            comments_per_post
        )
        
        if not submissions:
            logger.warning("[ADAPTIVE] No submissions fetched from Reddit")
            return AdaptiveSearchResult(
                subreddits_suggested=suggested,
                subreddits_validated=validated,
                submissions_fetched=0,
                comments_fetched=0,
                submissions_persisted=0,
                comments_persisted=0,
                submissions=[],
                comments=[]
            )
        
        # Step 4: Embed and filter
        filtered_submissions, filtered_comments = await self.embed_and_filter(
            submissions,
            comments,
            query_embedding,
            distance_threshold
        )
        
        # Step 5: Persist to KB
        subs_inserted, subs_skipped, comments_inserted, comments_skipped = await self.persist_to_kb(
            filtered_submissions,
            filtered_comments
        )
        
        logger.info(f"[ADAPTIVE] Adaptive search complete: {subs_inserted} submissions, {comments_inserted} comments added to KB")
        
        return AdaptiveSearchResult(
            subreddits_suggested=suggested,
            subreddits_validated=validated,
            submissions_fetched=len(submissions),
            comments_fetched=len(comments),
            submissions_persisted=subs_inserted,
            comments_persisted=comments_inserted,
            submissions=filtered_submissions,
            comments=filtered_comments
        )
