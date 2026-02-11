"""RAG pipeline implementation based on test.ipynb."""
import json
import numpy as np
from typing import List, Dict, Any, AsyncGenerator, Optional
from openai import AsyncOpenAI
from scipy.spatial.distance import cosine
from hdbscan import HDBSCAN
from loguru import logger

from lib.db import Database
from rag.sql import (
    get_submissions_query,
    get_comments_query,
    format_embedding,
    parse_submissions,
    parse_comments
)
from rag.types import HyDEResult, ActionStep, RankedStep
from rag.adaptive import AdaptiveSearch


class RagPipeline:
    """RAG pipeline for generating actionable recommendations."""

    def __init__(self, db: Database, openai_client: AsyncOpenAI):
        """Initialize pipeline with database and OpenAI client."""
        self.db = db
        self.client = openai_client
        # Initialize adaptive search service
        self.adaptive_search = AdaptiveSearch(db, openai_client)

    async def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for the raw user query."""
        logger.info("[PIPELINE] Generating query embedding...")
        emb_response = await self.client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        embedding = emb_response.data[0].embedding
        logger.info("[PIPELINE] Query embedding generated")
        return embedding

    async def generate_hyde(
        self, user_story: str, current_steps: str, query: str
    ) -> HyDEResult:
        """Generate HyDE hypothetical Reddit post with persona context."""
        logger.info("[PIPELINE] Generating HyDE post...")
        
        hyde_prompt = f"""
You are an extremely empathetic Reddit user writing a raw, honest submission asking for help.
Write a detailed (150–250 word) Reddit post (title + body) from someone in exactly the same situation as described below.
Make it the kind of thoughtful, non-ranty, "I know I need to change, please help me" post that gets tons of supportive, practical, life-changing advice in the comments.

Background / User Story:
{user_story}

Steps Already Taken:
{current_steps}

Current Goal / Query:
{query}

Start directly with the post (include a realistic title and then the body). 
Do NOT write advice or a success story — only the help-seeking post.
"""

        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a Reddit user asking for help in a detailed, vulnerable way."
                },
                {"role": "user", "content": hyde_prompt}
            ],
            temperature=0.8,
            max_tokens=500
        )

        hyde_post = response.choices[0].message.content.strip()
        logger.info(f"[PIPELINE] HyDE post generated ({len(hyde_post)} chars)")

        # Generate embedding for HyDE post
        emb_response = await self.client.embeddings.create(
            model="text-embedding-3-small",
            input=hyde_post
        )
        embedding = emb_response.data[0].embedding
        logger.info("[PIPELINE] HyDE embedding generated")

        # For now, return empty domains list (classifier not implemented in notebook)
        return {
            "hyde_post": hyde_post,
            "domains": [],
            "embedding": embedding
        }

    async def retrieve_submissions(
        self, embedding: List[float], max_submissions: int = 15, distance_threshold: float = 0.55
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant submissions by vector similarity."""
        logger.info(f"[PIPELINE] Retrieving submissions (max={max_submissions}, threshold={distance_threshold})...")
        
        query = get_submissions_query()
        emb_str = format_embedding(embedding)
        
        rows = await self.db.fetch(query, emb_str, max_submissions)
        submissions = parse_submissions(rows)

        logger.info(f"[PIPELINE] Retrieved {len(submissions)} submissions from KB")
        print(json.dumps(submissions, indent=5))
        
        # Filter by distance threshold
        filtered = [s for s in submissions if s["distance"] <= distance_threshold]
        logger.info(f"[PIPELINE] Filtered to {len(filtered)} submissions (distance <= {distance_threshold})")
        return filtered

    async def retrieve_comments(
        self, embedding: List[float], submission_ids: List[str], max_comments: int = 20
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant comments from submissions with score boost."""
        if not submission_ids:
            return []
        
        logger.info(f"[PIPELINE] Retrieving comments for {len(submission_ids)} submissions...")
            
        query = get_comments_query()
        emb_str = format_embedding(embedding)
        
        rows = await self.db.fetch(query, emb_str, submission_ids, max_comments)
        comments = parse_comments(rows)
        logger.info(f"[PIPELINE] Retrieved {len(comments)} comments")
        return comments

    async def extract_steps_from_comment(self, comment_body: str) -> List[str]:
        """Extract actionable steps from a single comment."""
        action_step_prompt = """
You will be given a single Reddit comment.
Extract ONLY clear actionable steps expressed or strongly implied.
Do NOT add new ideas. Do NOT give advice. Do NOT restate context.
Return JSON exactly as: {"steps": ["...", "..."]}

COMMENT:
"""
        
        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Extract only actionable, literal steps."},
                {"role": "user", "content": action_step_prompt + comment_body}
            ],
            temperature=0.2,
            max_tokens=120
        )

        raw = response.choices[0].message.content.strip()
        
        try:
            # Strip markdown code blocks if present
            cleaned = raw
            if cleaned.startswith("```"):
                cleaned = cleaned.split('\n', 1)[1] if '\n' in cleaned else cleaned
                if cleaned.endswith("```"):
                    cleaned = cleaned.rsplit("```", 1)[0]
            
            cleaned = cleaned.strip()
            parsed = json.loads(cleaned)
            steps = parsed.get("steps", [])
            return steps
        except Exception:
            return []

    async def extract_all_steps(
        self, comments: List[Dict[str, Any]]
    ) -> List[ActionStep]:
        """Extract and embed all actionable steps from comments."""
        logger.info(f"[PIPELINE] Extracting steps from {len(comments)} comments...")
        all_steps = []

        for comment in comments:
            steps_text = await self.extract_steps_from_comment(comment["body"])
            
            for step_text in steps_text:
                # Embed each step
                emb_response = await self.client.embeddings.create(
                    model="text-embedding-3-small",
                    input=step_text
                )
                embedding = emb_response.data[0].embedding

                all_steps.append({
                    "step": step_text,
                    "embedding": embedding,
                    "submission_id": comment["submission_id"],
                    "permalink": comment["permalink"],
                    "score": comment["score"]
                })

        logger.info(f"[PIPELINE] Extracted {len(all_steps)} actionable steps")
        return all_steps

    def cluster_and_rank_steps(
        self, all_steps: List[ActionStep], query_embedding: List[float]
    ) -> List[RankedStep]:
        """Cluster steps using HDBSCAN and rank by usefulness."""
        logger.info(f"[PIPELINE] Clustering {len(all_steps)} steps...")
        
        if not all_steps:
            return []

        # Convert to numpy array
        X = np.array([s["embedding"] for s in all_steps])
        
        # Cluster using HDBSCAN
        clusterer = HDBSCAN(
            metric="euclidean",
            min_cluster_size=2,
            min_samples=1,
            cluster_selection_epsilon=0.15
        )
        
        labels = clusterer.fit_predict(X)
        
        # Log cluster distribution for debugging
        unique_labels, counts = np.unique(labels, return_counts=True)
        logger.info(f"[PIPELINE] Cluster distribution: {dict(zip(unique_labels.tolist(), counts.tolist()))}")
        
        # Attach cluster id to steps
        for i, s in enumerate(all_steps):
            s["cluster"] = int(labels[i])
        
        # Group by cluster (filter out noise -1)
        clusters = {}
        for s in all_steps:
            cid = s["cluster"]
            if cid == -1:
                continue
            clusters.setdefault(cid, []).append(s)
        
        logger.info(f"[PIPELINE] Found {len(clusters)} clusters (excluding noise)")
        for cid, items in clusters.items():
            logger.info(f"[PIPELINE] Cluster {cid}: {len(items)} steps. Sample: {items[0]['step'][:50]}...")
        
        # Filter clusters by value (crowd validation)
        filtered_clusters = []
        for cid, items in clusters.items():
            src_scores = [s["score"] for s in items]
            if not src_scores:
                continue
            
            # Use a more robust cluster value calculation
            avg_score = np.mean(src_scores)
            max_score = np.max(src_scores)
            cluster_value = avg_score / max_score if max_score > 0 else 0
            
            logger.info(f"[PIPELINE] Cluster {cid} value: {cluster_value:.4f} (avg={avg_score:.1f}, max={max_score})")
            
            if cluster_value > 0.10: # Lowered from 0.20 to be less aggressive
                filtered_clusters.append({
                    "cluster_id": cid,
                    "items": items,
                    "cluster_value": cluster_value,
                    "frequency": len(items)
                })
            else:
                logger.info(f"[PIPELINE] Cluster {cid} rejected due to low value ({cluster_value:.4f} <= 0.10)")
        
        logger.info(f"[PIPELINE] {len(filtered_clusters)} clusters passed validation")
        
        # Select best representative from each cluster
        action_steps = []
        q = np.array(query_embedding)
        
        if not filtered_clusters and all_steps:
            logger.info("[PIPELINE] No clusters formed. Falling back to top individual steps by relevance.")
            # Fallback: Sort all steps by relevance to query and take top 5
            all_steps_ranked = []
            for s in all_steps:
                relevance = 1 - cosine(s["embedding"], q)
                all_steps_ranked.append({
                    "url": s["permalink"],
                    "step": s["step"],
                    "usefulness": float(relevance),
                    "cluster_count": 1,
                    "cluster_id": -1
                })
            all_steps_ranked.sort(key=lambda x: x["usefulness"], reverse=True)
            return all_steps_ranked[:5]

        for cluster in filtered_clusters:
            items = cluster["items"]
            step_embeds = np.array([np.array(item["embedding"], dtype=float) for item in items])
            
            # Compute centroid
            centroid = np.mean(step_embeds, axis=0)
            
            # Find best representative (closest to centroid)
            distances = [cosine(e, centroid) for e in step_embeds]
            best_index = int(np.argmin(distances))
            best_item = items[best_index]
            
            # Calculate usefulness score
            relevance = 1 - cosine(best_item["embedding"], q)
            centroid_sim = 1 - cosine(best_item["embedding"], centroid)
            cluster_value = cluster["cluster_value"]
            
            usefulness = 0.5 * relevance + 0.3 * centroid_sim + 0.2 * cluster_value
            
            action_steps.append({
                "url": best_item["permalink"],
                "step": best_item["step"],
                "usefulness": float(usefulness),
                "cluster_count": cluster["frequency"],
                "cluster_id": cluster["cluster_id"]
            })
        
        # Sort by usefulness
        action_steps.sort(key=lambda x: x["usefulness"], reverse=True)
        
        logger.info(f"[PIPELINE] Ranked {len(action_steps)} steps")
        return action_steps

    async def _run_adaptive_and_retry(
        self,
        query: str,
        user_story: str,
        query_embedding: List[float],
        hyde_embedding: List[float],
        distance_threshold: float,
        max_submissions: int,
        max_comments: int
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Run adaptive search and retry the full retrieval -> extraction -> clustering pipeline.
        Uses the original query embedding for adaptive search (more targeted than HyDE).
        Yields stage events throughout.
        """
        logger.info("[PIPELINE] Running adaptive search with original query embedding...")
        
        # Run adaptive search with QUERY embedding (not HyDE)
        adaptive_result = await self.adaptive_search.run_adaptive_search(
            query=query,
            user_story=user_story,
            query_embedding=query_embedding,  # Use original query embedding
            distance_threshold=0.60,  # Looser threshold for live adaptive data
            max_subreddits=3,
            posts_per_subreddit=10,
            comments_per_post=5
        )
        
        # Yield adaptive subreddits stage
        yield {
            "stage": "adaptive_subreddits",
            "suggested": adaptive_result.subreddits_suggested,
            "validated": adaptive_result.subreddits_validated
        }
        
        # Yield adaptive fetch stage
        yield {
            "stage": "adaptive_fetch",
            "submissions_fetched": adaptive_result.submissions_fetched,
            "comments_fetched": adaptive_result.comments_fetched
        }
        
        # Yield adaptive persist stage
        yield {
            "stage": "adaptive_persist",
            "submissions_persisted": adaptive_result.submissions_persisted,
            "comments_persisted": adaptive_result.comments_persisted
        }
        
        # Re-query KB with HyDE embedding (for consistency with main flow)
        logger.info("[PIPELINE] Re-querying KB after adaptive search...")
        submissions = await self.retrieve_submissions(
            hyde_embedding,
            max_submissions,
            distance_threshold
        )
        
        yield {
            "stage": "submissions_updated",
            "items": submissions,
            "count": len(submissions),
            "source": "adaptive"
        }
        
        if not submissions:
            yield {
                "stage": "error",
                "message": "No relevant submissions found even after adaptive search",
                "where": "adaptive_submissions_retrieval"
            }
            return
        
        # Retrieve comments
        submission_ids = [s["submission_id"] for s in submissions]
        comments = await self.retrieve_comments(
            hyde_embedding,
            submission_ids,
            max_comments
        )
        
        yield {
            "stage": "comments_updated",
            "items": comments,
            "count": len(comments),
            "source": "adaptive"
        }
        
        # Extract steps
        all_steps = await self.extract_all_steps(comments)
        
        if not all_steps:
            yield {
                "stage": "error",
                "message": "No actionable steps extracted after adaptive search",
                "where": "adaptive_step_extraction"
            }
            return
        
        yield {
            "stage": "steps_updated",
            "count": len(all_steps),
            "sample": [s["step"] for s in all_steps[:5]],
            "source": "adaptive"
        }
        
        # Cluster and rank
        ranked_steps = self.cluster_and_rank_steps(all_steps, hyde_embedding)
        
        yield {
            "stage": "ranking_updated",
            "clusters": ranked_steps,
            "source": "adaptive"
        }
        
        # Final stage
        yield {
            "stage": "final",
            "final_steps": ranked_steps,
            "sources": list(set([s["permalink"] for s in submissions])),
            "source": "adaptive"
        }

    async def run_pipeline(
        self,
        user_story: str,
        current_steps: str,
        query: str,
        max_submissions: int = 15,
        max_comments: int = 20,
        distance_threshold: float = 0.55,
        adaptive_search: bool = False
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Run the full RAG pipeline and yield stage events."""
        
        logger.info(f"[PIPELINE] Starting pipeline (adaptive_search={adaptive_search})")
        
        try:
            # Generate query embedding first (used for adaptive search if needed)
            query_embedding = await self.generate_query_embedding(query)
            
            # Stage 1: Generate HyDE
            hyde_result = await self.generate_hyde(user_story, current_steps, query)
            yield {
                "stage": "hyde",
                "hyde_post": hyde_result["hyde_post"],
                "domains": hyde_result["domains"]
            }
            
            # Stage 2: Retrieve submissions
            submissions = await self.retrieve_submissions(
                hyde_result["embedding"],
                max_submissions,
                distance_threshold
            )
            
            yield {
                "stage": "submissions",
                "items": submissions,
                "count": len(submissions)
            }
            
            # Stage 2.5: If adaptive search enabled and no KB results, trigger adaptive flow
            if adaptive_search and len(submissions) == 0:
                logger.info("[PIPELINE] KB returned 0 submissions, triggering adaptive search...")
                
                async for event in self._run_adaptive_and_retry(
                    query=query,
                    user_story=user_story,
                    query_embedding=query_embedding,
                    hyde_embedding=hyde_result["embedding"],
                    distance_threshold=0.60,  # Use looser threshold for the retry as well
                    max_submissions=max_submissions,
                    max_comments=max_comments
                ):
                    yield event
                
                logger.info("[PIPELINE] Pipeline complete (via adaptive path)")
                return
            
            elif not submissions:
                yield {
                    "stage": "error",
                    "message": "No relevant submissions found in KB. Enable Adaptive Search to fetch live data.",
                    "where": "submissions_retrieval"
                }
                return
            
            # Stage 3: Retrieve comments
            submission_ids = [s["submission_id"] for s in submissions]
            comments = await self.retrieve_comments(
                hyde_result["embedding"],
                submission_ids,
                max_comments
            )
            
            yield {
                "stage": "comments",
                "items": comments,
                "count": len(comments)
            }
            
            # Stage 4: Extract steps
            all_steps = await self.extract_all_steps(comments)
            
            if not all_steps:
                yield {
                    "stage": "error",
                    "message": "No actionable steps extracted",
                    "where": "step_extraction"
                }
                return
            
            yield {
                "stage": "steps",
                "count": len(all_steps),
                "sample": [s["step"] for s in all_steps[:5]]
            }
            
            # Stage 5: Cluster and rank
            ranked_steps = self.cluster_and_rank_steps(all_steps, hyde_result["embedding"])
            
            yield {
                "stage": "ranking",
                "clusters": ranked_steps
            }
            
            # Stage 5.5: If clustering produced 0 results and adaptive is enabled, trigger adaptive
            if adaptive_search and len(ranked_steps) == 0:
                logger.info("[PIPELINE] Clustering produced 0 ranked steps, triggering adaptive search...")
                
                # Emit a specific stage for observability
                yield {
                    "stage": "no_clusters",
                    "message": "Clustering produced no validated groups. Triggering adaptive search...",
                    "steps_extracted": len(all_steps)
                }
                
                async for event in self._run_adaptive_and_retry(
                    query=query,
                    user_story=user_story,
                    query_embedding=query_embedding,
                    hyde_embedding=hyde_result["embedding"],
                    distance_threshold=0.60,  # Use looser threshold for retry
                    max_submissions=max_submissions,
                    max_comments=max_comments
                ):
                    yield event
                
                logger.info("[PIPELINE] Pipeline complete (via adaptive path after clustering failure)")
                return
            
            elif len(ranked_steps) == 0:
                # Adaptive not enabled, emit error
                yield {
                    "stage": "no_clusters",
                    "message": "Clustering produced no validated groups. Enable Adaptive Search to fetch more data.",
                    "steps_extracted": len(all_steps)
                }
                yield {
                    "stage": "final",
                    "final_steps": [],
                    "sources": list(set([s["permalink"] for s in submissions]))
                }
                logger.info("[PIPELINE] Pipeline complete (no clusters, adaptive disabled)")
                return
            
            # Final stage
            yield {
                "stage": "final",
                "final_steps": ranked_steps,
                "sources": list(set([s["permalink"] for s in submissions]))
            }
            
            logger.info("[PIPELINE] Pipeline complete")
            
        except Exception as e:
            logger.error(f"[PIPELINE] Error: {e}")
            yield {
                "stage": "error",
                "message": str(e),
                "where": "pipeline_execution"
            }
