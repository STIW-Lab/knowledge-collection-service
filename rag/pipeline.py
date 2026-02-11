"""RAG pipeline implementation based on test.ipynb."""
import json
import numpy as np
from typing import List, Dict, Any, AsyncGenerator, Optional
from openai import AsyncOpenAI
from scipy.spatial.distance import cosine
from hdbscan import HDBSCAN

from lib.db import Database
from rag.sql import (
    get_submissions_query,
    get_comments_query,
    format_embedding,
    parse_submissions,
    parse_comments
)
from rag.types import HyDEResult, ActionStep, RankedStep


class RagPipeline:
    """RAG pipeline for generating actionable recommendations."""

    def __init__(self, db: Database, openai_client: AsyncOpenAI):
        """Initialize pipeline with database and OpenAI client."""
        self.db = db
        self.client = openai_client

    async def generate_hyde(
        self, user_story: str, current_steps: str, query: str
    ) -> HyDEResult:
        """Generate HyDE hypothetical Reddit post with persona context."""
        
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

        # Generate embedding for HyDE post
        emb_response = await self.client.embeddings.create(
            model="text-embedding-3-small",
            input=hyde_post
        )
        embedding = emb_response.data[0].embedding

        # For now, return empty domains list (classifier not implemented in notebook)
        return {
            "hyde_post": hyde_post,
            "domains": [],
            "embedding": embedding
        }

    async def retrieve_submissions(
        self, embedding: List[float], max_submissions: int = 15, distance_threshold: float = 0.50
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant submissions by vector similarity."""
        query = get_submissions_query()
        emb_str = format_embedding(embedding)
        
        rows = await self.db.fetch(query, emb_str, max_submissions)
        submissions = parse_submissions(rows)

        print(json.dumps(submissions, indent=5))
        
        # Filter by distance threshold
        filtered = [s for s in submissions if s["distance"] < distance_threshold]
        return filtered

    async def retrieve_comments(
        self, embedding: List[float], submission_ids: List[str], max_comments: int = 20
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant comments from submissions with score boost."""
        if not submission_ids:
            return []
            
        query = get_comments_query()
        emb_str = format_embedding(embedding)
        
        rows = await self.db.fetch(query, emb_str, submission_ids, max_comments)
        comments = parse_comments(rows)
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

        return all_steps

    def cluster_and_rank_steps(
        self, all_steps: List[ActionStep], query_embedding: List[float]
    ) -> List[RankedStep]:
        """Cluster steps using HDBSCAN and rank by usefulness."""
        if not all_steps:
            return []

        # Convert to numpy array
        X = np.array([s["embedding"] for s in all_steps])
        
        # Cluster using HDBSCAN
        clusterer = HDBSCAN(
            metric="euclidean",
            min_cluster_size=4,
            min_samples=2,
            cluster_selection_epsilon=0.05
        )
        
        labels = clusterer.fit_predict(X)
        
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
        
        # Filter clusters by value (crowd validation)
        filtered_clusters = []
        for cid, items in clusters.items():
            src_scores = [s["score"] for s in items]
            if not src_scores:
                continue
            cluster_value = np.mean(src_scores) / np.max(src_scores)
            
            if cluster_value > 0.20:
                filtered_clusters.append({
                    "cluster_id": cid,
                    "items": items,
                    "cluster_value": cluster_value,
                    "frequency": len(items)
                })
        
        # Select best representative from each cluster
        action_steps = []
        q = np.array(query_embedding)
        
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
        
        return action_steps

    async def run_pipeline(
        self,
        user_story: str,
        current_steps: str,
        query: str,
        max_submissions: int = 15,
        max_comments: int = 20,
        distance_threshold: float = 0.45
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Run the full RAG pipeline and yield stage events."""
        
        try:
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
            
            if not submissions:
                yield {
                    "stage": "error",
                    "message": "No relevant submissions found",
                    "where": "submissions_retrieval"
                }
                return
            
            yield {
                "stage": "submissions",
                "items": submissions,
                "count": len(submissions)
            }
            
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
            
            # Final stage
            yield {
                "stage": "final",
                "final_steps": ranked_steps,
                "sources": list(set([s["permalink"] for s in submissions]))
            }
            
        except Exception as e:
            yield {
                "stage": "error",
                "message": str(e),
                "where": "pipeline_execution"
            }
