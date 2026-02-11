"""SQL queries for the RAG pipeline."""
import json
from typing import List, Any


def get_submissions_query() -> str:
    """Get query for retrieving submissions by vector similarity."""
    return """
        SELECT 
            submission_id,
            domain_id,
            title, 
            selftext, 
            permalink, 
            (embedding <=> $1::vector) AS distance
        FROM submissions
        ORDER BY embedding <=> $1::vector
        LIMIT $2;
    """


def get_comments_query() -> str:
    """Get query for retrieving comments by vector similarity with score boost."""
    return """
        WITH ranked_comments AS (
            SELECT 
                c.body,
                c.score,
                s.title,
                c.submission_id,
                s.permalink,
                c.embedding <=> $1::vector AS distance
            FROM comments c
            JOIN submissions s ON c.submission_id = s.submission_id
            WHERE c.submission_id = ANY($2)
        )
        SELECT 
            body,
            score,
            title,
            submission_id,
            permalink,
            distance
        FROM ranked_comments
        ORDER BY 
            distance + (1.0 / (score + 10)) ASC 
        LIMIT $3;
    """


def format_embedding(embedding: List[float]) -> str:
    """Format embedding as JSON string for PostgreSQL."""
    return json.dumps(embedding)


def parse_submissions(rows: List[Any]) -> List[dict]:
    """Parse submission rows into dicts."""
    return [
        {
            "submission_id": str(row["submission_id"]),
            "domain_id": str(row["domain_id"]),
            "title": row["title"],
            "selftext": row["selftext"] or "",
            "permalink": row["permalink"],
            "distance": float(row["distance"])
        }
        for row in rows
    ]


def parse_comments(rows: List[Any]) -> List[dict]:
    """Parse comment rows into dicts."""
    return [
        {
            "body": row["body"],
            "score": row["score"],
            "title": row["title"],
            "submission_id": row["submission_id"],
            "permalink": row["permalink"],
            "distance": float(row["distance"])
        }
        for row in rows
    ]
