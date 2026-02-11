"""Type definitions for the RAG pipeline."""
from typing import TypedDict, List, Optional
from pydantic import BaseModel


class RagRequest(BaseModel):
    """Request model for RAG endpoint."""
    user_story: str
    current_steps: str
    query: str
    max_submissions: int = 15
    max_comments: int = 20
    distance_threshold: float = 0.45


class HyDEResult(TypedDict):
    """HyDE generation result."""
    hyde_post: str
    domains: List[str]
    embedding: List[float]


class Submission(TypedDict):
    """Submission document."""
    submission_id: str
    domain_id: str
    title: str
    selftext: str
    permalink: str
    distance: float


class Comment(TypedDict):
    """Comment document."""
    body: str
    score: int
    title: str
    submission_id: str
    permalink: str
    distance: float


class ActionStep(TypedDict):
    """Extracted action step."""
    step: str
    embedding: List[float]
    submission_id: str
    permalink: str
    score: int


class RankedStep(TypedDict):
    """Ranked step with usefulness score."""
    url: str
    step: str
    usefulness: float
    cluster_count: int
    cluster_id: int


class StageEvent(BaseModel):
    """SSE stage event."""
    stage: str
    data: dict


class ErrorEvent(BaseModel):
    """SSE error event."""
    stage: str
    message: str
    where: str
