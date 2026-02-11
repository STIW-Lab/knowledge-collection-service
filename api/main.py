"""FastAPI application for RAG pipeline."""
import os
import json
from uuid import UUID
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from dotenv import load_dotenv

from lib.db import Database
from rag.pipeline import RagPipeline
from rag.types import RagRequest

load_dotenv()

# Custom JSON encoder to handle UUIDs and other non-serializable objects
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, UUID):
            return str(obj)
        return super().default(obj)

# Global instances
db: Database = None
pipeline: RagPipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - initialize and cleanup resources."""
    global db, pipeline
    
    # Initialize database
    db = Database(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        database=os.getenv("POSTGRES_DB", "knowledgebase"),
        user=os.getenv("POSTGRES_USER", "stiw_user"),
        password=os.getenv("POSTGRES_PASSWORD", "stiw_pwd")
    )
    await db.initialize()
    
    # Initialize OpenAI client
    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Initialize pipeline
    pipeline = RagPipeline(db, openai_client)
    
    print("✅ FastAPI application started")
    
    yield
    
    # Cleanup
    await db.close()
    print("✅ FastAPI application shutdown")


app = FastAPI(
    title="Knowledge Collection RAG API",
    description="RAG pipeline for actionable recommendations from Reddit knowledge base",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "rag-api"}


async def event_generator(request: RagRequest) -> AsyncGenerator[str, None]:
    """Generate SSE events from pipeline stages."""
    async for stage_data in pipeline.run_pipeline(
        user_story=request.user_story,
        current_steps=request.current_steps,
        query=request.query,
        max_submissions=request.max_submissions,
        max_comments=request.max_comments,
        distance_threshold=request.distance_threshold
    ):
        # Format as SSE event
        event_type = "stage" if stage_data.get("stage") != "error" else "error"
        data_json = json.dumps(stage_data, cls=CustomJSONEncoder)
        
        yield f"event: {event_type}\n"
        yield f"data: {data_json}\n\n"
    
    # Send done event
    yield "event: done\n"
    yield "data: {}\n\n"


@app.post("/rag/stream")
async def rag_stream(request: RagRequest):
    """
    Stream RAG pipeline stages as Server-Sent Events.
    
    Emits events:
    - stage: hyde - HyDE generation complete
    - stage: submissions - Submissions retrieved
    - stage: comments - Comments retrieved
    - stage: steps - Steps extracted
    - stage: ranking - Clusters ranked
    - stage: final - Final results
    - error: error - Error occurred
    - done: Pipeline complete
    """
    return StreamingResponse(
        event_generator(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=True
    )
