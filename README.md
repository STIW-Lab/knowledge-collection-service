# knowledge-collection-service

Collects Data from Reddit, Twitter, Quora while establishing ground truth and doing so legally.

## RAG API

### Prerequisites

1. PostgreSQL with pgvector extension running
2. Database populated with submissions and comments (see `kb_init.ipynb`)
3. Python 3.12+
4. uv package manager

### Environment Variables

Create a `.env` file with:

```env
# OpenAI
OPENAI_API_KEY=your_openai_api_key

# PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=knowledgebase
POSTGRES_USER=stiw_user
POSTGRES_PASSWORD=stiw_pwd

# API (optional)
PORT=8000
```

### Running the API

```bash
# Install dependencies
uv sync

# Run the FastAPI server
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### API Endpoints

#### `GET /healthz`
Health check endpoint.

#### `POST /rag/stream`
Stream RAG pipeline stages as Server-Sent Events.

**Request:**
```json
{
  "user_story": "I'm a 35-year-old software developer...",
  "current_steps": "I've tried going to the gym twice...",
  "query": "How do I stay consistent with fitness?",
  "max_submissions": 15,
  "max_comments": 20,
  "distance_threshold": 0.45
}
```

**Response:** SSE stream with events:
- `stage: hyde` - HyDE post generated
- `stage: submissions` - Submissions retrieved
- `stage: comments` - Comments retrieved
- `stage: steps` - Steps extracted
- `stage: ranking` - Clusters ranked
- `stage: final` - Final recommendations
- `error: error` - Error occurred
