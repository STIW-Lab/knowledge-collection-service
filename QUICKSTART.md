# Quick Start Guide

## Prerequisites

✅ PostgreSQL with pgvector extension running  
✅ Knowledge base populated (see `knowledge-collection-service/kb_init.ipynb`)  
✅ Python 3.12+ with uv installed  
✅ Node.js 18+ with npm/yarn/pnpm  

## Step 1: Backend Setup

```bash
cd knowledge-collection-service

# Create .env file
cat > .env << EOF
OPENAI_API_KEY=your_openai_key_here
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=knowledgebase
POSTGRES_USER=stiw_user
POSTGRES_PASSWORD=stiw_pwd
EOF

# Install dependencies
uv sync

# Start the API server
chmod +x run_api.sh
./run_api.sh

# Or on Windows:
# run_api.bat

# Or manually:
# uv run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at http://localhost:8000

## Step 2: Frontend Setup

```bash
# In a new terminal
cd system-playground

# Install dependencies
npm install
# or: yarn install
# or: pnpm install

# Create .env.local
echo "FASTAPI_BASE_URL=http://localhost:8000" > .env.local

# Start development server
npm run dev
# or: yarn dev
# or: pnpm dev
```

The UI will be available at http://localhost:3000

## Step 3: Test the System

1. Open http://localhost:3000 in your browser
2. Select a persona (e.g., "Alex Chen - Software developer trying to get fit")
3. Review the persona's background and current steps
4. The query field will be pre-filled with a default question
5. Click "Recommend Next Steps"
6. Watch the RAG pipeline process in real-time:
   - HyDE generation
   - Submissions retrieval
   - Comments retrieval
   - Step extraction
   - Clustering & ranking
7. View the final recommendations with usefulness scores and Reddit sources

## Troubleshooting

### Backend won't start
- Check that PostgreSQL is running: `pg_isready`
- Verify database credentials in `.env`
- Ensure OpenAI API key is valid
- Check Python version: `python --version` (need 3.12+)

### Frontend won't start
- Check Node.js version: `node --version` (need 18+)
- Delete `node_modules` and reinstall: `rm -rf node_modules && npm install`
- Verify `FASTAPI_BASE_URL` in `.env.local`

### No recommendations returned
- Verify knowledge base has data: Check `submissions` and `comments` tables
- Check FastAPI logs for errors
- Try a different persona or query
- Ensure distance threshold (0.45) isn't too restrictive

### CORS errors
- Verify `FASTAPI_BASE_URL` matches the actual backend URL
- Check that FastAPI CORS middleware is configured correctly

## API Endpoints

### Backend (FastAPI)
- `GET http://localhost:8000/healthz` - Health check
- `POST http://localhost:8000/rag/stream` - RAG streaming endpoint

### Frontend (Next.js)
- `GET http://localhost:3000` - Main UI
- `POST http://localhost:3000/api/rag` - Proxy to FastAPI

## What Each Persona Tests

- **Alex Chen**: Physical health & fitness queries
- **Maria Rodriguez**: Mental health & anxiety management
- **James Wilson**: Career change & skill development
- **Sarah Patel**: Work-life balance & burnout
- **David Kim**: Purpose & direction finding

## Next Steps

- Modify personas in `system-playground/src/personas.ts`
- Adjust pipeline parameters in `knowledge-collection-service/rag/pipeline.py`
- Customize UI in `system-playground/src/app/page.tsx`
- Deploy to Vercel (frontend) and your preferred host (backend)

## Need Help?

See detailed documentation:
- Backend: `knowledge-collection-service/README.md`
- Frontend: `system-playground/README.md`
- Implementation: `IMPLEMENTATION_SUMMARY.md`
