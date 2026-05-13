"""
backend/main.py
───────────────
FastAPI application entry point. Handles lifespan (startup/shutdown) events,
global middleware, and mounts API routers.
"""

import sys
import logging
import pathlib
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

_PROJECT_ROOT = pathlib.Path(__file__).parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.services.inference_service import inference_service
from backend.api.router import router as api_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-12s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("backend.main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler.
    Executes ONCE when the server starts up, and ONCE when it shuts down.
    Crucial for loading large deep learning/transformer models into memory efficiently.
    """
    logger.info("Initializing Copestone Comparative NLP API...")
    
    # Pre-load all machine learning models into memory globally
    # This prevents blocking on the first request and ensures high throughput
    try:
        inference_service.load_models()
        logger.info("All comparative models loaded successfully.")
    except Exception as e:
        logger.error("Critical failure during model initialization: %s", e)
        # We don't raise here, so the server can still start and report the error via /health
    
    yield  # Hand control back to FastAPI
    
    logger.info("Shutting down Copestone Comparative NLP API...")
    # Clean up resources if necessary

app = FastAPI(
    title="Comparative Sentiment Analysis API",
    description="Unified inference engine comparing Classical ML, Deep Learning, and Transformer models.",
    version="1.0.0",
    lifespan=lifespan
)

# Allow CORS for the upcoming frontend dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

_FRONTEND_DIST = _PROJECT_ROOT / "frontend" / "dist"

@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    """
    Serve the React frontend build. 
    Handles SPA routing by falling back to index.html for unknown paths.
    """
    file_path = _FRONTEND_DIST / full_path
    if file_path.is_file():
        return FileResponse(file_path)
    
    index_path = _FRONTEND_DIST / "index.html"
    if index_path.is_file():
        return FileResponse(index_path)
        
    return {"error": "Frontend build not found. Please run 'npm run build' in the frontend directory."}
