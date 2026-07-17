import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from api.routes import router as api_router

# Configure basic logging for the entire application
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for handling startup and shutdown events.
    """
    logger.info(f"Starting up {settings.app_name} in {settings.environment} mode...")
    
    if settings.vlm_enabled:
        logger.info(f"VLM reasoning is ENABLED. Configured provider: {settings.vlm_provider}")
    else:
        logger.warning("VLM reasoning is DISABLED via settings.")
        
    logger.info(f"Image storage directory set to: {settings.upload_dir}")
    
    yield  # Application is running
    
    logger.info(f"Shutting down {settings.app_name} gracefully...")


# Initialize the FastAPI application
app = FastAPI(
    title=settings.app_name,
    description="Backend API for analyzing user productivity via chronological screenshots.",
    version="1.0.0",
    debug=settings.debug,
    lifespan=lifespan
)

# Configure CORS (Cross-Origin Resource Sharing)
# For this local prototype, we allow all origins. 
# In production, this should be restricted to specific frontend domains.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

# Mount the router containing our /sessions/analyze endpoint
app.include_router(api_router)

# Serve index.html at root
@app.get("/")
async def serve_frontend():
    # Attempt to find frontend directory whether running from ai-engine or root
    frontend_dir = "frontend" if os.path.exists("frontend") else "../frontend"
    return FileResponse(os.path.join(frontend_dir, "index.html"))

# Mount static assets (css, js, etc.) if they exist
if os.path.exists("frontend"):
    app.mount("/", StaticFiles(directory="frontend"), name="frontend")
elif os.path.exists("../frontend"):
    app.mount("/", StaticFiles(directory="../frontend"), name="frontend")

@app.get("/health", tags=["System"])
async def health_check():
    """
    Lightweight endpoint to verify the server is running.
    """
    return {
        "status": "ok", 
        "app": settings.app_name, 
        "environment": settings.environment,
        "vlm_provider": settings.vlm_provider
    }