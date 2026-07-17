"""
FastAPI router and endpoints for the ai-engine backend.
"""

import logging

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status

from config.settings import settings
from api.schemas import AnalyzeSessionResponse
from pipeline import orchestrator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Sessions"])


@router.post(
    "/sessions/analyze", 
    response_model=AnalyzeSessionResponse, 
    status_code=status.HTTP_200_OK
)
async def analyze_session_endpoint(
    screenshots: list[UploadFile] = File(
        ..., 
        description="Chronological list of image files (screenshots) to analyze."
    ),
    model_override: str | None = Form(
        default=None,
        description="Optional OpenRouter model slug to override the default VLM model for this request."
    ),
    persona_id: str | None = Form(
        default="backend_dev",
        description="Optional persona ID to select dynamic activity benchmarking rules."
    ),
    include_keystrokes: bool = Form(
        default=False,
        description="Whether to include keyboard and mouse strokes in the analysis."
    ),
):
    """
    Analyzes a chronological batch of screenshots to determine productivity and task progress.
    
    This endpoint operates synchronously and processes the entire pipeline (Preprocessing, OCR,
    Embeddings, Comparison, Domain Detectors, and VLM Reasoning) in a single request.
    Accepts an optional `model_override` form field to dynamically select a different AI model.
    """
    if not screenshots:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No screenshots provided. Please upload at least one image."
        )

    if len(screenshots) > settings.max_screenshots_per_batch:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Too many screenshots. Maximum allowed per batch is {settings.max_screenshots_per_batch}."
        )

    if model_override:
        logger.info(f"Model override requested from frontend: {model_override}")

    # Read all multipart files into memory as raw bytes
    raw_images: list[bytes] = []
    for file in screenshots:
        try:
            # Basic validation to ensure we aren't trying to process PDFs or text files
            if not file.content_type or not file.content_type.startswith("image/"):
                raise HTTPException(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail=f"File '{file.filename}' is not a valid image. Expected MIME type 'image/*'."
                )
            
            content = await file.read()
            raw_images.append(content)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to read file {file.filename}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to process file upload: {str(e)}"
            )
        finally:
            await file.close()

    try:
        # Pass the extracted byte arrays to the synchronous orchestrator pipeline.
        # FastAPI will automatically run this in a worker thread if it blocks, 
        # but for our local prototype, this directly satisfies the synchronous request requirement.
        result = orchestrator.analyze_session(
            raw_images, 
            model_override=model_override,
            persona_id=persona_id,
            include_keystrokes=include_keystrokes
        )
        
        # The internal AggregatedResult dataclass maps perfectly to our AnalyzeSessionResponse Pydantic schema.
        # FastAPI handles the final JSON serialization automatically.
        return result
        
    except ValueError as e:
        # ValueErrors from the orchestrator usually indicate unprocessable data (e.g., all images were corrupted)
        logger.warning(f"Validation error during pipeline execution: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Fatal error during session analysis: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during pipeline analysis."
        )