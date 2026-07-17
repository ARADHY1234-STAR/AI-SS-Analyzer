"""
Client for generating vector embeddings for images and text.
"""

import io
import logging
import os
import requests

import numpy as np
from PIL import Image, UnidentifiedImageError
from sentence_transformers import SentenceTransformer

from config.settings import settings

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """
    Client for generating semantic embeddings.
    Supports both visual (CLIP) and textual (MiniLM) embeddings.
    """

    def __init__(self) -> None:
        """
        Initializes the embedding client and eagerly loads the pre-trained models.
        Model loading is heavy, so this class is intended to be instantiated once 
        and shared across the pipeline.
        """
        os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"
        os.environ["HF_HUB_ETAG_TIMEOUT"] = "60"

        # Load CLIP Model (online first, then offline cache)
        def _load_clip_model(model_name: str) -> SentenceTransformer:
            try:
                return SentenceTransformer(model_name)
            except Exception:
                logger.warning("Network error loading CLIP online. Switching to offline cache...")
                base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                clip_local_path = os.path.join(base_dir, "my_models", "clip-ViT-B-32")
                os.environ["HF_HUB_OFFLINE"] = "1"
                return SentenceTransformer(clip_local_path, local_files_only=True)

        self.clip_model = _load_clip_model(settings.clip_model_name)

        # Load Text Model (default fallback behavior)
        def _load_text_model(model_name: str) -> SentenceTransformer:
            try:
                return SentenceTransformer(model_name)
            except (requests.exceptions.RequestException, ConnectionError, TimeoutError):
                logger.warning("Hugging Face timeout detected for text model, switching to offline cache...")
                os.environ["HF_HUB_OFFLINE"] = "1"
                return SentenceTransformer(model_name, local_files_only=True)

        self.text_model = _load_text_model(settings.text_embedding_model_name)

    def get_image_embedding(self, image_bytes: bytes) -> np.ndarray:
        """
        Generates a semantic vector embedding for an image using CLIP.

        Args:
            image_bytes: Raw image data (e.g., JPEG, PNG).

        Returns:
            A 1D numpy array representing the image embedding. 
            Returns an empty array if image processing fails.
        """
        if not image_bytes:
            return np.array([])

        try:
            image = Image.open(io.BytesIO(image_bytes))
            # sentence-transformers natively supports PIL Image inputs for CLIP models
            embedding: np.ndarray = self.clip_model.encode(image)
            return embedding
        except UnidentifiedImageError:
            logger.error("Failed to generate image embedding: Invalid or corrupted image bytes.")
            return np.array([])
        except Exception as e:
            logger.error(f"Unexpected error during image embedding generation: {str(e)}")
            return np.array([])

    def get_text_embedding(self, text: str) -> np.ndarray:
        """
        Generates a semantic vector embedding for a string of text.

        Args:
            text: The string to embed (e.g., OCR output).

        Returns:
            A 1D numpy array representing the text embedding.
            Returns an empty array if the input is empty or processing fails.
        """
        if not text or not text.strip():
            return np.array([])

        try:
            embedding: np.ndarray = self.text_model.encode(text.strip())
            return embedding
        except Exception as e:
            logger.error(f"Unexpected error during text embedding generation: {str(e)}")
            return np.array([])