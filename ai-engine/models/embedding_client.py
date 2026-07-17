"""
Client for generating vector embeddings for images and text.
Refactored for Render: uses Hugging Face Inference API instead of
local sentence-transformers models to stay within the 512 MB RAM limit.
"""

import base64
import io
import logging
import os

import numpy as np
import requests
from PIL import Image, UnidentifiedImageError

from config.settings import settings

logger = logging.getLogger(__name__)

# --- Hugging Face Inference API Configuration ---
HF_TOKEN = os.getenv("HF_TOKEN", "")
CLIP_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/clip-ViT-B-32"
TEXT_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}


class EmbeddingClient:
    """
    Client for generating semantic embeddings via Hugging Face Inference API.
    Supports both visual (CLIP) and textual (MiniLM) embeddings without loading
    any models into local RAM.
    """

    def __init__(self) -> None:
        """
        Initializes the embedding client. No local models are loaded;
        all inference is offloaded to the Hugging Face API.
        """
        logger.info(
            "EmbeddingClient initialized (API mode). "
            "CLIP and MiniLM inference offloaded to Hugging Face Inference API."
        )

    def get_image_embedding(self, image_bytes: bytes) -> np.ndarray:
        """
        Generates a semantic vector embedding for an image using CLIP via HF API.

        Args:
            image_bytes: Raw image data (e.g., JPEG, PNG).

        Returns:
            A 1D numpy array representing the image embedding. 
            Returns an empty array if image processing fails.
        """
        if not image_bytes:
            return np.array([])

        try:
            # Convert image bytes to base64 for the HF API
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode in ("RGBA", "P"):
                image = image.convert("RGB")

            # Resize to reduce payload size for API
            max_dim = 224
            if max(image.size) > max_dim:
                ratio = max_dim / max(image.size)
                new_size = (int(image.width * ratio), int(image.height * ratio))
                image = image.resize(new_size, Image.LANCZOS)

            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=75)
            b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

            response = requests.post(
                CLIP_API_URL,
                headers=HF_HEADERS,
                json={"inputs": {"image": b64_str}},
                timeout=30,
            )

            if response.status_code != 200:
                logger.warning(f"CLIP API error ({response.status_code}): {response.text[:200]}")
                return np.array([])

            embedding_data = response.json()
            return np.array(embedding_data, dtype=np.float32).flatten()

        except UnidentifiedImageError:
            logger.error("Failed to generate image embedding: Invalid or corrupted image bytes.")
            return np.array([])
        except requests.exceptions.Timeout:
            logger.error("CLIP API request timed out.")
            return np.array([])
        except Exception as e:
            logger.error(f"Unexpected error during image embedding generation: {str(e)}")
            return np.array([])

    def get_text_embedding(self, text: str) -> np.ndarray:
        """
        Generates a semantic vector embedding for a string of text via HF API.

        Args:
            text: The string to embed (e.g., OCR output).

        Returns:
            A 1D numpy array representing the text embedding.
            Returns an empty array if the input is empty or processing fails.
        """
        if not text or not text.strip():
            return np.array([])

        try:
            response = requests.post(
                TEXT_API_URL,
                headers=HF_HEADERS,
                json={"inputs": text.strip()[:512]},  # Truncate to save API bandwidth
                timeout=30,
            )

            if response.status_code != 200:
                logger.warning(f"Text embedding API error ({response.status_code}): {response.text[:200]}")
                return np.array([])

            embedding_data = response.json()
            return np.array(embedding_data, dtype=np.float32).flatten()

        except requests.exceptions.Timeout:
            logger.error("Text embedding API request timed out.")
            return np.array([])
        except Exception as e:
            logger.error(f"Unexpected error during text embedding generation: {str(e)}")
            return np.array([])