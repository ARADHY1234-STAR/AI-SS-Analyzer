"""
Pipeline stage for generating semantic vector embeddings.
"""

import logging
import numpy as np

from models.embedding_client import EmbeddingClient

logger = logging.getLogger(__name__)

# Instantiate the embedding client once at the module level.
# This ensures the heavy ML models (CLIP and MiniLM) are loaded into memory 
# exactly once per worker process, avoiding massive latency on every request.
_embedding_client = EmbeddingClient()


def process_batch(images: list[bytes], ocr_texts: list[str]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Generates semantic embeddings for both the visual frames and their extracted text.

    Args:
        images: A chronological list of preprocessed image bytes.
        ocr_texts: A chronological list of extracted OCR text strings corresponding to the images.

    Returns:
        A tuple containing two lists:
        1. list[np.ndarray]: Visual embeddings for the images.
        2. list[np.ndarray]: Semantic embeddings for the OCR texts.
        Both lists are guaranteed to match the length of the shortest input list.
    """
    if not images:
        return [], []

    image_embeddings: list[np.ndarray] = []
    text_embeddings: list[np.ndarray] = []

    # Safeguard against upstream misalignment
    batch_size = min(len(images), len(ocr_texts))
    if len(images) != len(ocr_texts):
        logger.warning(
            f"Mismatch in batch sizes: {len(images)} images vs {len(ocr_texts)} texts. "
            f"Processing up to index {batch_size}."
        )

    for idx in range(batch_size):
        img_bytes = images[idx]
        text = ocr_texts[idx]
        
        logger.debug(f"Generating embeddings for frame {idx + 1}/{batch_size}...")
        
        # 1. Generate image embedding
        try:
            img_emb = _embedding_client.get_image_embedding(img_bytes)
            image_embeddings.append(img_emb)
        except Exception as e:
            logger.error(f"Failed to generate image embedding for frame {idx}: {str(e)}")
            # Append empty array to maintain index alignment for downstream comparison
            image_embeddings.append(np.array([]))
            
        # 2. Generate text embedding
        try:
            txt_emb = _embedding_client.get_text_embedding(text)
            text_embeddings.append(txt_emb)
        except Exception as e:
            logger.error(f"Failed to generate text embedding for frame {idx}: {str(e)}")
            text_embeddings.append(np.array([]))

    return image_embeddings, text_embeddings