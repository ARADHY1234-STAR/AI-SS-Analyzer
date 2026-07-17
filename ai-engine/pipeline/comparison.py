"""
Pipeline stage for calculating frame-to-frame pairwise comparisons.
"""

import io
import logging
from dataclasses import dataclass

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ComparisonSignal:
    """
    Holds the delta metrics between a frame and its immediate predecessor.
    """
    frame_index: int
    ssim_score: float
    visual_similarity: float
    text_similarity: float
    is_trivial_change: bool
    is_idle: bool


def _compute_ssim(img_bytes1: bytes, img_bytes2: bytes) -> float:
    """
    Calculates the Structural Similarity Index Measure (SSIM) between two images.
    Returns a value between -1.0 and 1.0 (1.0 means identical).
    """
    # Convert to grayscale for structural comparison
    i1 = Image.open(io.BytesIO(img_bytes1)).convert('L')
    i2 = Image.open(io.BytesIO(img_bytes2)).convert('L')
    
    arr1 = np.array(i1)
    arr2 = np.array(i2)
    
    # Preprocessing guarantees uniform max dimensions, but edge cases (like differing aspect 
    # ratios if standard cropping failed) must be handled safely before feeding into skimage.
    if arr1.shape != arr2.shape:
        logger.debug(f"Shape mismatch in SSIM ({arr1.shape} vs {arr2.shape}). Resizing to match.")
        i2 = i2.resize(i1.size, Image.Resampling.LANCZOS)
        arr2 = np.array(i2)
        
    # Calculate SSIM (data_range is 255 for standard 8-bit grayscale images)
    score, _ = ssim(arr1, arr2, full=True, data_range=255)
    return float(score)


def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculates cosine similarity between two embedding vectors.
    Returns a value between -1.0 and 1.0.
    """
    if vec1.size == 0 or vec2.size == 0:
        return 0.0
        
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
        
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def process_batch(
    images: list[bytes], 
    image_embeddings: list[np.ndarray], 
    text_embeddings: list[np.ndarray]
) -> list[ComparisonSignal]:
    """
    Compares consecutive frames to generate productivity heuristics (e.g., detecting idle states).

    Args:
        images: Chronological list of preprocessed image bytes.
        image_embeddings: Chronological list of CLIP embeddings.
        text_embeddings: Chronological list of MiniLM OCR text embeddings.

    Returns:
        A list of ComparisonSignal objects matching the length of the batch.
        The first frame always defaults to perfect similarity since it has no predecessor.
    """
    signals: list[ComparisonSignal] = []
    if not images:
        return signals
        
    # Frame 0 establishes the baseline
    signals.append(ComparisonSignal(
        frame_index=0,
        ssim_score=1.0,
        visual_similarity=1.0,
        text_similarity=1.0,
        is_trivial_change=False,
        is_idle=False
    ))
    
    batch_size = min(len(images), len(image_embeddings), len(text_embeddings))
    
    for i in range(1, batch_size):
        try:
            # 1. Pixel-level structural comparison
            ssim_val = _compute_ssim(images[i-1], images[i])
        except Exception as e:
            logger.error(f"Failed to compute SSIM for frames {i-1} and {i}: {str(e)}")
            ssim_val = 0.0
            
        # 2. Semantic visual comparison
        vis_sim = _cosine_similarity(image_embeddings[i-1], image_embeddings[i])
        
        # 3. Semantic text comparison
        txt_sim = _cosine_similarity(text_embeddings[i-1], text_embeddings[i])
        
        # Evaluate thresholds based on central config
        is_trivial = ssim_val >= settings.ssim_trivial_change_threshold
        
        # If both the visual meaning and text meaning are nearly identical, the user is likely idle
        is_idle = (
            vis_sim >= settings.embedding_similarity_idle_threshold 
            and txt_sim >= settings.embedding_similarity_idle_threshold
        )
        
        signals.append(ComparisonSignal(
            frame_index=i,
            ssim_score=ssim_val,
            visual_similarity=vis_sim,
            text_similarity=txt_sim,
            is_trivial_change=is_trivial,
            is_idle=is_idle
        ))
        
    return signals