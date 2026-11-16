"""Claim-Check resolution resolver integrating with WRedis / Redis RAM offload."""

# pylint: disable=no-member

from typing import Any, Optional

import cv2
from loguru import logger
import numpy as np


class ClaimResolver:
    """Resolver for Claim-Check tokens storing image payloads in Redis RAM."""

    @staticmethod
    def is_claim_token(source: Any) -> bool:
        """Check if source is a claim-check token identifier.

        Args:
            source (Any): Candidate source object.

        Returns:
            bool: True if source is a string matching claim pattern, False otherwise.
        """
        if isinstance(source, str):
            clean = source.strip()
            return clean.startswith("wkafka:claim:") or clean.startswith("claim:")
        return False

    @staticmethod
    def resolve_claim(source: str, redis_client: Any) -> np.ndarray:
        """Fetch image bytes from Redis using Claim-Check token and decode to NumPy array.

        Args:
            source (str): Claim-Check token string (e.g. 'wkafka:claim:<uuid>').
            redis_client (Any): Connected Redis or WRedis instance.

        Returns:
            np.ndarray: Decoded OpenCV BGR image array.

        Raises:
            ValueError: If Redis client is missing, key is not found, or decoding fails.
        """
        if redis_client is None:
            raise ValueError(
                f"Cannot resolve Claim-Check '{source}': No redis_client provided."
            )

        key = source.strip()
        raw_bytes: Optional[bytes] = None

        if hasattr(redis_client, "get"):
            raw_bytes = redis_client.get(key)
        elif hasattr(redis_client, "read"):
            raw_bytes = redis_client.read(key)
        else:
            raise ValueError(
                f"Unsupported redis_client interface: {type(redis_client)}"
            )

        if not raw_bytes:
            raise ValueError(
                f"Claim-Check token '{key}' not found or expired in Redis."
            )

        img_buf = np.frombuffer(raw_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_buf, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to decode image from Claim-Check token '{key}'.")

        logger.debug(
            f"Successfully resolved claim '{key}' from Redis RAM "
            f"({len(raw_bytes)} bytes, shape={img.shape})"
        )
        return img
