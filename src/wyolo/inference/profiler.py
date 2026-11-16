"""Profiling telemetry for YOLO inference pipeline latency breakdown."""

from typing import Dict

from loguru import logger


class LatencyBreakdown:
    """Latency breakdown telemetry tracker for individual or batched YOLO inferences."""

    def __init__(self, warn_threshold_ratio: float = 1.5) -> None:
        """Initialize LatencyBreakdown profiler.

        Args:
            warn_threshold_ratio (float): Ratio of CPU preprocess time vs GPU inference
                time that triggers a CPU bottleneck warning. Defaults to 1.5.
        """
        self.warn_threshold_ratio: float = warn_threshold_ratio
        self.preprocess_ms: float = 0.0
        self.pcie_transfer_ms: float = 0.0
        self.gpu_inference_ms: float = 0.0
        self.postprocess_ms: float = 0.0
        self.total_ms: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert latency measurements to dictionary.

        Returns:
            Dict[str, float]: Latency measurements formatted in milliseconds.
        """
        return {
            "preprocess_ms": round(self.preprocess_ms, 2),
            "pcie_transfer_ms": round(self.pcie_transfer_ms, 2),
            "gpu_inference_ms": round(self.gpu_inference_ms, 2),
            "postprocess_ms": round(self.postprocess_ms, 2),
            "total_ms": round(self.total_ms, 2),
        }

    def log_summary(self, batch_size: int = 1) -> None:
        """Log structured telemetry breakdown and evaluate CPU bottleneck warning.

        Args:
            batch_size (int): Number of images processed in the current batch. Defaults to 1.
        """
        logger.info(
            f"Latency Breakdown (batch={batch_size}, total={self.total_ms:.2f}ms): "
            f"Preprocess={self.preprocess_ms:.2f}ms | "
            f"PCIe={self.pcie_transfer_ms:.2f}ms | "
            f"GPU={self.gpu_inference_ms:.2f}ms | "
            f"Postprocess={self.postprocess_ms:.2f}ms"
        )

        if self.gpu_inference_ms > 0 and self.preprocess_ms > (
            self.gpu_inference_ms * self.warn_threshold_ratio
        ):
            ratio = self.preprocess_ms / self.gpu_inference_ms
            logger.warning(
                f"⚠️ CPU BOTTLENECK DETECTED: Preprocess time ({self.preprocess_ms:.2f}ms) "
                f"is {ratio:.1f}x higher than GPU inference ({self.gpu_inference_ms:.2f}ms). "
                "Consider source downscaling, GPU Zero-Copy ingestion, or batching."
            )
