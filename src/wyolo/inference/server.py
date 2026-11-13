"""LitServe API server for high-throughput batched YOLO TensorRT inference."""

from typing import Any, Dict, List, Optional

import cv2
from loguru import logger
import numpy as np

try:
    import litserve as ls

    LITSERVE_AVAILABLE = True
except ImportError:
    ls = None
    LITSERVE_AVAILABLE = False

from .tensorrt_engine import TensorRTYOLO


if LITSERVE_AVAILABLE:

    class YOLOTensorRTAPI(ls.LitAPI):
        """LitServe API handler for TensorRT YOLO with adaptive dynamic batching."""

        def setup(self, device: str) -> None:
            """Initialize the TensorRT model on the designated GPU device.

            Args:
                device (str): Device descriptor string from LitServer (e.g. 'cuda:0').
            """
            self.imgsz = getattr(self, "imgsz", 640)
            self.conf_threshold = getattr(self, "conf_threshold", 0.45)
            self.max_batch_size = getattr(self, "max_batch_size", 16)
            engine_path = getattr(self, "engine_path", "model.engine")

            logger.info(
                f"Setting up YOLOTensorRTAPI on device '{device}' with engine: {engine_path}"
            )
            self.trt_model = TensorRTYOLO(
                engine_path=engine_path,
                max_batch_size=self.max_batch_size,
                imgsz=self.imgsz,
            )

        def decode_request(self, request: Dict[str, Any]) -> np.ndarray:
            """Decode image payload from incoming HTTP request.

            Args:
                request (Dict[str, Any]): Request payload containing raw image bytes.

            Returns:
                np.ndarray: Preprocessed float32 image array (3, imgsz, imgsz).
            """
            raw_data = request.get("image")
            if isinstance(raw_data, (bytes, bytearray)):
                img_bytes = np.frombuffer(raw_data, dtype=np.uint8)
            else:
                raise ValueError("Expected 'image' field with raw binary image bytes.")

            img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to decode image from bytes.")

            resized = cv2.resize(
                img, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR
            )
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            chw = rgb.transpose((2, 0, 1)).astype(np.float32) / 255.0
            return chw

        def batch(self, inputs: List[np.ndarray]) -> np.ndarray:
            """Stack individual preprocessed images into batched NumPy array.

            Args:
                inputs (List[np.ndarray]): List of image arrays of shape (3, imgsz, imgsz).

            Returns:
                np.ndarray: Stacked batch tensor of shape (B, 3, imgsz, imgsz).
            """
            return np.stack(inputs, axis=0)

        def predict(self, x: np.ndarray) -> np.ndarray:
            """Execute batched TensorRT engine inference.

            Args:
                x (np.ndarray): Batched input array.

            Returns:
                np.ndarray: Raw model output array of shape (B, 300, 6).
            """
            return self.trt_model.infer_batch(x)

        def unbatch(self, output: np.ndarray) -> List[np.ndarray]:
            """Unstack batched model output into per-request slices.

            Args:
                output (np.ndarray): Batched output array of shape (B, 300, 6).

            Returns:
                List[np.ndarray]: List of 2D arrays (300, 6).
            """
            return [output[i] for i in range(output.shape[0])]

        def encode_response(self, output: np.ndarray) -> Dict[str, Any]:
            """Format single-image prediction array into canonical JSON response.

            Args:
                output (np.ndarray): Detections array with shape (300, 6).

            Returns:
                Dict[str, Any]: Response dictionary with filtered detections.
            """
            valid = output[output[:, 4] >= self.conf_threshold]
            detections = [
                {
                    "bbox": [int(round(c)) for c in row[:4]],
                    "conf": round(float(row[4]), 4),
                    "conf_percentage": round(float(row[4]) * 100.0, 2),
                    "class": int(row[5]),
                    "status": True,
                }
                for row in valid
            ]
            return {"detections": detections}

else:

    class YOLOTensorRTAPI:  # type: ignore
        """Fallback placeholder when LitServe is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """Initialize fallback placeholder raising error.

            Args:
                *args: Variable length argument list.
                **kwargs: Arbitrary keyword arguments.
            """
            raise RuntimeError(
                "LitServe is not installed. Install with 'pip install litserve' to use YOLOTensorRTAPI."
            )


def serve_yolo(
    engine_path: str,
    port: int = 8000,
    max_batch_size: int = 16,
    batch_timeout: float = 0.025,
    imgsz: int = 640,
    conf_threshold: float = 0.45,
    workers_per_device: int = 1,
    accelerator: str = "gpu",
) -> None:
    """Launch a high-performance LitServe API server for TensorRT YOLO models.

    Args:
        engine_path (str): Path to serialized TensorRT .engine model file.
        port (int): HTTP port to listen on. Defaults to 8000.
        max_batch_size (int): Maximum batch size for auto-batching. Defaults to 16.
        batch_timeout (float): Max wait window in seconds to aggregate batch. Defaults to 0.025.
        imgsz (int): Square image resolution expected by model. Defaults to 640.
        conf_threshold (float): Confidence filtering threshold. Defaults to 0.45.
        workers_per_device (int): Number of worker threads per GPU device. Defaults to 1.
        accelerator (str): Accelerator type ('gpu' or 'auto'). Defaults to 'gpu'.

    Raises:
        RuntimeError: If LitServe is not installed.
    """
    if not LITSERVE_AVAILABLE:
        raise RuntimeError(
            "LitServe is required to serve YOLO models. Please install with 'pip install litserve'."
        )

    api = YOLOTensorRTAPI()
    api.engine_path = engine_path  # type: ignore
    api.imgsz = imgsz  # type: ignore
    api.conf_threshold = conf_threshold  # type: ignore
    api.max_batch_size = max_batch_size  # type: ignore

    logger.info(
        f"Starting LitServe TensorRT YOLO service on port {port} "
        f"(max_batch_size={max_batch_size}, timeout={batch_timeout}s)..."
    )

    server = ls.LitServer(
        api,
        accelerator=accelerator,
        max_batch_size=max_batch_size,
        batch_timeout=batch_timeout,
        workers_per_device=workers_per_device,
    )
    server.run(port=port)
