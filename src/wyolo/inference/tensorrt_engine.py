"""TensorRT execution engine for accelerated YOLO inference using PyCUDA."""

import os
from typing import Any, Dict, List, Optional

from loguru import logger
import numpy as np

try:
    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
    import tensorrt as trt

    TRT_AVAILABLE = True
except ImportError:
    trt = None
    cuda = None
    TRT_AVAILABLE = False


class TensorRTYOLO:
    """High-performance TensorRT runtime for YOLO with pinned memory and async DMA transfers."""

    def __init__(
        self,
        engine_path: str,
        max_batch_size: int = 16,
        imgsz: int = 640,
    ) -> None:
        """Initialize TensorRTYOLO engine.

        Args:
            engine_path (str): Path to serialized TensorRT .engine file.
            max_batch_size (int): Maximum batch size supported by pre-allocated buffers. Defaults to 16.
            imgsz (int): Square image resolution expected by model input. Defaults to 640.

        Raises:
            RuntimeError: If TensorRT or PyCUDA dependencies are not installed.
            FileNotFoundError: If engine file does not exist.
        """
        if not TRT_AVAILABLE:
            raise RuntimeError(
                "TensorRT and PyCUDA are required to run TensorRTYOLO. "
                "Ensure 'tensorrt' and 'pycuda' are installed in the environment."
            )

        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"Engine file not found: {engine_path}")

        self.engine_path: str = engine_path
        self.max_batch_size: int = max_batch_size
        self.imgsz: int = imgsz

        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as engine_file:
            with trt.Runtime(self.logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(engine_file.read())

        if self.engine is None:
            raise RuntimeError(
                f"Failed to deserialize TensorRT engine from {engine_path}"
            )

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.input_name: str = self.engine.get_tensor_name(0)
        self.output_name: str = self.engine.get_tensor_name(1)

        input_shape = (max_batch_size, 3, imgsz, imgsz)
        output_shape = (max_batch_size, 300, 6)

        self.d_input = cuda.mem_alloc(int(np.prod(input_shape) * np.float32().nbytes))
        self.d_output = cuda.mem_alloc(int(np.prod(output_shape) * np.float32().nbytes))
        self.h_output = cuda.pagelocked_empty(output_shape, dtype=np.float32)

        logger.info(
            f"TensorRTYOLO initialized successfully from '{engine_path}' "
            f"(max_batch_size={max_batch_size}, imgsz={imgsz})."
        )

    def infer_batch(self, batch_images: np.ndarray) -> np.ndarray:
        """Execute TensorRT inference on a batch of preprocessed images.

        Args:
            batch_images (np.ndarray): Contiguous float32 array with shape (B, 3, imgsz, imgsz).

        Returns:
            np.ndarray: Sliced output array of shape (B, 300, 6).

        Raises:
            ValueError: If input batch exceeds max_batch_size or dimensions do not match.
        """
        batch_size = int(batch_images.shape[0])
        if batch_size > self.max_batch_size:
            raise ValueError(
                f"Batch size {batch_size} exceeds configured max_batch_size {self.max_batch_size}"
            )

        self.context.set_input_shape(
            self.input_name, (batch_size, 3, self.imgsz, self.imgsz)
        )
        self.context.set_tensor_address(self.input_name, int(self.d_input))
        self.context.set_tensor_address(self.output_name, int(self.d_output))

        contiguous_images = np.ascontiguousarray(batch_images, dtype=np.float32)
        cuda.memcpy_htod_async(self.d_input, contiguous_images, self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        out_slice = self.h_output[:batch_size]
        cuda.memcpy_dtoh_async(out_slice, self.d_output, self.stream)
        self.stream.synchronize()

        return out_slice

    def postprocess(
        self,
        output: np.ndarray,
        conf_threshold: float = 0.25,
        class_names: Optional[Dict[int, str]] = None,
    ) -> List[List[Dict[str, Any]]]:
        """Postprocess TensorRT raw output into canonical detection dictionaries.

        Args:
            output (np.ndarray): TensorRT output array with shape (B, 300, 6).
            conf_threshold (float): Minimum confidence threshold. Defaults to 0.25.
            class_names (Optional[Dict[int, str]]): Mapping of class IDs to class names. Defaults to None.

        Returns:
            List[List[Dict[str, Any]]]: List of detections per image in the batch.
        """
        batch_results: List[List[Dict[str, Any]]] = []
        batch_size = output.shape[0]

        for i in range(batch_size):
            image_preds = output[i]
            valid_mask = image_preds[:, 4] >= conf_threshold
            valid_preds = image_preds[valid_mask]

            detections: List[Dict[str, Any]] = []
            for row in valid_preds:
                x1, y1, x2, y2 = [int(round(coord)) for coord in row[:4]]
                conf = float(row[4])
                class_id = int(row[5])
                class_name = (
                    class_names.get(class_id, str(class_id))
                    if class_names
                    else str(class_id)
                )

                detections.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "conf": round(conf, 4),
                        "conf_percentage": round(conf * 100.0, 2),
                        "class_id": class_id,
                        "name": class_name,
                        "status": True,
                    }
                )
            batch_results.append(detections)

        return batch_results
