"""Unified YOLO inference module supporting PyTorch and TensorRT engine backends.

Provides transparent single-call inference across image types, Claim-Check Redis tokens,
Zero-Copy GPU tensors, and dynamic micro-batching with latency profiling.
"""

# pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-positional-arguments
# pylint: disable=too-many-locals,too-many-branches,no-member,broad-exception-caught

import os
import time
from typing import Any, Dict, List, Optional, Union

import cv2
from loguru import logger
import numpy as np
from PIL import Image
from ultralytics import YOLO

from .async_engine import AsyncInferenceFuture, AsyncWorkerPool
from .claim_resolver import ClaimResolver
from .dynamic_batcher import DynamicBatcher
from .engine_exporter import ensure_tensorrt_engine
from .profiler import LatencyBreakdown
from .tensorrt_engine import TRT_AVAILABLE, TensorRTYOLO
from .zero_copy import is_cupy_array, is_torch_cuda_tensor


class YOLOInference:
    """Unified high-performance YOLO predictor supporting .pt and .engine model formats."""

    def __init__(
        self,
        model_path: str,
        conf: float = 0.25,
        imgsz: int = 640,
        quantization: Optional[int] = 16,
        max_batch_size: Optional[int] = None,
        class_names: Optional[Dict[int, str]] = None,
        auto_engine: bool = True,
        redis_client: Optional[Any] = None,
        profiling: bool = False,
        enable_microbatching: bool = False,
        microbatch_delay_ms: float = 3.0,
    ) -> None:
        """Initialize YOLOInference predictor.

        Args:
            model_path (str): Path to .pt or .engine model file.
            conf (float): Default confidence threshold for detections. Defaults to 0.25.
            imgsz (int): Square image resolution expected by model input. Defaults to 640.
            quantization (Optional[int]): Integer quantization mode (8, 16, or 32).
                Automatically sets max_batch_size = quantization if omitted.
                Defaults to 16 (FP16 & batch=16).
            max_batch_size (Optional[int]): Max batch size for TensorRT. Defaults to None.
            class_names (Optional[Dict[int, str]]): Class ID to name dictionary.
            auto_engine (bool): Auto resolve or compile TensorRT engine. Defaults to True.
            redis_client (Optional[Any]): Redis client instance for Claim-Check tokens.
            profiling (bool): Enable latency profiling and bottleneck alerts.
            enable_microbatching (bool): Enable temporal micro-batching. Defaults to False.
            microbatch_delay_ms (float): Temporal window in ms. Defaults to 3.0.

        Raises:
            FileNotFoundError: If model file does not exist.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Quantization rule: batch defaults to quantization value (8 -> 8, 16 -> 16, 32 -> 32)
        effective_batch: int = max_batch_size or (
            quantization if quantization in (8, 16, 32) else 16
        )
        int8_mode: bool = quantization == 8
        half_mode: bool = (quantization == 16) or (quantization is None)

        if auto_engine and model_path.endswith(".pt"):
            model_path = ensure_tensorrt_engine(
                model_path=model_path,
                imgsz=imgsz,
                half=half_mode,
                int8=int8_mode,
                quantization=quantization,
                batch=effective_batch,
            )

        self.model_path: str = model_path
        self.conf: float = conf
        self.imgsz: int = imgsz
        self.quantization: Optional[int] = quantization
        self.max_batch_size: int = effective_batch
        self.class_names: Dict[int, str] = class_names or {}
        self.redis_client: Optional[Any] = redis_client
        self.profiling: bool = profiling

        self.is_engine: bool = model_path.endswith(".engine")
        self.trt_model: Optional[TensorRTYOLO] = None
        self.pt_model: Optional[YOLO] = None

        if self.is_engine and TRT_AVAILABLE:
            try:
                self.trt_model = TensorRTYOLO(
                    engine_path=model_path,
                    max_batch_size=effective_batch,
                    imgsz=imgsz,
                )
                logger.info(
                    f"Loaded TensorRT engine via PyCUDA: {model_path} (batch={effective_batch})"
                )
            except Exception as e:
                logger.warning(
                    f"Direct TensorRTYOLO load failed ({e}), falling back to Ultralytics engine."
                )
                self.pt_model = YOLO(model_path)
        else:
            self.pt_model = YOLO(model_path)
            logger.info(f"Loaded model via Ultralytics: {model_path}")

        if (
            not self.class_names
            and self.pt_model is not None
            and hasattr(self.pt_model, "names")
        ):
            self.class_names = {int(k): str(v) for k, v in self.pt_model.names.items()}

        self._async_pool: Optional[AsyncWorkerPool] = None
        self._batcher: Optional[DynamicBatcher] = None
        if enable_microbatching:
            self._batcher = DynamicBatcher(
                infer_fn=self.predict_batch,
                max_batch_size=self.max_batch_size,
                max_delay_ms=microbatch_delay_ms,
            )

    def _resolve_source_to_image(
        self, source: Any
    ) -> Union[np.ndarray, Image.Image, Any]:
        """Resolve claim tokens, bytes, or file paths into decodable image or tensor.

        Args:
            source (Any): Input source (claim token string, bytes, path, ndarray, or tensor).

        Returns:
            Union[np.ndarray, Image.Image, Any]: Resolved image or GPU tensor.
        """
        if is_torch_cuda_tensor(source) or is_cupy_array(source):
            return source

        if ClaimResolver.is_claim_token(source):
            return ClaimResolver.resolve_claim(
                source=str(source), redis_client=self.redis_client
            )

        return source

    def _preprocess_image(
        self, source: Union[str, bytes, np.ndarray, Image.Image, Any]
    ) -> np.ndarray:
        """Convert input image source into normalized float32 CHW array (3, imgsz, imgsz).

        Args:
            source (Union[str, bytes, np.ndarray, Image.Image, Any]): Image input formats.

        Returns:
            np.ndarray: Preprocessed float32 array with shape (3, imgsz, imgsz).

        Raises:
            ValueError: If source format is unsupported or image decoding fails.
        """
        source = self._resolve_source_to_image(source)

        if is_torch_cuda_tensor(source):
            cpu_tensor = source.detach().cpu().float()
            if cpu_tensor.ndim == 3 and cpu_tensor.shape[0] == 3:
                return cpu_tensor.numpy()
            return cpu_tensor.permute(2, 0, 1).numpy()

        if isinstance(source, bytes):
            img_buf = np.frombuffer(source, dtype=np.uint8)
            img = cv2.imdecode(img_buf, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to decode image from raw bytes.")
        elif isinstance(source, str):
            if not os.path.exists(source):
                raise FileNotFoundError(f"Image file does not exist: {source}")
            img = cv2.imread(source)
            if img is None:
                raise ValueError(f"Failed to load image from path: {source}")
        elif isinstance(source, Image.Image):
            img = cv2.cvtColor(np.array(source), cv2.COLOR_RGB2BGR)
        elif isinstance(source, np.ndarray):
            img = source.copy()
        else:
            raise ValueError(f"Unsupported image input type: {type(source)}")

        resized = cv2.resize(
            img, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        chw = rgb.transpose((2, 0, 1)).astype(np.float32) / 255.0
        return chw

    def predict(
        self,
        source: Union[str, bytes, np.ndarray, Image.Image, Any],
        conf: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Run unified object detection inference on a single image, claim, or tensor.

        Supports NumPy arrays, raw bytes, claim-check strings ('wkafka:claim:...'),
        PyTorch CUDA tensors, and PIL Images.

        Args:
            source (Union[str, bytes, np.ndarray, Image.Image, Any]): Image source.
            conf (Optional[float]): Optional confidence threshold override. Defaults to None.

        Returns:
            List[Dict[str, Any]]: Canonical list of detected objects.
        """
        if self._batcher is not None:
            return self._batcher.submit(source)

        batch_results = self.predict_batch([source], conf=conf)
        return batch_results[0] if batch_results else []

    def predict_async(
        self,
        source: Union[str, bytes, np.ndarray, Image.Image, Any],
        conf: Optional[float] = None,
    ) -> AsyncInferenceFuture:
        """Run asynchronous non-blocking object detection inference.

        Args:
            source (Union[str, bytes, np.ndarray, Image.Image, Any]): Image source.
            conf (Optional[float]): Optional confidence threshold override. Defaults to None.

        Returns:
            AsyncInferenceFuture: Future object that can be queried or awaited via .result().
        """
        if self._async_pool is None:
            self._async_pool = AsyncWorkerPool(max_workers=2)
        return self._async_pool.submit(self.predict, source=source, conf=conf)

    def predict_bytes(
        self,
        image_bytes: bytes,
        conf: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Convenience method to run inference directly from raw image bytes.

        Args:
            image_bytes (bytes): Raw JPEG/PNG image bytes (e.g. from Kafka message).
            conf (Optional[float]): Optional confidence threshold override. Defaults to None.

        Returns:
            List[Dict[str, Any]]: Canonical list of detected objects.
        """
        return self.predict(source=image_bytes, conf=conf)

    def predict_claim(
        self,
        claim_id: str,
        conf: Optional[float] = None,
        redis_client: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Run inference resolving an image directly from a Claim-Check token in Redis RAM.

        Args:
            claim_id (str): Claim-Check token string (e.g. 'wkafka:claim:<uuid>').
            conf (Optional[float]): Optional confidence threshold override. Defaults to None.
            redis_client (Optional[Any]): Optional Redis client override. Defaults to None.

        Returns:
            List[Dict[str, Any]]: Canonical list of detected objects.
        """
        client = redis_client or self.redis_client
        img = ClaimResolver.resolve_claim(claim_id, redis_client=client)
        return self.predict(source=img, conf=conf)

    def predict_tensor(
        self,
        tensor: Any,
        conf: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Run Zero-Copy inference directly from a PyTorch CUDA or CuPy GPU tensor.

        Args:
            tensor (Any): PyTorch or CuPy tensor residing in GPU VRAM.
            conf (Optional[float]): Optional confidence threshold override. Defaults to None.

        Returns:
            List[Dict[str, Any]]: Canonical list of detected objects.
        """
        return self.predict(source=tensor, conf=conf)

    def predict_batch(
        self,
        sources: List[Union[str, bytes, np.ndarray, Image.Image, Any]],
        conf: Optional[float] = None,
    ) -> List[List[Dict[str, Any]]]:
        """Run batched object detection inference across multiple images.

        Args:
            sources (List[Union[str, bytes, np.ndarray, Image.Image, Any]]): List of image inputs.
            conf (Optional[float]): Optional confidence threshold override. Defaults to None.

        Returns:
            List[List[Dict[str, Any]]]: Detections per image.
        """
        if not sources:
            return []

        active_conf = conf if conf is not None else self.conf
        profiler = LatencyBreakdown() if self.profiling else None
        t_start = time.perf_counter()

        # Step 1: Preprocess
        t_pre = time.perf_counter()
        preprocessed_list = [self._preprocess_image(src) for src in sources]
        if profiler:
            profiler.preprocess_ms = (time.perf_counter() - t_pre) * 1000.0

        if self.trt_model is not None:
            batch_tensor = np.stack(preprocessed_list, axis=0)

            t_gpu = time.perf_counter()
            raw_output = self.trt_model.infer_batch(batch_tensor)
            if profiler:
                profiler.gpu_inference_ms = (time.perf_counter() - t_gpu) * 1000.0

            t_post = time.perf_counter()
            results = self.trt_model.postprocess(
                raw_output,
                conf_threshold=active_conf,
                class_names=self.class_names,
            )
            if profiler:
                profiler.postprocess_ms = (time.perf_counter() - t_post) * 1000.0
                profiler.total_ms = (time.perf_counter() - t_start) * 1000.0
                profiler.log_summary(batch_size=len(sources))

            return results

        if self.pt_model is not None:
            decoded_images: List[Any] = []
            for src in sources:
                src_resolved = self._resolve_source_to_image(src)
                if isinstance(src_resolved, bytes):
                    buf = np.frombuffer(src_resolved, dtype=np.uint8)
                    decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                    decoded_images.append(
                        decoded if decoded is not None else src_resolved
                    )
                else:
                    decoded_images.append(src_resolved)

            t_gpu = time.perf_counter()
            results = self.pt_model.predict(
                source=decoded_images,
                conf=active_conf,
                imgsz=self.imgsz,
                verbose=False,
            )
            if profiler:
                profiler.gpu_inference_ms = (time.perf_counter() - t_gpu) * 1000.0

            t_post = time.perf_counter()
            batch_res: List[List[Dict[str, Any]]] = []
            for r in results:
                dets: List[Dict[str, Any]] = []
                if hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0:
                    xyxy = r.boxes.xyxy.cpu().numpy()
                    confs = r.boxes.conf.cpu().numpy()
                    classes = r.boxes.cls.cpu().numpy()

                    for box, score, cls_id in zip(xyxy, confs, classes):
                        cid = int(cls_id)
                        name = self.class_names.get(cid) or (
                            r.names.get(cid) if hasattr(r, "names") else str(cid)
                        )
                        x1, y1, x2, y2 = [int(round(coord)) for coord in box]
                        dets.append(
                            {
                                "bbox": [x1, y1, x2, y2],
                                "conf": round(float(score), 4),
                                "conf_percentage": round(float(score) * 100.0, 2),
                                "class_id": cid,
                                "name": name,
                                "status": True,
                            }
                        )
                batch_res.append(dets)

            if profiler:
                profiler.postprocess_ms = (time.perf_counter() - t_post) * 1000.0
                profiler.total_ms = (time.perf_counter() - t_start) * 1000.0
                profiler.log_summary(batch_size=len(sources))

            return batch_res

        return []
