"""Unified YOLO inference module supporting PyTorch and TensorRT engine backends."""

import os
from typing import Any, Dict, List, Optional, Union

import cv2
from loguru import logger
import numpy as np
from PIL import Image
from ultralytics import YOLO

from .engine_exporter import ensure_tensorrt_engine
from .tensorrt_engine import TRT_AVAILABLE, TensorRTYOLO


class YOLOInference:
    """Unified high-performance YOLO predictor supporting .pt and .engine model formats."""

    def __init__(
        self,
        model_path: str,
        conf: float = 0.25,
        imgsz: int = 640,
        max_batch_size: int = 16,
        class_names: Optional[Dict[int, str]] = None,
        auto_engine: bool = True,
        half: bool = True,
        int8: bool = False,
        data: Optional[str] = None,
    ) -> None:
        """Initialize YOLOInference predictor.

        Args:
            model_path (str): Path to .pt or .engine model file.
            conf (float): Default confidence threshold for detections. Defaults to 0.25.
            imgsz (int): Square image resolution expected by model input. Defaults to 640.
            max_batch_size (int): Maximum batch size for TensorRT engine. Defaults to 16.
            class_names (Optional[Dict[int, str]]): Class ID to name dictionary. Defaults to None.
            auto_engine (bool): Automatically resolve or compile TensorRT engine on startup. Defaults to True.
            half (bool): Enable FP16 quantization when compiling TensorRT engine. Defaults to True.
            int8 (bool): Enable INT8 quantization when compiling TensorRT engine. Defaults to False.
            data (Optional[str]): Dataset YAML path for INT8 calibration. Defaults to None.

        Raises:
            FileNotFoundError: If model file does not exist.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if auto_engine and model_path.endswith(".pt"):
            model_path = ensure_tensorrt_engine(
                model_path=model_path,
                imgsz=imgsz,
                half=half,
                int8=int8,
                data=data,
                batch=max_batch_size,
            )

        self.model_path: str = model_path
        self.conf: float = conf
        self.imgsz: int = imgsz
        self.max_batch_size: int = max_batch_size
        self.class_names: Dict[int, str] = class_names or {}

        self.is_engine: bool = model_path.endswith(".engine")
        self.trt_model: Optional[TensorRTYOLO] = None
        self.pt_model: Optional[YOLO] = None

        if self.is_engine and TRT_AVAILABLE:
            try:
                self.trt_model = TensorRTYOLO(
                    engine_path=model_path,
                    max_batch_size=max_batch_size,
                    imgsz=imgsz,
                )
                logger.info(f"Loaded TensorRT engine directly via PyCUDA: {model_path}")
            except Exception as e:
                logger.warning(
                    f"Direct TensorRTYOLO load failed ({e}), falling back to Ultralytics engine runner."
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

    def _preprocess_image(
        self, source: Union[str, bytes, np.ndarray, Image.Image]
    ) -> np.ndarray:
        """Convert input image source into normalized float32 CHW array (3, imgsz, imgsz).

        Args:
            source (Union[str, bytes, np.ndarray, Image.Image]): Image input in various formats.

        Returns:
            np.ndarray: Preprocessed float32 array with shape (3, imgsz, imgsz).

        Raises:
            ValueError: If source format is unsupported or image decoding fails.
        """
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
        source: Union[str, bytes, np.ndarray, Image.Image],
        conf: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Run object detection inference on a single image.

        Args:
            source (Union[str, bytes, np.ndarray, Image.Image]): Image source.
            conf (Optional[float]): Optional confidence threshold override. Defaults to None.

        Returns:
            List[Dict[str, Any]]: Canonical list of detected objects.
        """
        batch_results = self.predict_batch([source], conf=conf)
        return batch_results[0] if batch_results else []

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

    def predict_batch(
        self,
        sources: List[Union[str, bytes, np.ndarray, Image.Image]],
        conf: Optional[float] = None,
    ) -> List[List[Dict[str, Any]]]:
        """Run batched object detection inference across multiple images.

        Args:
            sources (List[Union[str, bytes, np.ndarray, Image.Image]]): List of image inputs.
            conf (Optional[float]): Optional confidence threshold override. Defaults to None.

        Returns:
            List[List[Dict[str, Any]]]: Detections per image.
        """
        if not sources:
            return []

        active_conf = conf if conf is not None else self.conf

        if self.trt_model is not None:
            preprocessed_list = [self._preprocess_image(src) for src in sources]
            batch_tensor = np.stack(preprocessed_list, axis=0)
            raw_output = self.trt_model.infer_batch(batch_tensor)
            return self.trt_model.postprocess(
                raw_output,
                conf_threshold=active_conf,
                class_names=self.class_names,
            )

        if self.pt_model is not None:
            decoded_images: List[Any] = []
            for src in sources:
                if isinstance(src, bytes):
                    buf = np.frombuffer(src, dtype=np.uint8)
                    decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                    decoded_images.append(decoded if decoded is not None else src)
                else:
                    decoded_images.append(src)

            results = self.pt_model.predict(
                source=decoded_images,
                conf=active_conf,
                imgsz=self.imgsz,
                verbose=False,
            )

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
            return batch_res

        return [[] for _ in sources]
