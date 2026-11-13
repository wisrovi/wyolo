"""Module for exporting YOLO PyTorch models to TensorRT engine format."""

import os
import shutil
from typing import Optional

from loguru import logger
from ultralytics import YOLO


def export_to_engine(
    model_path: str,
    imgsz: int = 640,
    half: bool = True,
    dynamic: bool = True,
    batch: int = 16,
    nms: bool = True,
    device: int = 0,
    workspace: Optional[int] = None,
) -> str:
    """Export a YOLO PyTorch (.pt) model to TensorRT (.engine) format with FP16 and NMS.

    Args:
        model_path (str): Path to the source .pt model weights file.
        imgsz (int): Target square image dimension for inference. Defaults to 640.
        half (bool): Enable FP16 half-precision quantization. Defaults to True.
        dynamic (bool): Enable dynamic batch size support (1 to batch). Defaults to True.
        batch (int): Maximum batch size supported by the engine. Defaults to 16.
        nms (bool): Embed Non-Maximum Suppression (EfficientNMS) inside the engine. Defaults to True.
        device (int): CUDA device index to compile the engine on. Defaults to 0.
        workspace (Optional[int]): TensorRT workspace memory limit in GiB. Defaults to None.

    Returns:
        str: Path to the exported .engine file.

    Raises:
        FileNotFoundError: If the source model_path does not exist.
        RuntimeError: If export fails or produces an invalid file.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Source model file not found: {model_path}")

    logger.info(
        f"Exporting model '{model_path}' to TensorRT engine (imgsz={imgsz}, "
        f"half={half}, dynamic={dynamic}, batch={batch}, nms={nms}, device={device})..."
    )

    try:
        model = YOLO(model_path)
        export_kwargs = {
            "format": "engine",
            "imgsz": imgsz,
            "half": half,
            "dynamic": dynamic,
            "batch": batch,
            "nms": nms,
            "device": device,
        }
        if workspace is not None:
            export_kwargs["workspace"] = workspace

        exported_result = model.export(**export_kwargs)
        exported_path = str(exported_result) if exported_result else ""

        if not exported_path or not os.path.exists(exported_path):
            expected_engine = os.path.splitext(model_path)[0] + ".engine"
            if os.path.exists(expected_engine):
                exported_path = expected_engine
            else:
                raise RuntimeError(
                    f"TensorRT export completed but engine file could not be found at {exported_path}"
                )

        logger.info(f"TensorRT engine successfully exported to: {exported_path}")
        return exported_path

    except Exception as exc:
        logger.error(f"Failed to export '{model_path}' to TensorRT engine: {exc}")
        raise RuntimeError(f"TensorRT export failed: {exc}") from exc


def ensure_tensorrt_engine(
    model_path: str,
    imgsz: int = 640,
    half: bool = True,
    dynamic: bool = True,
    batch: int = 16,
    nms: bool = True,
    device: int = 0,
    cache_dir: Optional[str] = None,
) -> str:
    """Ensure a TensorRT engine exists for the model, compiling it if necessary.

    Checks if a compiled TensorRT (.engine) file already exists for the given
    PyTorch model (.pt). If not present, checks for CUDA and TensorRT availability
    and compiles the model on startup with FP16, dynamic batching, and GPU NMS.
    Handles read-only directories by writing engines to a cache directory.
    Falls back gracefully to the original .pt model if TensorRT or GPU is unavailable.

    Args:
        model_path (str): Path to the model file (.pt or .engine).
        imgsz (int): Target square image dimension for inference. Defaults to 640.
        half (bool): Enable FP16 half-precision quantization. Defaults to True.
        dynamic (bool): Enable dynamic batch size support. Defaults to True.
        batch (int): Maximum batch size supported by the engine. Defaults to 16.
        nms (bool): Embed Non-Maximum Suppression inside the engine. Defaults to True.
        device (int): CUDA device index to compile the engine on. Defaults to 0.
        cache_dir (Optional[str]): Custom directory for compiled engines.

    Returns:
        str: Path to the resolved model file (either .engine or original .pt fallback).
    """
    if not model_path or model_path.endswith(".engine"):
        return model_path

    # Check for adjacent engine
    adjacent_engine = os.path.splitext(model_path)[0] + ".engine"
    if os.path.exists(adjacent_engine):
        logger.info(f"Found existing TensorRT engine: {adjacent_engine}")
        return adjacent_engine

    # Check for cached engine in cache_dir or /tmp/trt_engines
    effective_cache_dir = cache_dir or os.environ.get(
        "TRT_CACHE_DIR", "/tmp/trt_engines"
    )
    base_name = os.path.splitext(os.path.basename(model_path))[0]
    cached_engine = os.path.join(effective_cache_dir, f"{base_name}.engine")
    if os.path.exists(cached_engine):
        logger.info(f"Found cached TensorRT engine: {cached_engine}")
        return cached_engine

    # Check model file exists
    if not os.path.exists(model_path):
        logger.warning(f"Source model file does not exist: {model_path}")
        return model_path

    # Verify CUDA availability
    try:
        import torch

        if not torch.cuda.is_available():
            logger.info(f"CUDA is not available. Using PyTorch model: {model_path}")
            return model_path
    except ImportError:
        logger.info(f"Torch not available. Using original model: {model_path}")
        return model_path

    # Verify TensorRT availability
    try:
        import tensorrt  # noqa: F401
    except ImportError:
        logger.info(
            f"TensorRT not installed in environment. Using PyTorch model: {model_path}"
        )
        return model_path

    # Check directory write permissions
    model_dir = os.path.dirname(os.path.abspath(model_path))
    is_writable = os.access(model_dir, os.W_OK)
    export_source = model_path

    if not is_writable:
        os.makedirs(effective_cache_dir, exist_ok=True)
        export_source = os.path.join(effective_cache_dir, os.path.basename(model_path))
        try:
            shutil.copy2(model_path, export_source)
        except Exception as exc:
            logger.warning(
                f"Could not copy model to cache dir {effective_cache_dir}: {exc}. "
                "Falling back to .pt"
            )
            return model_path

    try:
        engine_path = export_to_engine(
            model_path=export_source,
            imgsz=imgsz,
            half=half,
            dynamic=dynamic,
            batch=batch,
            nms=nms,
            device=device,
        )

        if (
            not is_writable
            and os.path.exists(export_source)
            and export_source != model_path
        ):
            try:
                os.remove(export_source)
            except OSError:
                pass

        if engine_path and os.path.exists(engine_path):
            logger.info(f"TensorRT engine ready: {engine_path}")
            return engine_path

        expected = os.path.splitext(export_source)[0] + ".engine"
        if os.path.exists(expected):
            logger.info(f"TensorRT engine ready: {expected}")
            return expected

        return model_path

    except Exception as exc:
        logger.warning(
            f"TensorRT export failed ({exc}). Gracefully falling back to PyTorch model: {model_path}"
        )
        if (
            not is_writable
            and os.path.exists(export_source)
            and export_source != model_path
        ):
            try:
                os.remove(export_source)
            except OSError:
                pass
        return model_path
