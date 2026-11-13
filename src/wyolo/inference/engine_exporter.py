"""Module for exporting YOLO PyTorch models to TensorRT engine format."""

import os
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
