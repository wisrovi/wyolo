"""Unit tests for wyolo inference subpackage and TensorRT acceleration."""

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest
from PIL import Image

from wyolo.inference.engine_exporter import export_to_engine
from wyolo.inference.predictor import YOLOInference
from wyolo.inference.server import YOLOTensorRTAPI, serve_yolo
from wyolo.inference.tensorrt_engine import TensorRTYOLO


class TestEngineExporter:
    """Test suite for YOLO to TensorRT engine exporter."""

    def test_export_file_not_found(self) -> None:
        """Verify FileNotFoundError is raised when input model file does not exist."""
        with pytest.raises(FileNotFoundError, match="Source model file not found"):
            export_to_engine("/non/existent/model.pt")

    @patch("wyolo.inference.engine_exporter.YOLO")
    @patch("os.path.exists")
    def test_export_success(self, mock_exists: MagicMock, mock_yolo: MagicMock) -> None:
        """Verify successful model export calls YOLO.export with expected parameters."""
        mock_exists.return_value = True
        mock_model = MagicMock()
        mock_model.export.return_value = "/tmp/test_model.engine"
        mock_yolo.return_value = mock_model

        result = export_to_engine(
            model_path="/tmp/test_model.pt",
            imgsz=640,
            half=True,
            dynamic=True,
            batch=16,
            nms=True,
        )

        assert result == "/tmp/test_model.engine"
        mock_model.export.assert_called_once_with(
            format="engine",
            imgsz=640,
            half=True,
            int8=False,
            dynamic=True,
            batch=16,
            nms=True,
            device=0,
        )

    @patch("wyolo.inference.engine_exporter.YOLO")
    @patch("os.path.exists")
    def test_export_int8_success(
        self, mock_exists: MagicMock, mock_yolo: MagicMock
    ) -> None:
        """Verify successful INT8 model export with calibration dataset."""
        mock_exists.return_value = True
        mock_model = MagicMock()
        mock_model.export.return_value = "/tmp/test_int8_model.engine"
        mock_yolo.return_value = mock_model

        result = export_to_engine(
            model_path="/tmp/test_model.pt",
            imgsz=640,
            half=False,
            int8=True,
            data="coco.yaml",
            dynamic=False,
            batch=8,
            nms=True,
        )

        assert result == "/tmp/test_int8_model.engine"
        mock_model.export.assert_called_once_with(
            format="engine",
            imgsz=640,
            half=False,
            int8=True,
            data="coco.yaml",
            dynamic=False,
            batch=8,
            nms=True,
            device=0,
        )


class TestTensorRTEngine:
    """Test suite for low-level TensorRT engine execution and postprocessing."""

    def test_tensorrt_file_not_found(self) -> None:
        """Verify FileNotFoundError or RuntimeError when engine file does not exist."""
        with pytest.raises((FileNotFoundError, RuntimeError)):
            TensorRTYOLO("/non/existent/model.engine")

    def test_postprocess_canonical_schema(self) -> None:
        """Verify postprocessing translates raw (B, 300, 6) tensor into canonical detection schema."""
        # Create a dummy instance bypassing __init__ via __new__
        engine = TensorRTYOLO.__new__(TensorRTYOLO)
        engine.max_batch_size = 16
        engine.imgsz = 640

        # Construct raw mock output: 1 batch, 3 predictions [x1, y1, x2, y2, conf, cls]
        raw_output = np.zeros((1, 300, 6), dtype=np.float32)
        raw_output[0, 0] = [10.2, 20.4, 100.8, 200.1, 0.95, 0.0]  # Valid detection
        raw_output[0, 1] = [50.0, 60.0, 150.0, 160.0, 0.15, 1.0]  # Low confidence
        raw_output[0, 2] = [30.0, 40.0, 80.0, 90.0, 0.85, 2.0]  # Valid detection

        class_names = {0: "car", 1: "truck", 2: "motorcycle"}
        results = engine.postprocess(
            raw_output, conf_threshold=0.25, class_names=class_names
        )

        assert len(results) == 1
        dets = results[0]
        assert len(dets) == 2  # Low confidence filtered out

        # Validate first detection
        assert dets[0]["name"] == "car"
        assert dets[0]["class_id"] == 0
        assert dets[0]["conf"] == 0.95
        assert dets[0]["conf_percentage"] == 95.0
        assert dets[0]["bbox"] == [10, 20, 101, 200]
        assert dets[0]["status"] is True

        # Validate second detection
        assert dets[1]["name"] == "motorcycle"
        assert dets[1]["class_id"] == 2
        assert dets[1]["conf"] == 0.85
        assert dets[1]["conf_percentage"] == 85.0
        assert dets[1]["bbox"] == [30, 40, 80, 90]


class TestYOLOInference:
    """Test suite for high-level unified YOLOInference predictor."""

    def test_model_not_found(self) -> None:
        """Verify FileNotFoundError is raised if model does not exist."""
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            YOLOInference("/non/existent/model.pt")

    @patch("wyolo.inference.predictor.YOLO")
    @patch("os.path.exists")
    def test_predict_preprocessing_formats(
        self, mock_exists: MagicMock, mock_yolo: MagicMock
    ) -> None:
        """Verify predictor accurately handles NumPy, PIL Image, and raw byte arrays."""
        mock_exists.return_value = True
        mock_model = MagicMock()
        mock_model.names = {0: "person", 1: "bicycle"}
        mock_yolo.return_value = mock_model

        predictor = YOLOInference(
            model_path="/tmp/yolov8n.pt",
            conf=0.3,
            imgsz=640,
        )

        # Test preprocessing with NumPy array
        np_img = np.zeros((480, 640, 3), dtype=np.uint8)
        chw = predictor._preprocess_image(np_img)
        assert chw.shape == (3, 640, 640)
        assert chw.dtype == np.float32

        # Test preprocessing with PIL Image
        pil_img = Image.new("RGB", (320, 240), color="red")
        chw_pil = predictor._preprocess_image(pil_img)
        assert chw_pil.shape == (3, 640, 640)

        # Test preprocessing with JPEG bytes
        _, encoded = cv2.imencode(".jpg", np_img)
        jpeg_bytes = encoded.tobytes()
        chw_bytes = predictor._preprocess_image(jpeg_bytes)
        assert chw_bytes.shape == (3, 640, 640)

    @patch("wyolo.inference.predictor.YOLO")
    @patch("os.path.exists")
    def test_predict_pt_flow(
        self, mock_exists: MagicMock, mock_yolo: MagicMock
    ) -> None:
        """Verify prediction execution and canonical output parsing with PyTorch backend."""
        mock_exists.return_value = True
        mock_model = MagicMock()
        mock_model.names = {0: "car"}

        # Mock result boxes
        mock_box = MagicMock()
        mock_box.__len__.return_value = 1
        mock_box.xyxy = MagicMock()
        mock_box.xyxy.cpu().numpy.return_value = np.array([[12.0, 34.0, 56.0, 78.0]])
        mock_box.conf = MagicMock()
        mock_box.conf.cpu().numpy.return_value = np.array([0.92])
        mock_box.cls = MagicMock()
        mock_box.cls.cpu().numpy.return_value = np.array([0.0])

        mock_pred = MagicMock()
        mock_pred.boxes = mock_box
        mock_pred.names = {0: "car"}
        mock_model.predict.return_value = [mock_pred]
        mock_yolo.return_value = mock_model

        predictor = YOLOInference(model_path="/tmp/detector.pt", conf=0.5)

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        results = predictor.predict(img)

        assert len(results) == 1
        assert results[0]["name"] == "car"
        assert results[0]["class_id"] == 0
        assert results[0]["conf"] == 0.92
        assert results[0]["bbox"] == [12, 34, 56, 78]
        assert results[0]["status"] is True

        # Test predict_bytes
        _, encoded = cv2.imencode(".jpg", img)
        bytes_res = predictor.predict_bytes(encoded.tobytes())
        assert len(bytes_res) == 1
        assert bytes_res[0]["name"] == "car"


class TestLitServeAPI:
    """Test suite for LitServe server and request handler."""

    def test_decode_and_encode_flow(self) -> None:
        """Validate request decoding, batching, unbatching, and response encoding."""
        # Create a dummy image and encode as JPEG bytes
        dummy_img = np.zeros((120, 160, 3), dtype=np.uint8)
        _, enc = cv2.imencode(".jpg", dummy_img)
        jpeg_bytes = enc.tobytes()

        # Instantiate API object
        try:
            api = YOLOTensorRTAPI()
        except RuntimeError:
            pytest.skip("LitServe not installed in this environment.")

        api.imgsz = 640
        api.conf_threshold = 0.5
        api.max_batch_size = 16

        # 1. decode_request
        chw = api.decode_request({"image": jpeg_bytes})
        assert chw.shape == (3, 640, 640)

        # 2. batch
        batched = api.batch([chw, chw])
        assert batched.shape == (2, 3, 640, 640)

        # 3. unbatch
        mock_output = np.zeros((2, 300, 6), dtype=np.float32)
        mock_output[0, 0] = [10.0, 10.0, 50.0, 50.0, 0.9, 0.0]
        unbatched = api.unbatch(mock_output)
        assert len(unbatched) == 2
        assert unbatched[0].shape == (300, 6)

        # 4. encode_response
        resp = api.encode_response(unbatched[0])
        assert "detections" in resp
        assert len(resp["detections"]) == 1
        assert resp["detections"][0]["conf"] == 0.9
        assert resp["detections"][0]["bbox"] == [10, 10, 50, 50]

    def test_serve_yolo_missing_dependency(self) -> None:
        """Verify serve_yolo raises RuntimeError when LitServe is unavailable."""
        with patch("wyolo.inference.server.LITSERVE_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="LitServe is required"):
                serve_yolo("dummy.engine")
