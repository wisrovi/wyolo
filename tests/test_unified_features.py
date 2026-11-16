"""Unit tests for WYolo unified inference features:
quantization simplification, async engine, claim-check resolution,
zero-copy tensor ingestion, dynamic batching, profiler, and WPipe step.
"""

import time
from typing import Any, Dict, List
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import torch

from wyolo.inference.async_engine import AsyncInferenceFuture, AsyncWorkerPool
from wyolo.inference.claim_resolver import ClaimResolver
from wyolo.inference.dynamic_batcher import DynamicBatcher
from wyolo.inference.predictor import YOLOInference
from wyolo.inference.profiler import LatencyBreakdown
from wyolo.inference.zero_copy import get_cuda_ptr, is_cupy_array, is_torch_cuda_tensor
from wyolo.wpipe import WYoloInferenceStep


class TestLatencyProfiler(unittest.TestCase):
    """Test LatencyBreakdown profiler and bottleneck detection."""

    def test_latency_profiler_metrics_and_dict(self):
        profiler = LatencyBreakdown(warn_threshold_ratio=1.5)
        profiler.preprocess_ms = 30.0
        profiler.pcie_transfer_ms = 1.0
        profiler.gpu_inference_ms = 10.0
        profiler.postprocess_ms = 2.0
        profiler.total_ms = 43.0

        d = profiler.to_dict()
        self.assertEqual(d["preprocess_ms"], 30.0)
        self.assertEqual(d["gpu_inference_ms"], 10.0)
        self.assertEqual(d["total_ms"], 43.0)

        # Logging summary triggers CPU bottleneck warning without error
        profiler.log_summary(batch_size=1)


class TestClaimResolver(unittest.TestCase):
    """Test Claim-Check token pattern recognition and Redis resolution."""

    def test_is_claim_token(self):
        self.assertTrue(ClaimResolver.is_claim_token("wkafka:claim:abc-123"))
        self.assertTrue(ClaimResolver.is_claim_token("claim:xyz-789"))
        self.assertFalse(ClaimResolver.is_claim_token("/path/to/image.jpg"))
        self.assertFalse(ClaimResolver.is_claim_token(b"\x00\x01\x02"))
        self.assertFalse(ClaimResolver.is_claim_token(None))

    def test_resolve_claim_success(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        encoded_bytes = cv2.imencode(".jpg", img)[1].tobytes()

        mock_redis = MagicMock()
        mock_redis.get.return_value = encoded_bytes

        resolved = ClaimResolver.resolve_claim(
            "wkafka:claim:test-uuid", redis_client=mock_redis
        )
        self.assertIsInstance(resolved, np.ndarray)
        self.assertEqual(resolved.shape, (100, 100, 3))
        mock_redis.get.assert_called_with("wkafka:claim:test-uuid")

    def test_resolve_claim_missing_key(self):
        mock_redis = MagicMock()
        mock_redis.get.return_value = None

        with self.assertRaises(ValueError):
            ClaimResolver.resolve_claim("wkafka:claim:missing", redis_client=mock_redis)


class TestZeroCopyUtilities(unittest.TestCase):
    """Test CUDA tensor and pointer introspection."""

    def test_torch_cpu_vs_cuda(self):
        cpu_t = torch.zeros((3, 100, 100))
        self.assertFalse(is_torch_cuda_tensor(cpu_t))
        self.assertFalse(is_cupy_array(cpu_t))

        with self.assertRaises(TypeError):
            get_cuda_ptr(cpu_t)


class TestDynamicBatcher(unittest.TestCase):
    """Test DynamicBatcher grouping concurrent inputs."""

    def test_dynamic_batcher_accumulation(self):
        call_batches = []

        def mock_infer(batch: List[Any]) -> List[List[Dict[str, Any]]]:
            call_batches.append(len(batch))
            return [[{"name": f"item_{i}"}] for i in range(len(batch))]

        batcher = DynamicBatcher(
            infer_fn=mock_infer, max_batch_size=4, max_delay_ms=20.0
        )

        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            f1 = executor.submit(batcher.submit, "item1")
            f2 = executor.submit(batcher.submit, "item2")
            f3 = executor.submit(batcher.submit, "item3")
            r1 = f1.result()
            r2 = f2.result()
            r3 = f3.result()

        batcher.shutdown()
        self.assertTrue(len(r1) > 0)
        self.assertTrue(len(r2) > 0)
        self.assertTrue(len(r3) > 0)
        self.assertTrue(len(call_batches) >= 1)


class TestAsyncWorkerPool(unittest.TestCase):
    """Test asynchronous non-blocking future execution."""

    def test_async_worker_pool_future(self):
        pool = AsyncWorkerPool(max_workers=1)
        future = pool.submit(lambda x: [{"name": "detected", "conf": x}], 0.99)
        self.assertIsInstance(future, AsyncInferenceFuture)
        res = future.result(timeout=2.0)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["conf"], 0.99)
        pool.shutdown()


class TestWPipeStep(unittest.TestCase):
    """Test WYoloInferenceStep pipeline execution."""

    def test_wpipe_step_execution(self):
        step = WYoloInferenceStep(
            model_path="",
            classes_filter=["car"],
            input_key="image_payload",
            output_key="detections",
        )
        # Mock predictor
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = [
            {"name": "car", "conf": 0.9},
            {"name": "person", "conf": 0.8},
        ]
        step.predictor = mock_predictor

        context = {"image_payload": np.zeros((10, 10, 3), dtype=np.uint8)}
        updated = step(context)

        self.assertIn("detections", updated)
        # Only 'car' should pass the filter
        self.assertEqual(len(updated["detections"]), 1)
        self.assertEqual(updated["detections"][0]["name"], "car")
