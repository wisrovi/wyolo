"""Micro-batching dynamic dispatcher for high-throughput concurrent Kafka/HTTP YOLO inference."""

# pylint: disable=broad-exception-caught

import queue
import threading
import time
from typing import Any, Callable, Dict, List, Tuple

from loguru import logger


class DynamicBatcher:
    """Dynamic micro-batcher that collects asynchronous inference tasks into unified GPU batches."""

    def __init__(
        self,
        infer_fn: Callable[[List[Any]], List[List[Dict[str, Any]]]],
        max_batch_size: int = 16,
        max_delay_ms: float = 3.0,
    ) -> None:
        """Initialize DynamicBatcher.

        Args:
            infer_fn (Callable[[List[Any]], List[List[Dict[str, Any]]]]): Batched callback.
            max_batch_size (int): Maximum batch size accumulated per GPU dispatch. Defaults to 16.
            max_delay_ms (float): Maximum window in ms to wait for batch accumulation.
        """
        self.infer_fn: Callable[[List[Any]], List[List[Dict[str, Any]]]] = infer_fn
        self.max_batch_size: int = max_batch_size
        self.max_delay_sec: float = max_delay_ms / 1000.0

        self._queue: queue.Queue[Tuple[Any, threading.Event, Dict[str, Any]]] = (
            queue.Queue()
        )
        self._running: bool = True
        self._worker_thread: threading.Thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="WYoloDynamicBatcher"
        )
        self._worker_thread.start()
        logger.debug(
            f"DynamicBatcher started (batch={self.max_batch_size}, delay={max_delay_ms}ms)"
        )

    def submit(self, item: Any) -> List[Dict[str, Any]]:
        """Submit a single item for micro-batch execution and wait synchronously for result.

        Args:
            item (Any): Image input source.

        Returns:
            List[Dict[str, Any]]: Detections for the submitted item.
        """
        event = threading.Event()
        holder: Dict[str, Any] = {"result": None, "error": None}
        self._queue.put((item, event, holder))
        event.wait()

        err = holder.get("error")
        if isinstance(err, BaseException):
            raise err

        return holder.get("result") or []

    def _worker_loop(self) -> None:
        """Worker loop that batches items from queue within temporal window."""
        while self._running:
            try:
                first_item = self._queue.get(timeout=0.05)
            except queue.Empty:
                continue

            batch = [first_item]
            deadline = time.perf_counter() + self.max_delay_sec

            while len(batch) < self.max_batch_size:
                timeout = max(0.0, deadline - time.perf_counter())
                if timeout <= 0.0:
                    break
                try:
                    next_item = self._queue.get(timeout=timeout)
                    batch.append(next_item)
                except queue.Empty:
                    break

            items_to_infer = [b[0] for b in batch]
            try:
                batch_results = self.infer_fn(items_to_infer)
                for idx, (_, event, holder) in enumerate(batch):
                    holder["result"] = (
                        batch_results[idx] if idx < len(batch_results) else []
                    )
                    event.set()
            except Exception as exc:
                logger.error(f"Error during DynamicBatcher execution: {exc}")
                for _, event, holder in batch:
                    holder["error"] = exc
                    event.set()

    def shutdown(self) -> None:
        """Gracefully terminate worker thread."""
        self._running = False
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
