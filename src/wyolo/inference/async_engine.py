"""Async Future wrapper and worker for non-blocking GPU inference."""

import concurrent.futures
from typing import Any, Callable, Dict, List, Optional


class AsyncInferenceFuture:
    """Future representing an asynchronous YOLO inference task execution."""

    def __init__(self, future: concurrent.futures.Future) -> None:
        """Initialize AsyncInferenceFuture.

        Args:
            future (concurrent.futures.Future): Underlying Python standard future.
        """
        self._future: concurrent.futures.Future = future

    def done(self) -> bool:
        """Check if asynchronous inference has completed.

        Returns:
            bool: True if completed, False otherwise.
        """
        return self._future.done()

    def result(self, timeout: Optional[float] = None) -> List[Dict[str, Any]]:
        """Block until inference is completed and retrieve canonical detections.

        Args:
            timeout (Optional[float]): Optional timeout in seconds to wait.

        Returns:
            List[Dict[str, Any]]: Canonical list of detected objects.
        """
        return self._future.result(timeout=timeout)


class AsyncWorkerPool:
    """Thread pool worker executing CUDA asynchronous inference tasks."""

    def __init__(self, max_workers: int = 2) -> None:
        """Initialize AsyncWorkerPool.

        Args:
            max_workers (int): Number of worker threads. Defaults to 2.
        """
        self.executor: concurrent.futures.ThreadPoolExecutor = (
            concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers, thread_name_prefix="WYoloAsyncWorker"
            )
        )

    def submit(
        self, fn: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> AsyncInferenceFuture:
        """Submit a task to the pool and return an AsyncInferenceFuture.

        Args:
            fn (Callable): Function to execute.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            AsyncInferenceFuture: Wrapped future object.
        """
        f = self.executor.submit(fn, *args, **kwargs)
        return AsyncInferenceFuture(f)

    def shutdown(self) -> None:
        """Shutdown thread pool."""
        self.executor.shutdown(wait=False)
