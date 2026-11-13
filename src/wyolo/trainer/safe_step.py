"""Defensive execution wrapper for WPipe pipeline steps.

Guarantees that individual step failures do not crash the overarching post-training
pipeline, preserving critical training results, weights, and primary artifacts.
"""

from typing import Any, Callable


class SafeStep:
    """Wraps a WPipe step instance to catch exceptions without breaking the pipeline."""

    def __init__(self, step_instance: Any) -> None:
        """Initializes the wrapper preserving step metadata.

        Args:
            step_instance: The callable WPipe step instance to wrap.
        """
        self.step_instance = step_instance
        self.NAME = getattr(
            step_instance,
            "NAME",
            getattr(step_instance, "__name__", step_instance.__class__.__name__),
        )
        self.VERSION = getattr(step_instance, "VERSION", "v1.0")
        self._wpipe_metadata = getattr(step_instance, "_wpipe_metadata", {})

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Executes the step within a try-except guard.

        Args:
            *args: Positional arguments passed to the step (typically context dict).
            **kwargs: Keyword arguments passed to the step.

        Returns:
            The output of the step if successful, or the original input argument if an exception occurs.
        """
        try:
            return self.step_instance(*args, **kwargs)
        except Exception as e:
            print(f"[{self.NAME}] Warning: Step execution encountered an error: {e}")
            if args and isinstance(args[0], dict):
                return args[0]
            elif args:
                return args[0]
            return {}


def safe_step(step_instance: Any) -> SafeStep:
    """Convenience factory function to wrap a step in SafeStep.

    Args:
        step_instance: The step to wrap.

    Returns:
        SafeStep: The wrapped safe step instance.
    """
    return SafeStep(step_instance)
