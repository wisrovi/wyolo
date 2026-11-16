"""Zero-Copy and GPU tensor memory utilities for YOLO inference."""

# pylint: disable=broad-exception-caught,import-outside-toplevel

from typing import Any, Tuple


def is_torch_cuda_tensor(obj: Any) -> bool:
    """Check if an object is a PyTorch CUDA tensor.

    Args:
        obj (Any): Candidate object to test.

    Returns:
        bool: True if obj is a torch.Tensor residing on a CUDA device, False otherwise.
    """
    if not (hasattr(obj, "is_cuda") and hasattr(obj, "data_ptr")):
        return False
    try:
        import torch

        return isinstance(obj, torch.Tensor) and bool(obj.is_cuda)
    except Exception:
        return False


def is_cupy_array(obj: Any) -> bool:
    """Check if an object is a CuPy array.

    Args:
        obj (Any): Candidate object to test.

    Returns:
        bool: True if obj is a cupy.ndarray, False otherwise.
    """
    if not (hasattr(obj, "data") and hasattr(obj, "shape")):
        return False
    try:
        import cupy

        return isinstance(obj, cupy.ndarray)
    except Exception:
        return False


def get_cuda_ptr(obj: Any) -> Tuple[int, Tuple[int, ...]]:
    """Extract raw GPU device pointer and shape from PyTorch or CuPy array.

    Args:
        obj (Any): PyTorch or CuPy tensor.

    Returns:
        Tuple[int, Tuple[int, ...]]: Raw integer device memory pointer and tensor shape.

    Raises:
        TypeError: If obj is not a supported GPU tensor type.
    """
    if is_torch_cuda_tensor(obj):
        return obj.data_ptr(), tuple(obj.shape)
    if is_cupy_array(obj):
        return obj.data.ptr, tuple(obj.shape)
    raise TypeError(f"Object {type(obj)} is not a supported CUDA GPU tensor.")
