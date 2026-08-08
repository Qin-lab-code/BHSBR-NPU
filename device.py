"""Runtime device selection for Ascend NPU, CUDA, and CPU."""

import torch

try:
    import torch_npu  # noqa: F401 - importing registers the ``npu`` backend
except ImportError:
    torch_npu = None


def resolve_device(requested="auto", index=0):
    """Return the requested accelerator, preferring Ascend NPU in auto mode."""
    requested = requested.lower()
    if requested not in {"auto", "npu", "cuda", "cpu"}:
        raise ValueError("device must be one of: auto, npu, cuda, cpu")

    npu_available = (
        torch_npu is not None
        and hasattr(torch, "npu")
        and torch.npu.is_available()
    )
    if requested == "npu" and not npu_available:
        raise RuntimeError("NPU was requested, but torch_npu or an Ascend device is unavailable")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available")

    if requested == "npu" or (requested == "auto" and npu_available):
        torch.npu.set_device(index)
        return torch.device(f"npu:{index}")
    if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()):
        torch.cuda.set_device(index)
        return torch.device(f"cuda:{index}")
    return torch.device("cpu")
