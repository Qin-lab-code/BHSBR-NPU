#!/usr/bin/env bash
set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CANN_ENV="/usr/local/Ascend/ascend-toolkit/set_env.sh"

if [[ ! -f "${CANN_ENV}" ]]; then
  echo "ERROR: CANN environment script not found: ${CANN_ENV}" >&2
  exit 1
fi

export PYTHONPATH="${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
set +u
source "${CANN_ENV}"

echo "Installing Python runtime dependencies..."
python -m pip install -r "${PROJECT_DIR}/requirements.txt"

echo "Checking CANN libraries and Python packages..."
python - <<'PY'
import ctypes
import sys

for library in ("libhccl.so", "libascendcl.so"):
    try:
        ctypes.CDLL(library)
    except OSError as exc:
        raise SystemExit(f"ERROR: unable to load {library}: {exc}")

try:
    import yaml
    import importlib_metadata
    import torch
    import torch_npu
except Exception as exc:
    raise SystemExit(f"ERROR: Python environment check failed: {exc}") from exc

print("Python:", sys.version.split()[0])
print("PyYAML:", yaml.__version__)
print("importlib-metadata:", importlib_metadata.version("importlib-metadata"))
print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("NPU available:", torch.npu.is_available())
print("NPU count:", torch.npu.device_count())

if not torch.npu.is_available():
    raise SystemExit("ERROR: torch_npu imported, but no usable NPU was detected")
PY

echo
echo "Setup completed successfully."
