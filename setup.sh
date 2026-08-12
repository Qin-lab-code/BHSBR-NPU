#!/usr/bin/env bash
set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CANN_ENV="/usr/local/Ascend/ascend-toolkit/set_env.sh"
MS_VERSION="${MS_VERSION:-2.9.0}"

if [[ ! -f "${CANN_ENV}" ]]; then
  echo "ERROR: CANN environment script not found: ${CANN_ENV}" >&2
  exit 1
fi

export PYTHONPATH="${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
set +u
source "${CANN_ENV}"

python -m pip install "mindspore==${MS_VERSION}" \
  -i https://repo.mindspore.cn/pypi/simple \
  --trusted-host repo.mindspore.cn \
  --extra-index-url https://repo.huaweicloud.com/repository/pypi/simple/
python -m pip install numpy scipy

python - <<'PY'
import ctypes
import mindspore as ms

for library in ("libascendcl.so", "libhccl.so"):
    ctypes.CDLL(library)

ms.set_context(device_target="Ascend", device_id=0)
print("MindSpore:", ms.__version__)
print("Device target:", ms.get_context("device_target"))
ms.run_check()
PY

echo "Setup completed. Example:"
echo "  bash run_mindspore.sh --dataset Computers --batch-size 128"
