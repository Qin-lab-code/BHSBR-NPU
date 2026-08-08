#!/usr/bin/env bash
set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CANN_ENV="/usr/local/Ascend/ascend-toolkit/set_env.sh"

if [[ ! -f "${CANN_ENV}" ]]; then
  echo "CANN environment script not found: ${CANN_ENV}" >&2
  exit 1
fi

export PYTHONPATH="${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
set +u
source "${CANN_ENV}"
set -u
python -c "import yaml, importlib_metadata" >/dev/null
exec python "${PROJECT_DIR}/main.py" --device npu --device-id 0 "$@"
