#!/usr/bin/env bash
set -euo pipefail

readonly TENSORRT_COMMIT="c93b7d4893184af4882f9f1862e13a5c64b8677d"
readonly TARGET_DIR="${1:-work_dirs/centerpoint_deployment/deps/TensorRT}"

if [[ -e "${TARGET_DIR}" ]]; then
  if [[ -f "${TARGET_DIR}/include/NvInferVersion.h" ]] && \
     [[ "$(git -C "${TARGET_DIR}" rev-parse HEAD)" == "${TENSORRT_COMMIT}" ]]; then
    echo "TensorRT 11.2.1.2 headers already present: ${TARGET_DIR}"
    exit 0
  fi
  echo "Refusing to replace existing non-matching path: ${TARGET_DIR}" >&2
  exit 1
fi

mkdir -p "$(dirname "${TARGET_DIR}")"
git init "${TARGET_DIR}"
git -C "${TARGET_DIR}" remote add origin https://github.com/NVIDIA/TensorRT.git
git -C "${TARGET_DIR}" sparse-checkout init --cone
git -C "${TARGET_DIR}" sparse-checkout set include
git -C "${TARGET_DIR}" fetch --depth 1 origin "${TENSORRT_COMMIT}"
git -C "${TARGET_DIR}" checkout --detach FETCH_HEAD
echo "Fetched TensorRT 11.2.1.2 headers: ${TARGET_DIR}"
