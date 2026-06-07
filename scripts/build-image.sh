#!/usr/bin/env bash
# Build the SM75 vLLM image with BuildKit (pip cache mount + China mirror).
set -euo pipefail

IMAGE="${IMAGE:-vllm-sm75:latest}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT"
echo "[build] image=${IMAGE} context=${ROOT}"
DOCKER_BUILDKIT=1 docker build \
  --progress=plain \
  -t "${IMAGE}" \
  "$@" \
  .
echo "[build] done: ${IMAGE}"
docker images "${IMAGE%%:*}"
