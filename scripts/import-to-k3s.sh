#!/usr/bin/env bash
# Import a locally-built docker image into the k3s containerd image store so
# pods can use it with imagePullPolicy: Never (no registry required).
set -euo pipefail

IMAGE="${IMAGE:-vllm-sm75:latest}"
echo "[import] saving ${IMAGE} from docker -> k3s containerd ..."
docker save "${IMAGE}" | sudo k3s ctr images import -
echo "[import] present in k3s:"
sudo k3s ctr images ls -q | grep -F "${IMAGE}" || {
  echo "[import] WARNING: ${IMAGE} not found in k3s image list"; exit 1; }
echo "[import] done"
