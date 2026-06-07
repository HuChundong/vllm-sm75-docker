#!/usr/bin/env bash
# Apply the k8s manifests and wait for the deployment to become ready.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

kubectl apply -f "$ROOT/k8s/namespace.yaml"
kubectl apply -f "$ROOT/k8s/deployment.yaml"
kubectl apply -f "$ROOT/k8s/service.yaml"

echo "[deploy] waiting for rollout (model load + JIT can take several minutes) ..."
kubectl -n vllm rollout status deploy/vllm-qwen3 --timeout=1800s

echo "[deploy] pods:"
kubectl -n vllm get pods -o wide
echo "[deploy] service:"
kubectl -n vllm get svc vllm-qwen3
