# vllm-sm75-docker

为 **NVIDIA Turing（SM75 / RTX 2080 Ti）** 维护的 vLLM 可复现部署：镜像构建 + k8s（k3s）部署。

服务模型：`Qwen3.6-27B-AWQ`，拓扑 **DP2 × TP2（4×2080Ti）**，200K 上下文。

## 组成

```
Dockerfile                                  多阶段镜像（builder 编 wheel / runtime 装 wheel+flashinfer）
patches/flashinfer-0.6.12-sm75-ebo.patch    FlashInfer 0.6.12 的 SM75 共享内存(EBO)修复（head_dim=256）
scripts/build-image.sh                      本机构建镜像（BuildKit + pip 缓存 + 清华源）
scripts/import-to-k3s.sh                    把镜像导入 k3s containerd（免 registry）
scripts/deploy-k8s.sh                       应用 k8s manifests 并等待就绪
k8s/                                        namespace / deployment / service
```

## 关键设计

- **vLLM 来源**：fork [`HuChundong/vllm`](https://github.com/HuChundong/vllm) 的 `sm75-upstream-main`
  分支（基于 `v0.22.1rc0`）。SM75 改动：跳过 FA2/FlashMLA 编译、FA 扩展设为 optional、
  rotary/flash_attn 运行时优雅回退。服务路径用 FlashQLA / FlashInfer / TurboQuant。
- **运行时镜像保留 CUDA devel 工具链**：flashinfer 与 FlashQLA/GDN 在运行时 JIT 编译 CUDA
  kernel，需要 `nvcc` + `ninja`。
- **FlashInfer 0.6.12 + EBO 补丁**：`SharedStorageQKVO` 用空基类优化把 NVFP4/FP8 暂存缓冲在
  非对应 dtype 时占 0 字节，使 `head_dim=256` 的 prefill 在 Turing 64KiB opt-in 共享内存内
  （否则 `cudaErrorInvalidValue`）。上游修复见 flashinfer PR #3526 / issue #3528。
- **版本钉死**：Python 3.11 · CUDA 12.8 · torch 2.11.0+cu128 · flashinfer 0.6.12 · triton 3.6，
  与已验证的裸机 conda 环境一致。

## 用法

```bash
# 1) 构建镜像（首次较久：拉基础镜像 + 编 vLLM kernel for sm_75）
scripts/build-image.sh

# 2) 导入 k3s（pod 用 imagePullPolicy: Never）
scripts/import-to-k3s.sh

# 3) 部署到 k8s 并等待就绪
scripts/deploy-k8s.sh
```

调用（NodePort 30832，api-key=abc123）：

```bash
curl -s http://<node-ip>:30832/v1/models -H "Authorization: Bearer abc123"
```

## 模型

模型不进镜像（21G）。通过 hostPath 挂载宿主 `/home/hucd/models` 到容器 `/models`（只读）。
HF 缓存与 flashinfer/torch 的 JIT 缓存持久化到宿主 `/home/hucd/.cache/k8s-vllm`，
首次编译后复用，避免每次重编。

## 升级 / 重建

```bash
cd ~/qwen3_6/vllm && git fetch origin && git rebase <new-upstream-tag>   # 同步上游
# 解决冲突后推回 fork，再重建镜像
git push fork sm75-upstream-main
scripts/build-image.sh && scripts/import-to-k3s.sh
kubectl -n vllm rollout restart deploy/vllm-qwen3
```
