# syntax=docker/dockerfile:1.7-labs
#
# vLLM for NVIDIA Turing (SM75 / RTX 2080 Ti) — reproducible serving image.
#
# Replicates the validated bare-metal conda env:
#   Python 3.11 · CUDA 12.8 · torch 2.11.0+cu128 · flashinfer 0.6.12 (+SM75 EBO patch)
#   vLLM built from the HuChundong/vllm `sm75-upstream-main` fork (FA2/FlashMLA build skipped).
#
# Runtime stage keeps the CUDA *devel* toolchain on purpose: flashinfer and the
# FlashQLA/GDN path JIT-compile CUDA kernels at runtime and need nvcc + ninja.
#
# Build (BuildKit required):
#   DOCKER_BUILDKIT=1 docker build -t vllm-sm75:latest .
#
# Per project rule, pip installs use a BuildKit cache mount + a China mirror.

ARG CUDA_TAG=12.8.1-devel-ubuntu24.04
ARG PY=3.11
ARG PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
ARG TORCH_VERSION=2.11.0
ARG FLASHINFER_VERSION=0.6.12
ARG VLLM_REPO=https://github.com/HuChundong/vllm.git
ARG VLLM_REF=sm75-upstream-main
ARG FLASHQLA_REPO=https://github.com/HuChundong/FlashQLA-SM70-SM75.git
ARG FLASHQLA_REF=sm70-sm75-gdn-forward
ARG TORCH_CUDA_ARCH_LIST=7.5

# ---------------------------------------------------------------------------
# Stage 1: builder — compile the vLLM wheel for sm_75 only.
# ---------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_TAG} AS builder
ARG PY PIP_INDEX TORCH_INDEX TORCH_VERSION VLLM_REPO VLLM_REF TORCH_CUDA_ARCH_LIST
ENV DEBIAN_FRONTEND=noninteractive

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    sed -i 's@//.*archive.ubuntu.com@//mirrors.tuna.tsinghua.edu.cn@g; s@//security.ubuntu.com@//mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list.d/* 2>/dev/null || true; \
    apt-get update && apt-get install -y --no-install-recommends \
      software-properties-common ca-certificates git curl ninja-build ccache \
      build-essential cmake && \
    add-apt-repository -y ppa:deadsnakes/ppa && apt-get update && \
    apt-get install -y --no-install-recommends \
      python${PY} python${PY}-dev python${PY}-venv && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PY} 1 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python${PY}

ENV CUDA_HOME=/usr/local/cuda
ENV RUSTUP_HOME=/usr/local/rustup CARGO_HOME=/usr/local/cargo
ENV PATH=/usr/local/cargo/bin:/usr/local/cuda/bin:/usr/lib/ccache:${PATH}
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV MAX_JOBS=24 CCACHE_DIR=/root/.cache/ccache CMAKE_BUILD_TYPE=Release

# Rust toolchain for vLLM's Rust frontend (vllm-rs), via rsproxy.cn (China mirror).
ENV RUSTUP_DIST_SERVER=https://rsproxy.cn RUSTUP_UPDATE_ROOT=https://rsproxy.cn/rustup
RUN curl --proto '=https' --tlsv1.2 -sSf https://rsproxy.cn/rustup-init.sh | sh -s -- -y \
      --default-toolchain stable --profile minimal && \
    printf '[source.crates-io]\nreplace-with = "rsproxy-sparse"\n[source.rsproxy-sparse]\nregistry = "sparse+https://rsproxy.cn/index/"\n[net]\ngit-fetch-with-cli = true\n' \
      > ${CARGO_HOME}/config.toml && \
    cargo --version

# Torch first (pinned, cu128) so the vLLM build links against the right ABI.
# Build against torch +cu128 so it matches the base image's nvcc 12.8 (the kernels
# compile cleanly, exactly like the validated host build whose version tag is
# "cu128"). The runtime stage then installs torch's default CUDA 13 build, which
# is what production actually runs on (built-with-12.8, runs-on-13 — same as host).
# Build requirements mirror vLLM's pyproject [build-system].requires because the
# wheel is built with --no-isolation (deps are NOT auto-installed).
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --index-url ${TORCH_INDEX} torch==${TORCH_VERSION} && \
    python3 -m pip install -i ${PIP_INDEX} \
      "cmake>=3.26.1" ninja "packaging>=24.2" \
      "setuptools>=77.0.3,<81.0.0" "setuptools-scm>=8.0" "setuptools-rust>=1.9.0" \
      wheel jinja2 numpy build

# Fetch the SM75 fork at the pinned ref.
WORKDIR /src
RUN git clone --depth 1 --branch ${VLLM_REF} ${VLLM_REPO} vllm
WORKDIR /src/vllm

# Build the wheel for sm_75 only. FA2/FlashMLA fetch+build are skipped by the
# sm75 CMake patches in the fork, which keeps this build fast.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/vllm/rust/target \
    VLLM_TARGET_DEVICE=cuda \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    python3 -m build --wheel --no-isolation -o /wheels && \
    ls -lh /wheels

# ---------------------------------------------------------------------------
# Stage 2: runtime — install wheel + deps + flashinfer (patched), keep nvcc.
# ---------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_TAG} AS runtime
ARG PY PIP_INDEX TORCH_INDEX TORCH_VERSION FLASHINFER_VERSION TORCH_CUDA_ARCH_LIST
ARG FLASHQLA_REPO FLASHQLA_REF
ENV DEBIAN_FRONTEND=noninteractive

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    sed -i 's@//.*archive.ubuntu.com@//mirrors.tuna.tsinghua.edu.cn@g; s@//security.ubuntu.com@//mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list.d/* 2>/dev/null || true; \
    apt-get update && apt-get install -y --no-install-recommends \
      software-properties-common ca-certificates git curl ninja-build \
      libnuma1 libibverbs1 && \
    add-apt-repository -y ppa:deadsnakes/ppa && apt-get update && \
    apt-get install -y --no-install-recommends \
      python${PY} python${PY}-dev python${PY}-venv && \
    rm -rf /var/lib/apt/lists/*

# Use an isolated venv so pip never collides with apt-managed packages in the
# shared /usr/lib/python3/dist-packages (e.g. python3-jwt without a pip RECORD).
RUN python${PY} -m venv /opt/venv
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/opt/venv/bin:/usr/local/cuda/bin:${PATH}
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -i ${PIP_INDEX} --upgrade pip setuptools wheel

# Torch (pinned, CUDA 13 build from default index) + flashinfer JIT runtime.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -i ${PIP_INDEX} \
      torch==${TORCH_VERSION} \
      flashinfer-python==${FLASHINFER_VERSION} flashinfer-cubin==${FLASHINFER_VERSION}

# Install the vLLM wheel (pulls remaining runtime deps from the mirror).
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,from=builder,source=/wheels,target=/wheels \
    pip install -i ${PIP_INDEX} /wheels/*.whl

# FlashQLA (flashqla_legacy GDN prefill backend for SM70/SM75). Pure-Python wheel
# whose gdn_forward.cu is JIT-compiled at runtime via torch.utils.cpp_extension
# (needs pybind11 + nvcc + ninja, all present). Installed editable so the bundled
# .cu source stays on disk for that runtime build.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -i ${PIP_INDEX} pybind11 && \
    git clone --depth 1 --branch ${FLASHQLA_REF} ${FLASHQLA_REPO} /opt/flash_qla && \
    pip install -i ${PIP_INDEX} -e /opt/flash_qla

# Apply the SM75 shared-memory (EBO) fix to flashinfer 0.6.12 headers so the
# head_dim=256 prefill kernels fit the 64 KiB opt-in smem cap on Turing.
COPY patches/flashinfer-0.6.12-sm75-ebo.patch /tmp/fi-sm75.patch
RUN SP=$(python3 -c "import site;print(site.getsitepackages()[0])") && \
    patch -p1 -d "$SP" < /tmp/fi-sm75.patch && \
    echo "flashinfer SM75 EBO patch applied under $SP"

# Stable serving environment (matches the validated bare-metal launcher).
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    OMP_NUM_THREADS=4 \
    HF_HUB_OFFLINE=1 \
    VLLM_MARLIN_USE_ATOMIC_ADD=1 \
    VLLM_USE_FLASHINFER_SAMPLER=0 \
    NCCL_CUMEM_ENABLE=0 \
    NCCL_CUMEM_HOST_ENABLE=0 \
    NCCL_IB_DISABLE=1 \
    NCCL_P2P_DISABLE=0 \
    HF_HOME=/root/.cache/huggingface

EXPOSE 8000
ENTRYPOINT ["/opt/venv/bin/python3", "-m", "vllm.entrypoints.openai.api_server"]
CMD ["--help"]
