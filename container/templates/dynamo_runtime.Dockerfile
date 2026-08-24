{#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#}
# === BEGIN templates/dynamo_runtime.Dockerfile ===
#######################################
########## Runtime image ##############
#######################################

FROM dynamo_base AS runtime

ARG PYTHON_VERSION
ARG CUDA_MAJOR

# Create dynamo user with group 0 for OpenShift compatibility
RUN userdel -r ubuntu > /dev/null 2>&1 || true \
    && useradd -m -s /bin/bash -g 0 dynamo \
    && [ `id -u dynamo` -eq 1000 ] \
    && mkdir -p /home/dynamo/.cache /opt/dynamo \
    # Non-recursive chown - only the directories themselves, not contents
    && chown dynamo:0 /home/dynamo /home/dynamo/.cache /opt/dynamo /workspace \
    # No chmod needed: umask 002 handles new files, COPY --chmod handles copied content
    # Set umask globally for all subsequent RUN commands (must be done as root before USER dynamo)
    # NOTE: Setting ENV UMASK=002 does NOT work - umask is a shell builtin, not an environment variable
    && mkdir -p /etc/profile.d && echo 'umask 002' > /etc/profile.d/00-umask.sh

# NIXL environment variables
ENV NIXL_PREFIX=/opt/nvidia/nvda_nixl \
    NIXL_LIB_DIR=/opt/nvidia/nvda_nixl/lib64 \
    NIXL_PLUGIN_DIR=/opt/nvidia/nvda_nixl/lib64/plugins \
    CARGO_TARGET_DIR=/opt/dynamo/target

ENV LD_LIBRARY_PATH=\
${NIXL_LIB_DIR}:\
${NIXL_PLUGIN_DIR}:\
/usr/local/ucx/lib:\
/usr/local/ucx/lib/ucx:\
${LD_LIBRARY_PATH}

# ===========================================================================
# Source-INDEPENDENT runtime layers
# ===========================================================================
# Everything below up to the "wheel_builder-derived layers" banner depends only
# on the base image + requirements files — NOT on the dynamo source tree (lib/,
# components/), the freshly-built wheels, or ANY wheel_builder snapshot content.
# That last part matters: a COPY --from / bind mount of wheel_builder keys its
# cache on the mounted content's hash, and wheel_builder re-runs on every
# source change — one unstable byte anywhere in the mounted tree re-executes
# the layer and everything after it (observed: the broad /usr/local bind mount
# broke the chain and dragged the apt + pip installs with it). So: environment
# setup first, wheel_builder-derived layers strictly after.

# Install Python for framework=none runtime (cuda-dl-base doesn't include Python)
# This is needed to create venv and install dynamo packages
ARG PYTHON_VERSION
# Cache apt downloads; sharing=locked avoids apt/dpkg races with concurrent builds.
# Clear partial downloads first to avoid stale rename failures from prior interrupted builds.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    rm -rf /var/cache/apt/archives/partial/* && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv \
        build-essential \
        cmake \
        protobuf-compiler \
        pkg-config \
        clang \
        libclang-dev \
        patchelf \
        git \
        git-lfs && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3

# Switch to dynamo user and create virtual environment
USER dynamo
ENV HOME=/home/dynamo

# Create and activate virtual environment
# Use login shell to pick up umask 002 from /etc/profile.d/00-umask.sh for group-writable files
SHELL ["/bin/bash", "-l", "-o", "pipefail", "-c"]
# Cache uv downloads; uv handles its own locking for the cache.
RUN --mount=type=cache,target=/home/dynamo/.cache/uv,uid=1000,gid=0,mode=0775,sharing=shared \
    export UV_CACHE_DIR=/home/dynamo/.cache/uv && \
    uv venv /opt/dynamo/venv --python ${PYTHON_VERSION}

ENV VIRTUAL_ENV=/opt/dynamo/venv \
    PATH="/opt/dynamo/venv/bin:${PATH}"

# Initialize Git LFS (required for git+https dependencies with LFS artifacts)
RUN git lfs install

# Install runtime dependencies (common + planner + frontend) for ALL targets.
# Driven entirely by the requirements files, so this expensive network layer sits
# in the source-independent section and stays cached across source-only changes.
# Frontend deps (tritonclient + grpcio/protobuf pins) resolve here in one pass;
# the wheel install below re-applies these files as --constraint so installing
# the dynamo wheels cannot drift the pins (the old wheels-then-requirements
# ordering existed to prevent exactly that grpcio downgrade).
RUN --mount=type=bind,source=./container/deps/requirements.common.txt,target=/tmp/requirements.common.txt \
    --mount=type=bind,source=./container/deps/requirements.planner.txt,target=/tmp/requirements.planner.txt \
    --mount=type=bind,source=./container/deps/requirements.frontend.txt,target=/tmp/requirements.frontend.txt \
    --mount=type=cache,target=/home/dynamo/.cache/uv,uid=1000,gid=0,mode=0775,sharing=shared \
    export UV_CACHE_DIR=/home/dynamo/.cache/uv UV_GIT_LFS=1 UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=5 && \
    uv pip install \
        --index-strategy unsafe-best-match \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        --requirement /tmp/requirements.common.txt \
        --requirement /tmp/requirements.planner.txt \
        --requirement /tmp/requirements.frontend.txt

# ===========================================================================
# wheel_builder-derived layers (re-validated whenever wheel_builder re-runs)
# ===========================================================================
# ucx / NIXL / ffmpeg content is byte-stable across source changes, so these
# COPYs normally cache — but keep them BELOW all environment layers so a cache
# miss here can never re-trigger apt/pip work.

# Copy ucx and nixl libs
COPY --chown=dynamo: --from=wheel_builder /usr/local/ucx/ /usr/local/ucx/
COPY --chown=dynamo: --from=wheel_builder ${NIXL_PREFIX}/ ${NIXL_PREFIX}/
COPY --chown=dynamo: --from=wheel_builder /opt/dynamo/dist/nixl/ /opt/dynamo/wheelhouse/nixl/
COPY --chown=dynamo: --from=wheel_builder /workspace/nixl/build/src/bindings/python/nixl-meta/nixl-*.whl /opt/dynamo/wheelhouse/nixl/

# Always copy FFmpeg so libs are available for Rust checks in CI.
# libvpx.so* is included because the in-tree ffmpeg is built with --enable-libvpx,
# so libavcodec.so has a runtime dependency on libvpx.so.9.
# Bind-mount the specific subtrees rather than all of /usr/local: the broad
# mount hashed unstable wheel_builder state (cargo home etc.) into this layer's
# cache key and re-ran it — and everything after it — on every source change.
# Needs root (writes /usr/local, runs ldconfig); this stage switched to the
# dynamo user for the venv layers above.
USER root
RUN --mount=type=bind,from=wheel_builder,source=/usr/local/include,target=/tmp/usr/local/include \
    --mount=type=bind,from=wheel_builder,source=/usr/local/lib,target=/tmp/usr/local/lib \
    --mount=type=bind,from=wheel_builder,source=/usr/local/src/ffmpeg,target=/tmp/usr/local/src/ffmpeg \
    mkdir -p /usr/local/lib/pkgconfig && \
    cp -rnL /tmp/usr/local/include/libav* /tmp/usr/local/include/libsw* /usr/local/include/ && \
    cp -nL /tmp/usr/local/lib/libav*.so /tmp/usr/local/lib/libsw*.so /usr/local/lib/ && \
    cp -nL /tmp/usr/local/lib/lib*vpx*.so* /usr/local/lib/ 2>/dev/null || true && \
    cp -nL /tmp/usr/local/lib/pkgconfig/libav*.pc /tmp/usr/local/lib/pkgconfig/libsw*.pc /usr/local/lib/pkgconfig/ && \
    cp -r /tmp/usr/local/src/ffmpeg /usr/local/src/ && \
    ldconfig
USER dynamo

# Install the NIXL wheels (all targets). The nixl wheelhouse copied above is
# independent of dynamo source, so this layer stays cached when only
# Rust/Python source changes. That caching carries real weight: nixl-cu12
# pulls in torch and the full nvidia-* CUDA library stack (~6 GB), which
# previously resolved inside the source-dependent dynamo-wheel install below —
# re-installing and re-pushing a multi-GB layer on every build. Constraints and
# indexes match the dynamo-wheel install so nothing here can drift its pins.
RUN --mount=type=bind,source=./container/deps/requirements.common.txt,target=/tmp/requirements.common.txt \
    --mount=type=bind,source=./container/deps/requirements.planner.txt,target=/tmp/requirements.planner.txt \
    --mount=type=bind,source=./container/deps/requirements.frontend.txt,target=/tmp/requirements.frontend.txt \
    --mount=type=cache,target=/home/dynamo/.cache/uv,uid=1000,gid=0,mode=0775,sharing=shared \
    export UV_CACHE_DIR=/home/dynamo/.cache/uv && \
    uv pip install \
        --index-strategy unsafe-best-match \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        --constraint /tmp/requirements.common.txt \
        --constraint /tmp/requirements.planner.txt \
        --constraint /tmp/requirements.frontend.txt \
        /opt/dynamo/wheelhouse/nixl/nixl-*-py3-none-any.whl \
        /opt/dynamo/wheelhouse/nixl/nixl_cu${CUDA_MAJOR}-*.whl

# ===========================================================================
# Source-DEPENDENT layers (rebuilt when dynamo source / wheels change)
# ===========================================================================
# The cargo target dir is NOT copied here: it lives in a --mount=type=cache in
# wheel_builder (absent from the stage filesystem), and the runtime image
# consumes the built wheels below — build intermediates would be dead weight.
COPY --chown=dynamo: --from=wheel_builder /opt/dynamo/dist/*.whl /opt/dynamo/wheelhouse/

{% if target not in ("dev", "local-dev") %}
# Install dynamo wheels (runtime packages only, no test dependencies).
# The requirements files ride along as --constraint so this install cannot
# up/downgrade anything the requirements layer pinned (grpcio/protobuf etc.).
# The nixl wheels (and their torch/CUDA dependency stack) are already installed
# in the cached layer above and deliberately NOT re-listed here — re-installing
# local wheels would duplicate their content into this source-dependent layer.
# uv handles its own locking for the cache, no need to add sharing=locked
ARG ENABLE_KVBM
RUN --mount=type=bind,source=./container/deps/requirements.common.txt,target=/tmp/requirements.common.txt \
    --mount=type=bind,source=./container/deps/requirements.planner.txt,target=/tmp/requirements.planner.txt \
    --mount=type=bind,source=./container/deps/requirements.frontend.txt,target=/tmp/requirements.frontend.txt \
    --mount=type=cache,target=/home/dynamo/.cache/uv,uid=1000,gid=0,mode=0775,sharing=shared \
    export UV_CACHE_DIR=/home/dynamo/.cache/uv && \
    uv pip install \
    --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    --constraint /tmp/requirements.common.txt \
    --constraint /tmp/requirements.planner.txt \
    --constraint /tmp/requirements.frontend.txt \
    /opt/dynamo/wheelhouse/ai_dynamo_runtime*.whl \
    /opt/dynamo/wheelhouse/ai_dynamo*any.whl && \
    if [ "$ENABLE_KVBM" = "true" ]; then \
        KVBM_WHEEL=$(ls /opt/dynamo/wheelhouse/kvbm*.whl 2>/dev/null | head -1); \
        if [ -z "$KVBM_WHEEL" ]; then \
            echo "ERROR: ENABLE_KVBM is true but no KVBM wheel found in wheelhouse" >&2; \
            exit 1; \
        fi; \
        uv pip install \
            --constraint /tmp/requirements.common.txt \
            --constraint /tmp/requirements.planner.txt \
            --constraint /tmp/requirements.frontend.txt \
            "$KVBM_WHEEL"; \
    fi
{% endif %}

# Install gpu_memory_service wheel if enabled (all targets). Guarded so it is a
# no-op when the wheel is absent (e.g. dev/local-dev, which keep the wheel in the
# wheelhouse for downstream COPY --from consumers but do not install it).
ARG ENABLE_GPU_MEMORY_SERVICE
RUN --mount=type=cache,target=/home/dynamo/.cache/uv,uid=1000,gid=0,mode=0775,sharing=shared \
    if [ "${ENABLE_GPU_MEMORY_SERVICE}" = "true" ]; then \
        export UV_CACHE_DIR=/home/dynamo/.cache/uv && \
        GMS_WHEEL=$(ls /opt/dynamo/wheelhouse/gpu_memory_service*.whl 2>/dev/null | head -1); \
        if [ -n "$GMS_WHEEL" ]; then uv pip install "$GMS_WHEEL"; fi; \
    fi

# TODO: skip /workspace COPY for dev/local-dev (bind-mounted from host, gets shadowed)
# Copy workspace source code
ARG WORKSPACE_DIR=/workspace
WORKDIR ${WORKSPACE_DIR}
COPY --chmod=775 --chown=dynamo:0 ./ ${WORKSPACE_DIR}/

# Baseten version stamp — produced by tools/version-stamp.sh and passed in by
# container/build.sh. Downstream images (e.g. mp/baseten_dynamo harness) copy
# this file forward via `COPY --from=<dynamo image> /etc/baseten/version/dynamo`.
# Placed LAST on purpose: version-stamp.sh records a fresh `built_at` timestamp on
# every build, so this file's content changes every run. Keeping the COPY at the end
# means that churn only invalidates this final tiny layer instead of busting the
# expensive apt/pip/wheel layers above it (which is what kills warm rebuilds).
ARG BASETEN_VERSION_FILE
COPY --chown=dynamo: ${BASETEN_VERSION_FILE} /etc/baseten/version/dynamo

ARG DYNAMO_COMMIT_SHA
ENV DYNAMO_COMMIT_SHA=$DYNAMO_COMMIT_SHA

ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]
CMD []
