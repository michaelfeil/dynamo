#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Wrapper around render.py + docker buildx build that recovers the
# build.sh interface from v0.7.0–v0.9.0.  Delegates Dockerfile generation
# to the upstream render.py template system while keeping the familiar
# CLI, versioning, tagging, and logging behaviour.

if [ "${BASH_VERSINFO[0]}" -lt 4 ]; then
    echo "Error: Bash version 4.0 or higher is required. Current version: ${BASH_VERSINFO[0]}.${BASH_VERSINFO[1]}"
    exit 1
fi

set -e

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SOURCE_DIR=$(dirname "$(readlink -f "$0")")
BUILD_CONTEXT=$(dirname "$(readlink -f "$SOURCE_DIR")")

# ---------------------------------------------------------------------------
# Version detection (same logic as v0.9.0 build.sh)
# ---------------------------------------------------------------------------
commit_id=${commit_id:-$(git rev-parse --short HEAD)}
current_tag=${current_tag:-$(git describe --tags --exact-match 2>/dev/null | sed 's/^v//' || true)}

latest_release_branch=$(git branch -r 2>/dev/null | grep -E 'origin/release/[0-9]+\.[0-9]+\.[0-9]+$' | sed 's|.*/||' | sort -V | tail -1 || true)
if [[ -n ${latest_release_branch} ]]; then
    latest_tag=${latest_tag:-$latest_release_branch}
    echo "INFO: Using version from latest release branch: ${latest_tag}"
else
    latest_tag=${latest_tag:-$(git tag --merged HEAD --sort=-version:refname | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | head -1 | sed 's/^v//' || true)}
fi
if [[ -z ${latest_tag} ]]; then
    latest_tag="0.0.1"
    echo "No git release tag or branch found, setting to unknown version: ${latest_tag}"
fi

VERSION=v${current_tag:-$latest_tag.dev.$commit_id}

# ---------------------------------------------------------------------------
# Framework mapping: old names → render.py names
# ---------------------------------------------------------------------------
declare -A FRAMEWORK_MAP=(
    ["VLLM"]="vllm"
    ["TRTLLM"]="trtllm"
    ["SGLANG"]="sglang"
    ["NONE"]="dynamo"
    ["DYNAMO"]="dynamo"
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
TAG=""
RUN_PREFIX=""
PLATFORM="linux/amd64"
DEFAULT_FRAMEWORK="VLLM"
FRAMEWORK=""
TARGET=""
CUDA_VERSION=""
NO_CACHE=""
NO_LOAD=""
PUSH=""
BUILD_ARGS=""
CACHE_FROM=""
CACHE_TO=""
MAKE_EFA=""
NO_TAG_LATEST=""
DRY_RUN=""
CUSTOM_UID=""
CUSTOM_GID=""

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
show_help() {
    echo "usage: build.sh"
    echo "  [--framework {VLLM|TRTLLM|SGLANG|NONE}]  Inference framework (default: VLLM)"
    echo "  [--target {runtime|dev|local-dev|frontend}] Build target (default: dev)"
    echo "  [--platform PLATFORM]                       Docker platform (default: linux/amd64)"
    echo "  [--cuda-version VERSION]                    CUDA version: 12.9, 13.0, 13.1"
    echo "  [--tag TAG]                                 Custom image tag"
    echo "  [--build-arg KEY=VALUE]                     Extra docker build args"
    echo "  [--cache-from SOURCE]                       Docker cache source"
    echo "  [--cache-to DEST]                           Docker cache destination"
    echo "  [--no-cache]                                Disable Docker build cache"
    echo "  [--no-load]                                 Do not load image into docker"
    echo "  [--push]                                    Push image to registry"
    echo "  [--dry-run]                                 Print commands without running"
    echo "  [--make-efa]                                Enable AWS EFA support"
    echo "  [--uid UID]                                 User ID for local-dev (only with --target local-dev)"
    echo "  [--gid GID]                                 Group ID for local-dev (only with --target local-dev)"
    echo "  [--no-tag-latest]                           Do not add latest-{framework} tag"
    echo "  [--release-build]                           Perform a release build"
    echo ""
    echo "  Flags preserved for CLI compatibility (values come from context.yaml):"
    echo "  [--enable-kvbm]  [--enable-gpu-memory-service]  [--enable-media-nixl]"
    echo "  [--enable-media-ffmpeg]  [--use-sccache]  [--sccache-bucket B]"
    echo "  [--sccache-region R]  [--vllm-max-jobs N]  [--efa-version V]"
    echo "  [--nixl-ref REF]  [--base-image IMG]  [--base-image-tag TAG]"
    exit 0
}

missing_requirement() { echo "ERROR: $1 requires an argument." >&2; exit 1; }
error() { printf '%s %s\n' "$1" "$2" >&2; exit 1; }

# sccache
USE_SCCACHE=""
SCCACHE_BUCKET=""
SCCACHE_REGION=""

while :; do
    case ${1:-} in
    -h|-\?|--help)       show_help ;;
    --platform)          [ "$2" ] && PLATFORM=$2 && shift || missing_requirement "$1" ;;
    --framework)         [ "$2" ] && FRAMEWORK=$2 && shift || missing_requirement "$1" ;;
    --cuda-version)      [ "$2" ] && CUDA_VERSION=$2 && shift || missing_requirement "$1" ;;
    --target)            [ "$2" ] && TARGET=$2 && shift || missing_requirement "$1" ;;
    --tag)               [ "$2" ] && TAG="$2" && shift || missing_requirement "$1" ;;
    --build-arg)         [ "$2" ] && BUILD_ARGS+=" --build-arg $2" && shift || missing_requirement "$1" ;;
    --cache-from)        [ "$2" ] && CACHE_FROM+=" --cache-from $2" && shift || missing_requirement "$1" ;;
    --cache-to)          [ "$2" ] && CACHE_TO+=" --cache-to $2" && shift || missing_requirement "$1" ;;
    --uid)               [ "$2" ] && CUSTOM_UID=$2 && shift || missing_requirement "$1" ;;
    --gid)               [ "$2" ] && CUSTOM_GID=$2 && shift || missing_requirement "$1" ;;
    --no-cache)          NO_CACHE=" --no-cache" ;;
    --no-load)           NO_LOAD=true ;;
    --push)              PUSH=" --push" ;;
    --dry-run)           RUN_PREFIX="echo"; DRY_RUN=true
                         echo ""; echo "=============================="; echo "DRY RUN: COMMANDS PRINTED ONLY"; echo "=============================="; echo "" ;;
    --make-efa)          MAKE_EFA=true ;;
    --no-tag-latest)     NO_TAG_LATEST=true ;;
    --release-build)     ;; # accepted for compat, no-op (render.py handles)
    --enable-kvbm)       BUILD_ARGS+=" --build-arg ENABLE_KVBM=true" ;;
    --enable-gpu-memory-service) BUILD_ARGS+=" --build-arg ENABLE_GPU_MEMORY_SERVICE=true" ;;
    --enable-media-nixl) BUILD_ARGS+=" --build-arg ENABLE_MEDIA_NIXL=true" ;;
    --enable-media-ffmpeg) BUILD_ARGS+=" --build-arg ENABLE_MEDIA_FFMPEG=true" ;;
    --use-sccache)       USE_SCCACHE=true ;;
    --sccache-bucket)    [ "$2" ] && SCCACHE_BUCKET=$2 && shift || missing_requirement "$1" ;;
    --sccache-region)    [ "$2" ] && SCCACHE_REGION=$2 && shift || missing_requirement "$1" ;;
    --vllm-max-jobs)     [ "$2" ] && BUILD_ARGS+=" --build-arg MAX_JOBS=$2" && shift || missing_requirement "$1" ;;
    --efa-version)       [ "$2" ] && BUILD_ARGS+=" --build-arg EFA_VERSION=$2" && shift || missing_requirement "$1" ;;
    --nixl-ref)          [ "$2" ] && BUILD_ARGS+=" --build-arg NIXL_REF=$2" && shift || missing_requirement "$1" ;;
    --base-image)        [ "$2" ] && BUILD_ARGS+=" --build-arg BASE_IMAGE=$2" && shift || missing_requirement "$1" ;;
    --base-image-tag)    [ "$2" ] && BUILD_ARGS+=" --build-arg BASE_IMAGE_TAG=$2" && shift || missing_requirement "$1" ;;
    # Silently accept TRT-LLM flags for CLI compat (context.yaml handles defaults)
    --tensorrtllm-*)     shift ;; # consume the value
    -?*|?*)              error "ERROR: Unknown option:" "$1" ;;
    *)                   break ;;
    esac
    shift
done

# ---------------------------------------------------------------------------
# Resolve framework
# ---------------------------------------------------------------------------
if [ -z "$FRAMEWORK" ]; then
    FRAMEWORK=$DEFAULT_FRAMEWORK
fi
FRAMEWORK=${FRAMEWORK^^}

if [[ -z "${FRAMEWORK_MAP[$FRAMEWORK]:-}" ]]; then
    error "ERROR: Unknown framework:" "$FRAMEWORK"
fi
RENDER_FRAMEWORK="${FRAMEWORK_MAP[$FRAMEWORK]}"

# ---------------------------------------------------------------------------
# Resolve target
# ---------------------------------------------------------------------------
if [ -z "$TARGET" ]; then
    TARGET="dev"
fi

# Validate --uid/--gid
if [[ -n "${CUSTOM_UID:-}" || -n "${CUSTOM_GID:-}" ]]; then
    if [[ "$TARGET" != "local-dev" ]]; then
        error "ERROR:" "--uid and --gid can only be used with --target local-dev"
    fi
fi

# ---------------------------------------------------------------------------
# Resolve CUDA version for render.py
# ---------------------------------------------------------------------------
if [ -z "$CUDA_VERSION" ]; then
    if [[ "$RENDER_FRAMEWORK" == "trtllm" ]]; then
        CUDA_VERSION="13.1"
    else
        CUDA_VERSION="12.9"
    fi
fi

# ---------------------------------------------------------------------------
# Resolve platform / arch
# ---------------------------------------------------------------------------
ARCH="amd64"
RENDER_PLATFORM="amd64"
if [[ "$PLATFORM" == *"arm64"* ]]; then
    ARCH="arm64"
    RENDER_PLATFORM="arm64"
    BUILD_ARGS+=" --build-arg ARCH=arm64 --build-arg ARCH_ALT=aarch64"
fi

# ---------------------------------------------------------------------------
# Resolve tags
# ---------------------------------------------------------------------------
# B10: custom baseten image tag.
if [ -z "$TAG" ]; then
    TAG="baseten/dynamo-test:${VERSION}-${FRAMEWORK,,}"
    if [ "$TARGET" != "dev" ] && [ "$TARGET" != "local-dev" ]; then
        TAG="${TAG}-${TARGET}"
    fi
fi

LATEST_TAG=""
if [ -z "${NO_TAG_LATEST}" ]; then
    LATEST_TAG="baseten/dynamo-test:latest-${FRAMEWORK,,}"
    if [ "$TARGET" != "dev" ] && [ "$TARGET" != "local-dev" ]; then
        LATEST_TAG="${LATEST_TAG}-${TARGET}"
    fi
fi

TAG_ARGS="--tag ${TAG}"
if [ -n "$LATEST_TAG" ]; then
    TAG_ARGS+=" --tag ${LATEST_TAG}"
fi

# ---------------------------------------------------------------------------
# Commit SHA
# ---------------------------------------------------------------------------
DYNAMO_COMMIT_SHA=${DYNAMO_COMMIT_SHA:-$(git rev-parse HEAD)}
BUILD_ARGS+=" --build-arg DYNAMO_COMMIT_SHA=$DYNAMO_COMMIT_SHA"

# ---------------------------------------------------------------------------
# Baseten version stamp (major.minor from .version-base, patch = commits ahead)
# Always produces .dynamo-version.env in the build context so Dockerfile COPY
# is unconditional. Falls back to a sentinel "unstamped" record on failure.
# ---------------------------------------------------------------------------
BASETEN_VERSION_FILE_REL=".dynamo-version.env"
BASETEN_VERSION_FILE_ABS="$BUILD_CONTEXT/$BASETEN_VERSION_FILE_REL"
BASETEN_VERSION_BASE=$(tr -d '[:space:]' < "$BUILD_CONTEXT/.version-base" 2>/dev/null || true)
if [[ -n "$BASETEN_VERSION_BASE" ]] \
   && "$BUILD_CONTEXT/tools/version-stamp.sh" \
        --base-ref "$BASETEN_VERSION_BASE" \
        --out "$BASETEN_VERSION_FILE_ABS"; then
    :
else
    echo "WARN: dynamo version stamp unavailable; emitting sentinel" >&2
    cat > "$BASETEN_VERSION_FILE_ABS" <<EOF
major=0
minor=0
patch=-1
sha=unknown
base=unstamped
built_at=unknown
EOF
fi
BUILD_ARGS+=" --build-arg BASETEN_VERSION_FILE=$BASETEN_VERSION_FILE_REL"

# ---------------------------------------------------------------------------
# local-dev UID/GID
# ---------------------------------------------------------------------------
if [[ "$TARGET" == "local-dev" ]]; then
    CUSTOM_UID=${CUSTOM_UID:-$(id -u)}
    CUSTOM_GID=${CUSTOM_GID:-$(id -g)}
    BUILD_ARGS+=" --build-arg USER_UID=${CUSTOM_UID} --build-arg USER_GID=${CUSTOM_GID}"
fi

# ---------------------------------------------------------------------------
# sccache
# ---------------------------------------------------------------------------
if [ "$USE_SCCACHE" = true ]; then
    if [ -z "$SCCACHE_BUCKET" ]; then error "ERROR:" "--sccache-bucket is required when --use-sccache is specified"; fi
    if [ -z "$SCCACHE_REGION" ]; then error "ERROR:" "--sccache-region is required when --use-sccache is specified"; fi
    BUILD_ARGS+=" --build-arg USE_SCCACHE=true"
    BUILD_ARGS+=" --build-arg SCCACHE_BUCKET=${SCCACHE_BUCKET}"
    BUILD_ARGS+=" --build-arg SCCACHE_REGION=${SCCACHE_REGION}"
    BUILD_ARGS+=" --secret id=aws-key-id,env=AWS_ACCESS_KEY_ID"
    BUILD_ARGS+=" --secret id=aws-secret-id,env=AWS_SECRET_ACCESS_KEY"
fi

# ---------------------------------------------------------------------------
# Render the Dockerfile
# ---------------------------------------------------------------------------
RENDER_ARGS="--framework ${RENDER_FRAMEWORK} --target ${TARGET} --platform ${RENDER_PLATFORM} --cuda-version ${CUDA_VERSION} --output-short-filename"
if [ "$MAKE_EFA" = true ]; then
    RENDER_ARGS+=" --make-efa"
fi

echo ""
echo "Building Dynamo Image: '${TAG}'"
echo ""
echo "   Version: '${VERSION}'"
echo "   Framework: '${FRAMEWORK}' (render.py: '${RENDER_FRAMEWORK}')"
echo "   Target: '${TARGET}'"
echo "   Platform: '${PLATFORM}'"
echo "   CUDA: '${CUDA_VERSION}'"
echo "   Build Context: '${BUILD_CONTEXT}'"
echo "   Build Arguments: '${BUILD_ARGS}'"
if [ "$USE_SCCACHE" = true ]; then
    echo "   sccache: Enabled (bucket=${SCCACHE_BUCKET}, region=${SCCACHE_REGION})"
fi
echo ""

echo "Rendering Dockerfile..."
$RUN_PREFIX python3 "${SOURCE_DIR}/render.py" ${RENDER_ARGS}

DOCKERFILE="${SOURCE_DIR}/rendered.Dockerfile"
if [ "$DRY_RUN" != "true" ] && [ ! -f "$DOCKERFILE" ]; then
    error "ERROR:" "render.py did not produce ${DOCKERFILE}"
fi

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
BUILD_LOG_DIR="${BUILD_CONTEXT}/build-logs"
mkdir -p "${BUILD_LOG_DIR}"
SINGLE_BUILD_LOG="${BUILD_LOG_DIR}/single-stage-build.log"

LOAD_FLAG=""
if [ "$NO_LOAD" != "true" ] && [ -z "$PUSH" ]; then
    LOAD_FLAG=" --load"
fi

$RUN_PREFIX docker buildx build \
    --progress=plain${LOAD_FLAG}${PUSH} \
    -f "${DOCKERFILE}" \
    --target "${TARGET}" \
    --platform "${PLATFORM}" \
    ${BUILD_ARGS} \
    ${CACHE_FROM} \
    ${CACHE_TO} \
    ${TAG_ARGS} \
    ${NO_CACHE} \
    "${BUILD_CONTEXT}" 2>&1 | tee "${SINGLE_BUILD_LOG}"

BUILD_EXIT_CODE=${PIPESTATUS[0]}
if [ ${BUILD_EXIT_CODE} -ne 0 ]; then
    exit ${BUILD_EXIT_CODE}
fi

# ---------------------------------------------------------------------------
# Clean up rendered Dockerfile
# ---------------------------------------------------------------------------
rm -f "${DOCKERFILE}"

echo ""
echo "Successfully built: ${TAG}"
if [ -n "$LATEST_TAG" ]; then
    echo "Also tagged: ${LATEST_TAG}"
fi
# tagged for version resolution
