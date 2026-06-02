#!/usr/bin/env bash
# Compute a build-time version stamp for a Baseten-fork component and write
# it to a key=value file consumed by version_metrics.py at runtime.
#
# Output format (one key per line):
#   major=<int>
#   minor=<int>
#   patch=<commit count from base ref to HEAD, optionally path-filtered>
#   sha=<short HEAD sha>
#   base=<base ref string>
#   built_at=<ISO 8601 UTC>
#
# Usage:
#   version-stamp.sh --base-ref v1.0.0 --out path/to/file
#   version-stamp.sh --base-ref <sha> --major 1 --minor 0 --path mp/x --out file
#
# When --major/--minor are omitted, the script parses them from a tag of the
# form vMAJOR.MINOR[.PATCH...]. When --path is given, the patch count is
# limited to commits touching that path.

set -euo pipefail

BASE_REF=""
MAJOR=""
MINOR=""
PATH_FILTER=""
OUT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base-ref) BASE_REF="$2"; shift 2 ;;
        --major)    MAJOR="$2";    shift 2 ;;
        --minor)    MINOR="$2";    shift 2 ;;
        --path)     PATH_FILTER="$2"; shift 2 ;;
        --out)      OUT="$2";      shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

[[ -n "$BASE_REF" ]] || { echo "--base-ref is required" >&2; exit 2; }
[[ -n "$OUT"      ]] || { echo "--out is required"      >&2; exit 2; }

if [[ -z "$MAJOR" || -z "$MINOR" ]]; then
    if [[ "$BASE_REF" =~ ^v?([0-9]+)\.([0-9]+) ]]; then
        MAJOR="${MAJOR:-${BASH_REMATCH[1]}}"
        MINOR="${MINOR:-${BASH_REMATCH[2]}}"
    else
        echo "--major/--minor required when --base-ref is not a vMAJOR.MINOR tag (got: $BASE_REF)" >&2
        exit 2
    fi
fi

if ! git rev-parse --verify --quiet "$BASE_REF^{commit}" >/dev/null; then
    echo "base ref not found in repo: $BASE_REF" >&2
    exit 1
fi

if [[ -n "$(git status --porcelain)" && "${BASETEN_ALLOW_DIRTY:-0}" != "1" ]]; then
    echo "refusing to stamp: working tree is dirty (set BASETEN_ALLOW_DIRTY=1 to override)" >&2
    exit 1
fi

if [[ -n "$PATH_FILTER" ]]; then
    PATCH=$(git rev-list --count "${BASE_REF}..HEAD" -- "$PATH_FILTER")
else
    PATCH=$(git rev-list --count "${BASE_REF}..HEAD")
fi

SHA=$(git rev-parse --short HEAD)
BUILT_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)

mkdir -p "$(dirname "$OUT")"
cat > "$OUT" <<EOF
major=${MAJOR}
minor=${MINOR}
patch=${PATCH}
sha=${SHA}
base=${BASE_REF}
built_at=${BUILT_AT}
EOF

echo "stamped $OUT: ${MAJOR}.${MINOR}.${PATCH} (${SHA})"
