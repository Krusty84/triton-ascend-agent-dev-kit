#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}" && pwd)"
DEFAULT_OUTPUT_DIR="${PROJECT_ROOT}/dist"
ARCHIVE_PREFIX="triton-ascend-agent-dev-kit"

usage() {
    echo "Usage: $0 [--dry-run] [OUTPUT_DIR]"
}

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    shift
fi

if [[ $# -gt 1 ]]; then
    usage >&2
    exit 2
fi

OUTPUT_DIR="${1:-${DEFAULT_OUTPUT_DIR}}"
if [[ "${OUTPUT_DIR}" != /* ]]; then
    OUTPUT_DIR="$(pwd)/${OUTPUT_DIR}"
fi

for required in AGENTS.md LICENSE triton-ascend-dev-doc.md triton-ascend-dev-doc skills; do
    if [[ ! -e "${PROJECT_ROOT}/${required}" ]]; then
        echo "Required source is missing: ${PROJECT_ROOT}/${required}" >&2
        exit 1
    fi
done

STAGING_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/triton-ascend-agent-kit.XXXXXX")"
cleanup() {
    rm -rf "${STAGING_ROOT}"
}
trap cleanup EXIT

copy_common_files() {
    local target_dir="$1"

    cp "${PROJECT_ROOT}/LICENSE" \
        "${target_dir}/LICENSE"
    cp "${PROJECT_ROOT}/triton-ascend-dev-doc.md" \
        "${target_dir}/triton-ascend-dev-doc.md"
    cp -R "${PROJECT_ROOT}/triton-ascend-dev-doc" \
        "${target_dir}/triton-ascend-dev-doc"
}

CODEX_DIR="${STAGING_ROOT}/codex"
OPENCODE_DIR="${STAGING_ROOT}/opencode"
CLAUDE_DIR="${STAGING_ROOT}/claude"

mkdir -p \
    "${CODEX_DIR}/.agents/skills" \
    "${OPENCODE_DIR}/.opencode/skills" \
    "${CLAUDE_DIR}/.claude/skills"

for target_dir in "${CODEX_DIR}" "${OPENCODE_DIR}" "${CLAUDE_DIR}"; do
    copy_common_files "${target_dir}"
done

cp "${PROJECT_ROOT}/AGENTS.md" "${CODEX_DIR}/AGENTS.md"
cp -R "${PROJECT_ROOT}/skills/." "${CODEX_DIR}/.agents/skills/"

cp "${PROJECT_ROOT}/AGENTS.md" "${OPENCODE_DIR}/AGENTS.md"
cp -R "${PROJECT_ROOT}/skills/." "${OPENCODE_DIR}/.opencode/skills/"

cp "${PROJECT_ROOT}/AGENTS.md" "${CLAUDE_DIR}/CLAUDE.md"
cp -R "${PROJECT_ROOT}/skills/." "${CLAUDE_DIR}/.claude/skills/"

if [[ "${DRY_RUN}" == true ]]; then
    for agent in codex opencode claude; do
        echo "[${agent}]"
        find "${STAGING_ROOT}/${agent}" -type f -print \
            | sed "s|^${STAGING_ROOT}/${agent}/||" \
            | LC_ALL=C sort
    done
    exit 0
fi

if ! command -v zip >/dev/null 2>&1; then
    echo "The 'zip' command is required." >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

create_archive() {
    local source_dir="$1"
    local output_file="$2"

    rm -f "${output_file}"
    (
        cd "${source_dir}"
        COPYFILE_DISABLE=1 zip -X -q -r "${output_file}" .
    )
    echo "Created ${output_file}"
}

create_archive \
    "${CODEX_DIR}" \
    "${OUTPUT_DIR}/${ARCHIVE_PREFIX}-codex.zip"
create_archive \
    "${OPENCODE_DIR}" \
    "${OUTPUT_DIR}/${ARCHIVE_PREFIX}-opencode.zip"
create_archive \
    "${CLAUDE_DIR}" \
    "${OUTPUT_DIR}/${ARCHIVE_PREFIX}-claude.zip"
