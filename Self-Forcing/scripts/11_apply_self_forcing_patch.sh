#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SF_DIR="${ROOT_DIR}/third_party/Self-Forcing"
PATCH_FILE="${ROOT_DIR}/docs/patches/self_forcing_kv_quant.patch"
# 旧版 patch 从本仓库 git 历史中取出，用于把已打过旧 patch 的 Self-Forcing 升级到当前版本：
#   036209f  active-prefix 整段重量化
#   3ba1d26  初版整段重量化
LEGACY_PATCH_REVS=(036209f 3ba1d26)

if [[ ! -d "${SF_DIR}/.git" ]]; then
  echo "Self-Forcing repo not found at ${SF_DIR}. Run scripts/10_clone_deps.sh first."
  exit 1
fi

if [[ ! -f "${PATCH_FILE}" ]]; then
  echo "Patch file not found: ${PATCH_FILE}"
  exit 1
fi

cd "${SF_DIR}"
if git apply --check "${PATCH_FILE}" >/dev/null 2>&1; then
  git apply "${PATCH_FILE}"
  echo "Applied KV quantization hook patch to Self-Forcing causal_model.py"
  exit 0
fi
if git apply --reverse --check "${PATCH_FILE}" >/dev/null 2>&1; then
  echo "Patch already applied in Self-Forcing causal_model.py"
  exit 0
fi

LEGACY_PATCH="$(mktemp)"
trap 'rm -f "${LEGACY_PATCH}"' EXIT
for rev in "${LEGACY_PATCH_REVS[@]}"; do
  git -C "${ROOT_DIR}" show "${rev}:./docs/patches/self_forcing_kv_quant.patch" > "${LEGACY_PATCH}" 2>/dev/null || continue
  if git apply --reverse --check "${LEGACY_PATCH}" >/dev/null 2>&1; then
    git apply --reverse "${LEGACY_PATCH}"
    git apply "${PATCH_FILE}"
    echo "Upgraded KV quantization hook from the ${rev} patch to the current patch"
    exit 0
  fi
done

echo "Patch cannot be applied cleanly; please verify third_party/Self-Forcing state."
exit 1
