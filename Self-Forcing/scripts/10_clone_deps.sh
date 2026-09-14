#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${ROOT_DIR}/third_party"

# Self-Forcing itself is vendored in third_party/Self-Forcing (see UPSTREAM.md);
# only the evaluation repositories are cloned here.

if [[ ! -d "${ROOT_DIR}/third_party/VBench/.git" ]]; then
  git clone https://github.com/Vchitect/VBench.git "${ROOT_DIR}/third_party/VBench"
fi

if [[ ! -d "${ROOT_DIR}/third_party/StoryEval/.git" ]]; then
  git clone https://github.com/ypwang61/StoryEval.git "${ROOT_DIR}/third_party/StoryEval"
fi

echo "Evaluation dependencies cloned in ${ROOT_DIR}/third_party"
