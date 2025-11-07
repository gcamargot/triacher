#!/usr/bin/env bash
set -euo pipefail
# Fetch latest release of whisper.cpp and build with Metal
REPO="ggerganov/whisper.cpp"
DEST_DIR="whisper.cpp"
BUILD_WITH_CMAKE=${BUILD_WITH_CMAKE:-0}

if ! command -v curl >/dev/null; then
  echo "curl is required" >&2; exit 1
fi

LATEST_TAG=$(curl -s https://api.github.com/repos/${REPO}/releases/latest | sed -n 's/ *"tag_name": *"\(.*\)",/\1/p' | head -n1)
if [ -z "${LATEST_TAG}" ]; then
  echo "Could not determine latest release tag via GitHub API; falling back to cloning default branch" >&2
  if [ ! -d "${DEST_DIR}" ]; then
    git clone https://github.com/${REPO}.git "${DEST_DIR}"
  fi
else
  echo "Latest release: ${LATEST_TAG}"
  rm -rf "${DEST_DIR}"
  git clone --depth 1 --branch "${LATEST_TAG}" https://github.com/${REPO}.git "${DEST_DIR}"
fi

# Build with Metal
cd "${DEST_DIR}"
if [ "${BUILD_WITH_CMAKE}" = "1" ]; then
  cmake -S . -B build -DGGML_METAL=ON
  cmake --build build -j
  echo "Built. Binary likely at build/bin/whisper (or main)."
else
  make clean || true
  make GGML_METAL=1 -j"$(sysctl -n hw.ncpu)"
  echo "Built. Binary at ./main"
fi
