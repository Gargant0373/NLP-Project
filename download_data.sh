#!/usr/bin/env zsh
set -euo pipefail

REPO_URL="https://github.com/clarin-eric/ParlaMint.git"
CLONE_DIR="ParlaMint"

if ! command -v git >/dev/null 2>&1; then
  echo "Error: git is not installed. Install git and re-run this script." >&2
  exit 1
fi

if [ -d "$CLONE_DIR" ]; then
  echo "Directory $CLONE_DIR already exists. Remove it to re-clone or choose a different destination." >&2
  exit 1
fi

echo "Cloning $REPO_URL into $CLONE_DIR"
git clone "$REPO_URL" "$CLONE_DIR"

echo "Done. ParlaMint is available at: $CLONE_DIR"
