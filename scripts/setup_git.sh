#!/usr/bin/env bash
# One-time git setup per clone: notebook output stripping (nbstripout) and Git LFS.
set -euo pipefail
cd "$(dirname "$0")/.."

# Strip notebook outputs on commit. `uv run` keeps it working from any checkout or worktree.
git config filter.nbstripout.clean "uv run --quiet nbstripout"
git config filter.nbstripout.smudge cat
git config filter.nbstripout.required true
git config diff.ipynb.textconv "uv run --quiet nbstripout -t"
echo "nbstripout filter configured"

if git lfs version >/dev/null 2>&1; then
  git lfs install --local
else
  echo "git-lfs not installed: install it (Fedora: sudo dnf install git-lfs) and re-run this script" >&2
  exit 1
fi
