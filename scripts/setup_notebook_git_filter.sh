#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
filter_cmd="python3 \"$repo_root/scripts/ipynb_clean_filter.py\""

git config filter.ipynb-clean.clean "$filter_cmd"
git config filter.ipynb-clean.smudge cat
git config filter.ipynb-clean.required true

echo "Configured git notebook clean filter:"
echo "  filter name: ipynb-clean"
echo "  clean command: $filter_cmd"
echo "  smudge command: cat"
echo "  required: true"
