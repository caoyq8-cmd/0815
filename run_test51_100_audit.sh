#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Usage options:
#   bash run_test51_100_audit.sh
#   bash run_test51_100_audit.sh /absolute/path/to/USCT_download
#   USCT_REPO_ROOT=/absolute/path/to/USCT_download bash run_test51_100_audit.sh
if [[ $# -ge 1 ]]; then
  REPO_ROOT="$(cd "$1" && pwd)"
elif [[ -n "${USCT_REPO_ROOT:-}" ]]; then
  REPO_ROOT="$(cd "${USCT_REPO_ROOT}" && pwd)"
elif git -C "${SCRIPT_DIR}" rev-parse --show-toplevel >/dev/null 2>&1; then
  REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
else
  echo "ERROR: cannot identify the USCT repository root." >&2
  echo "Run: bash $0 /home/featurize/work/USCT_repro/USCT_download" >&2
  exit 2
fi

if [[ ! -f "${REPO_ROOT}/cbs_model.py" ]]; then
  echo "ERROR: resolved path does not look like USCT_download: ${REPO_ROOT}" >&2
  echo "Expected file: ${REPO_ROOT}/cbs_model.py" >&2
  exit 2
fi

echo "audit script = ${SCRIPT_DIR}/audit_test51_100_history.py"
echo "repo root    = ${REPO_ROOT}"
echo "output dir   = ${REPO_ROOT}/final_holdout_audit/test51_100"

python "${SCRIPT_DIR}/audit_test51_100_history.py" \
  --repo_root "${REPO_ROOT}" \
  --output_dir "./final_holdout_audit/test51_100" \
  --start 51 \
  --end 100 \
  --max_file_mb 20 \
  --scan_git_history
