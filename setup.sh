#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_EXEC="${PYTHON_BIN}"
else
  for candidate in python3.11 python3.10 python3; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      PYTHON_EXEC="${candidate}"
      break
    fi
  done
fi

if [[ -z "${PYTHON_EXEC:-}" ]]; then
  echo "❌ Could not locate python3.10+ interpreter. Install Python 3.11 or set PYTHON_BIN."
  exit 1
fi

PY_VERSION="$("${PYTHON_EXEC}" -c 'import platform; print(platform.python_version())')"
echo "🐍 Using ${PYTHON_EXEC} (Python ${PY_VERSION})"

if [[ "${PY_VERSION}" =~ ^3\.(1[3-9]|[4-9]) ]]; then
  echo "⚠️  Warning: pandas wheels are not available for Python ${PY_VERSION} yet."
  echo "    Install Python 3.11 (e.g., 'brew install python@3.11') and rerun, or set PYTHON_BIN."
  exit 1
fi

if [[ ! -d .venv ]]; then
  echo "🔧 Creating Python virtual environment (.venv)..."
  "${PYTHON_EXEC}" -m venv .venv
else
  echo "ℹ️  Re-using existing virtual environment (.venv)"
fi

source .venv/bin/activate

echo "📦 Installing Python requirements..."
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

cat <<'EOF'

✅ Environment ready!
Activate it later with:
  source .venv/bin/activate

Run the TechM pipeline with:
  cd local_run_with_sample_Data
  python techm_local_pipeline.py

EOF
