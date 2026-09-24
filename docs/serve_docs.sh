#!/usr/bin/env bash
# Start the live-reload Sphinx server. Browser: http://127.0.0.1:8765
#
# Stop it with Ctrl+C. Use another port with: PORT=8766 bash serve_docs.sh
#
# Bootstraps a lightweight repo-local ".venv-docs" on first run -- no Isaac Sim,
# no PyTorch, no LeRobot needed to build the docs. About 210 MB total.
set -euo pipefail
VERSION="2.1.0"
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
VENV="../.venv-docs"
PORT="${PORT:-8765}"

# A server already on the port is the confusing case: the browser then shows
# somebody else's build, which looks like "my changes are not appearing".
if command -v ss >/dev/null && ss -ltn "sport = :${PORT}" | grep -q LISTEN; then
    echo "[serve_docs] Port ${PORT} is already serving something." >&2
    ss -ltnp "sport = :${PORT}" 2>/dev/null | tail -n +2 >&2
    echo >&2
    echo "[serve_docs] Stop that process first, or pick another port:" >&2
    echo "               PORT=8766 bash serve_docs.sh" >&2
    exit 1
fi

if [[ ! -f "${VENV}/bin/activate" ]]; then
    echo "[serve_docs] Bootstrapping ${VENV} (one-time)..."
    python3 -m venv "${VENV}"
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate"
    pip install --upgrade pip
    pip install \
        'sphinx>=8.0,<9' 'sphinx-book-theme>=1.1' 'myst-nb>=1.1' \
        'sphinx-autodoc-typehints>=2.4' 'sphinx-copybutton>=0.5' \
        'sphinx-design>=0.6' 'sphinxcontrib-mermaid>=1.0' sphinx-autobuild
else
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate"
fi

echo "[serve_docs] http://127.0.0.1:${PORT}  (Ctrl+C to stop)"
exec make livehtml PORT="${PORT}"
