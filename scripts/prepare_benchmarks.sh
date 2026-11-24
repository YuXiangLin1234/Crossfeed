#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" exec python -m crossfeed.evaluation.prepare "$@"
