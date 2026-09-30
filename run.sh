#!/bin/sh
# Start the gateway (builds the receiver first if needed).  Options are passed through: --gain 34 --port 8023 --bind 0.0.0.0
set -e
cd "$(dirname "$0")"
[ -x a3rx/target/release/a3rx ] || (cd a3rx && cargo build --release)
exec uv run --project . python gateway/live.py "$@"
