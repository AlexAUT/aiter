#!/usr/bin/env bash
# ATT (instruction-level thread trace) for mla_min.py on gfx1250.
# Captures one dispatch, zips the result, and prints an sftp get command.
#
#   ./mla_min_att_trace.sh <outdir> [extra mla_min args...]
#
# Examples:
#   ./mla_min_att_trace.sh /tmp/att_base
#   ./mla_min_att_trace.sh /tmp/att_best --pgpf --sched 5 --npf 2 --pvsplit n
#   MLA_MIN_CTAS=2 ./mla_min_att_trace.sh /tmp/att_cta2 --pgpf
#
# Env:
#   SFTP_HOST   ssh/sftp alias used in the printed download command (default: b0_triton_slow)
#   ZIP_PATH    destination zip (default: <triton-root>/mla_min_<args>_att_<YYYYMMDD>.zip)
set -euo pipefail

OUT=${1:?usage: mla_min_att_trace.sh <outdir> [mla_min args...]}
shift
# Resolve relative outdirs against the caller's cwd.
if [ "${OUT#/}" = "$OUT" ]; then
    OUT="$(pwd)/$OUT"
fi

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
TRITON_ROOT=$(cd "$SCRIPT_DIR/../../../.." && pwd)
SFTP_HOST=${SFTP_HOST:-b0_triton_slow}

if [ "$#" -eq 0 ]; then
    TAG=default
else
    TAG=$(printf '%s' "$*" | tr -cs 'A-Za-z0-9' '_' | sed 's/^_//;s/_$//')
fi
DATE=$(date +%Y%m%d)
ZIP_PATH=${ZIP_PATH:-"$TRITON_ROOT/mla_min_${TAG}_att_${DATE}.zip"}

# The ATT decoder is not in the gfx1250env venv; it ships with the ROCm SDK
# devel wheel in the neighbouring environment.
DECODER=/home/aweinrau/devFA/faGfx1250/lib/python3.12/site-packages/_rocm_sdk_devel/lib

# Stay in the triton root: defaultEnv.sh sets TRITON_OVERRIDE_DIR=build-out/override/
# (and dump/mlir paths) relative to PWD. Running from aiter misses overrides.
cd "$TRITON_ROOT"
# shellcheck disable=SC1091
source defaultEnv.sh
# Kernel/IR dumping only slows compilation and is unrelated to the trace.
# Leave TRITON_OVERRIDE_DIR so ISA override experiments still apply.
unset MLIR_DUMP_PATH TRITON_KERNEL_DUMP

rm -rf "$OUT"
mkdir -p "$OUT"

gpu-lock rocprofv3 \
    --att \
    --att-library-path "$DECODER" \
    --att-consecutive-kernels 1 \
    --att-target-cu 1 \
    --att-shader-engine-mask 0x1 \
    --att-buffer-size 402653184 \
    --kernel-include-regex mla_decode_min \
    -d "$OUT" -o att --output-format csv \
    -- python aiter/op_tests/op_benchmarks/triton/mla_min.py \
        --batch_size 512 --ctx_lens 16384 --iters 1 "$@"

UI=$(find "$OUT" -maxdepth 1 -name "ui_output*" -type d | head -1)
if [ -z "$UI" ]; then
    echo "FAILED: no ui_output directory produced" >&2
    exit 1
fi
# An empty decode still produces ui_output/, so check for actual wave data.
if ! ls "$UI"/se0_*_wv*.json >/dev/null 2>&1; then
    echo "FAILED: ui_output has no wave files -- the traced CU saw no waves." >&2
    echo "        Raise --batch_size so every CU gets a workgroup." >&2
    exit 1
fi

WAVES=$(ls "$UI"/se0_*_wv*.json | wc -l)
echo "ok: $UI  ($WAVES waves)"
echo "analyse: python3 $SCRIPT_DIR/mla_min_att_analyze.py $UI 'label'"

rm -f "$ZIP_PATH"
# Zip from the parent so the archive contains the outdir name, not ".".
PARENT=$(cd "$OUT/.." && pwd)
BASE=$(basename "$OUT")
(cd "$PARENT" && zip -r -q "$ZIP_PATH" "$BASE")
ZIP_ABS=$(cd "$(dirname "$ZIP_PATH")" && pwd)/$(basename "$ZIP_PATH")
ZIP_MB=$(du -h "$ZIP_ABS" | awk '{print $1}')
echo "zip: $ZIP_ABS  ($ZIP_MB)"
echo
echo "Download from your machine:"
echo
echo "  sftp ${SFTP_HOST}:${ZIP_ABS}"
echo
echo "If that opens an interactive session instead of fetching:"
echo
echo "  sftp ${SFTP_HOST} <<'EOF'"
echo "  get ${ZIP_ABS}"
echo "  EOF"
