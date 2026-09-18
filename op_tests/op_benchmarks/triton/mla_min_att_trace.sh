#!/usr/bin/env bash
# ATT (instruction-level thread trace) for mla_min.py on gfx1250.
#
#   ./att_trace.sh <outdir> [extra mla_min args...]
#
# Examples:
#   ./att_trace.sh /tmp/att_base
#   ./att_trace.sh /tmp/att_cta2 --pgpf          # MLA_MIN_CTAS=2 ./att_trace.sh ...
#
# The trace covers ONE dispatch on ONE compute unit of ONE shader engine, so
# the batch has to be big enough that the traced CU actually receives work --
# see the note at the bottom.
set -o pipefail

OUT=${1:?usage: att_trace.sh <outdir> [mla_min args...]}
shift

# The ATT decoder is not in the gfx1250env venv; it ships with the ROCm SDK
# devel wheel in the neighbouring environment.
DECODER=/home/aweinrau/devFA/faGfx1250/lib/python3.12/site-packages/_rocm_sdk_devel/lib

cd /home/aweinrau/dev/triton
source defaultEnv.sh
# Kernel/IR dumping only slows compilation and is unrelated to the trace.
unset MLIR_DUMP_PATH TRITON_KERNEL_DUMP
cd aiter

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
    -- python op_tests/op_benchmarks/triton/mla_min.py \
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
echo "ok: $UI  ($(ls "$UI"/se0_*_wv*.json | wc -l) waves)"
echo "analyse: python3 $(dirname "$0")/mla_min_att_analyze.py $UI 'label'"
