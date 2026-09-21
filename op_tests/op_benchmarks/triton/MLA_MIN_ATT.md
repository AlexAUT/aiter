# Instruction-level (ATT) tracing for `mla_min.py` on gfx1250

How to capture and read an AMD Advanced Thread Trace of the minimal MLA decode
kernel, and the traps that make a trace silently useless.

Everything here is verified on this machine (gfx1250, 256 CUs, 320 KB LDS).

## Files

| file | purpose |
|---|---|
| `mla_min.py` | the kernel + driver (correctness, benchmark, variant flags) |
| `mla_min_att_trace.sh` | captures a trace and validates it |
| `mla_min_att_analyze.py` | per-iteration cycle and stall breakdown |

## Prerequisites

Always `source defaultEnv.sh` from the triton root, and always run GPU work
under `gpu-lock` — the GPU is shared and concurrent use can hang the hardware.

The ATT **decoder library is not in the `gfx1250env` venv**. It ships with the
ROCm SDK devel wheel in a neighbouring environment:

```
/home/aweinrau/devFA/faGfx1250/lib/python3.12/site-packages/_rocm_sdk_devel/lib
```

Without `--att-library-path` pointing there you get raw `.att` files and no
decode. Re-check this path if the environment is rebuilt.

## Capturing

```bash
cd /home/aweinrau/devFA/triton
./aiter/op_tests/op_benchmarks/triton/mla_min_att_trace.sh /tmp/att_base
./aiter/op_tests/op_benchmarks/triton/mla_min_att_trace.sh /tmp/att_pf --pgpf
./aiter/op_tests/op_benchmarks/triton/mla_min_att_trace.sh /tmp/att_best \
    --pgpf --sched 5 --npf 2 --pvsplit n
MLA_MIN_CTAS=2 ./aiter/op_tests/op_benchmarks/triton/mla_min_att_trace.sh /tmp/att_cta2
```

Any argument after the output directory is forwarded to `mla_min.py`, so all
the variant flags work: `--sched 0..6`, `--npf N`, `--pfk N`, `--pfrope`,
`--stages N`, `--pvsplit {n,m,h}`, `--pgpf`. `MLA_MIN_CTAS=2` selects the
2-CTA cluster build (chosen at import time, so it is an environment variable
rather than a flag).

On success the script zips `$OUT` to
`<triton-root>/mla_min_<args>_att_<YYYYMMDD>.zip` and prints an `sftp` get
for host `b0_triton_slow` (override with `SFTP_HOST` / `ZIP_PATH`).

The script always `cd`s to the triton root before launching Python, so relative
`TRITON_OVERRIDE_DIR=build-out/override/` (and dump paths) from `defaultEnv.sh`
apply. Do not run `mla_min.py` from `aiter/` if you are using ISA overrides.

The raw command, if you need to adapt it (cwd must be the triton root):

```bash
gpu-lock rocprofv3 \
    --att \
    --att-library-path /home/aweinrau/devFA/faGfx1250/lib/python3.12/site-packages/_rocm_sdk_devel/lib \
    --att-consecutive-kernels 1 \
    --att-target-cu 1 \
    --att-shader-engine-mask 0x1 \
    --kernel-include-regex mla_decode_min \
    -d /tmp/att_out -o att --output-format csv \
    -- python aiter/op_tests/op_benchmarks/triton/mla_min.py \
        --batch_size 512 --ctx_lens 16384 --iters 1
```

From your machine, after the script prints the zip path:

```bash
sftp b0_triton_slow:/home/aweinrau/devFA/triton/mla_min_<tag>_att_<YYYYMMDD>.zip
```

`rocprofv3` also accepts `-i config.yaml` for the same options if a file is
preferred.

## Traps

**A trace with no waves still looks like success.** ATT watches one CU on one
shader engine. At `--batch_size 32` the grid is 64 CTAs over 256 CUs, the
target CU receives nothing, and rocprofv3 still writes a `ui_output*` directory
— with an empty `code.json` and no wave files. Always confirm:

```bash
ls <ui_output_dir>/se0_*_wv*.json
```

`mla_min_att_trace.sh` fails loudly when they are missing. Use
`--batch_size 512` so every CU gets work.

**Use `--iters 1`.** Only the first dispatch is captured; more iterations just
lengthen the run.

**Turn off IR dumping.** `defaultEnv.sh` sets `MLIR_DUMP_PATH` and
`TRITON_KERNEL_DUMP`, which dump after every pass and dominate runtime. The
script unsets them.

**Watch out when combining with `TRITON_KERNEL_OVERRIDE`.** The override
directory is keyed on a hash that includes *argument specialization*, so an
override built from a `--batch_size 512` dump will silently not apply at
`--batch_size 32`. Confirm the `Overriding kernel with file ...` line appears.

## Reading the trace

```bash
python3 op_tests/op_benchmarks/triton/mla_min_att_analyze.py <ui_output_dir> 'label'
```

Prints cycles per hot-loop iteration (median/p10/p90), a cycle breakdown by
instruction category with the stall portion of each, and the top stalling
instructions. It locates the hot loop from execution counts rather than labels,
because the decoded listing gives branch targets as byte offsets.

Cross-check the reported `ds_load` and `v_wmma` counts per iteration against a
static disassembly count — they should match exactly.

Instruction records in `se0_*_wv*.json` are
`[timestamp, type, stall_cycles, duration_cycles, code_index]`, and
`code_index` maps into the `code` array of `code.json`.

For interactive inspection, open the `ui_output*` directory with the rocprofv3
ATT viewer. Expect roughly 25 MB per decoded trace and about 20 s per capture.

## Comparing configurations

Occupancy differs between captures, so **absolute cycles-per-iteration are only
comparable within a single capture session**. A trace that catches 2 waves will
report far more cycles per iteration than one that catches 4, for the same
kernel. Category shares and per-instruction stall attribution are the robust
signals; capture both arms of an A/B back to back.

## Reference points

For the 1-CTA kernel at batch 512, a healthy hot loop shows 122 `ds_load`,
136 `v_wmma`, 7 barrier pairs and 8 full `s_wait_dscnt 0x0` drains. Typical
cycle attribution is roughly a third VALU, a third WMMA, and a quarter LDS
waits. The VALU share is dominated by ~144 `v_pk_mul_f32` per iteration from
the online-softmax accumulator rescale.

If you see `wait: vmem` / `s_wait_loadcnt` costing tens of cycles per
iteration, the block-table pointer has lost its `tl.const` annotation — that
annotation puts it in the constant address space so it is fetched with a scalar
load and retires on `kmcnt` instead of stalling the vector path.
