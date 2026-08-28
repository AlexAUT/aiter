# gfx1250 MLA Warp-Specialization Experiment

Date: 2026-08-25

Status: experimental; correct and spill-free, but not performance-competitive.

## Goal

The existing gfx1250 MLA decode loop serializes:

1. QK WMMA
2. online softmax
3. accumulator rescaling
4. PV WMMA

The experiment tests whether persistent wave roles can overlap
`QK/softmax(n+1)` with `PV(n)` without exposing the large FP32 PV accumulator
to compiler pipeline-stage versioning.

## Kernel design

The implementation is in:

- `aiter/ops/triton/_gluon_kernels/gfx1250/attention/mla.py`
- `aiter/ops/triton/attention/mla.py`

It is disabled by default and enabled with:

```bash
export MLA_WARP_SPECIALIZE_EXPERIMENT=1
```

`MLA_BLOCK_M_EXPERIMENT=64` can force the matched non-specialized B64
comparison.

The optimized experiment can also use an FP8 probability handoff:

```bash
export MLA_WS_FP8_P=1
```

The prototype is restricted to BF16 query/cache, `BLOCK_M=64`,
`TILE_SIZE=64`, and two KV buffers.

### Wave roles

The launch has eight waves in total:

- Four default-partition waves are persistent PV consumers.
- Four worker-partition waves are persistent QK/softmax producers.

The producer performs TDM loads, QK-LoRA WMMA, QK-RoPE WMMA, causal masking,
and online-softmax state updates. It never allocates or references the PV
accumulator.

The consumer allocates the `[64, 512]` FP32 accumulator and keeps it in
registers for the full context loop. It rescales the accumulator by `alpha`,
loads the V tile, and performs PV WMMA. The accumulator never crosses a
partition or pipeline boundary.

### Producer/consumer handoff

Two ping-pong LDS slots carry:

- BF16 probabilities `P[64, 64]`, or optional FP8 probabilities
- FP32 `alpha[64]`
- FP32 softmax state `L[64]` and `M[64]`

The initial implementation used separate empty and ready mbarriers. The
optimized implementation uses one alternating-phase LDS mbarrier per slot:
phase 1 is empty and phase 0 is ready. The producer and consumer each perform
one wait and one arrive per tile.

This permits the producer to work on tile `n+1` while the consumer performs
PV for tile `n`.

## Static compilation

Compilation was performed through Gluon's `warmup()` path only. No kernel was
launched during static analysis.

For the matched B64 baseline:

- Four waves
- 899 VGPRs
- 0 VGPR spills
- 0 scratch bytes and no scratch instructions
- 149,504 bytes LDS

For the warp-specialized B64 prototype:

- Eight waves
- 460 VGPRs
- 0 VGPR spills
- 0 scratch bytes and no scratch instructions
- 241,460 bytes LDS
- Eight SGPR-to-VGPR-lane saves in the common one-segment configuration
- Zero SGPR lane-save operations inside either hot loop

For the optimized FP8-handoff variant:

- Eight waves
- 460 VGPRs
- 0 VGPR spills
- 0 scratch bytes and no scratch instructions
- 233,236 bytes LDS

TTGIR contains one `ttg.warp_specialize` operation with the PV accumulator
only in the default partition. The QK worker partition has no
`tensor<64x512xf32>` accumulator state.

The static scheduler reports 16,268 aggregate cycles and 13% WMMA efficiency
for the matched B64 kernel, versus 17,139 aggregate cycles and 6% for the
specialized kernel. These aggregate values are not a valid concurrency model:
the estimator walks the warp-specialization switch CFG as if the partitions
were serial. Resource usage and spill counts are reliable; the aggregate
cycle and WMMA-efficiency estimates are included only as diagnostics.

## Correctness

The first hardware launch used batch 4, context 512, 128 query heads, one KV
head, BF16 inputs, and a shuffled KV cache. It was compared against the
existing PyTorch MLA reference.

- Maximum absolute error: 0.002930
- Mean absolute error: 0.000274
- Error ratio at `atol=0.015`, `rtol=0.01`: 0
- Result: passed

## Hardware methodology

Every hardware command:

1. Sourced `/home/aweinrau/devFA/triton/defaultEnv.sh`
2. Set the local Triton and AITER `PYTHONPATH`
3. Ran under `gpu-lock`

The main comparison used batch 128, decode query length 1, context 8192,
128 query heads, one KV head, BF16, rank 512, RoPE dimension 64, and block
size 64.

Timings use `triton.testing.do_bench`. The stable comparison interleaved the
three variants across five rounds, with 10 warmup and 100 measured iterations
per sample. Reported values are medians of the five samples.

Logical bandwidth uses the existing MLA benchmark convention, counting the
latent KV vector for both logical K and V traffic.

## Hardware results

Stable context-8192 comparison:

- Production B128: 0.215828 ms, 11.359 logical TB/s
- Matched non-specialized B64: 0.196386 ms, 12.483 logical TB/s
- Warp-specialized B64: 0.339623 ms, 7.219 logical TB/s

The specialized kernel is:

- 72.9% higher latency than the matched B64 baseline
- 42.2% lower logical bandwidth than the matched B64 baseline
- 57.4% higher latency than the production B128 path

The five specialized samples ranged from 0.339552 to 0.339754 ms, so this is
not measurement noise.

Context-length sweep at batch 128:

- Context 2048:
  - Production B128: 0.121945 ms, 5.245 logical TB/s
  - Matched B64: 0.069294 ms, 9.231 logical TB/s
  - Warp-specialized B64: 0.099327 ms, 6.440 logical TB/s
  - Specialized latency regression versus B64: 43.3%
- Context 8192:
  - Production B128: 0.214650 ms, 11.421 logical TB/s
  - Matched B64: 0.195973 ms, 12.510 logical TB/s
  - Warp-specialized B64: 0.339577 ms, 7.219 logical TB/s
  - Specialized latency regression versus B64: 73.3%
- Context 32768:
  - Production B128: 0.594475 ms, 16.316 logical TB/s
  - Matched B64: 0.728720 ms, 13.310 logical TB/s
  - Warp-specialized B64: 1.298413 ms, 7.470 logical TB/s
  - Specialized latency regression versus B64: 78.2%

The regression grows with context length, showing that the dominant cost is
inside the per-tile pipeline rather than launch or initial-fill overhead.

## Optimization iterations

The following variants were compiled and measured after the initial result.

### Three KV buffers

This attempted to overlap `TDM(n+2)`, `QK/softmax(n+1)`, and `PV(n)` with
three KV buffers and two probability slots.

- 512 VGPRs
- 276 VGPR spills
- 1,108 bytes scratch
- 164 scratch load/store instructions

It failed the static spill gate and was not launched.

### Final-only L/M stores with a loop branch

Writing `L/M` only on the final iteration remained correct and spill-free,
but the runtime branch increased latency to 0.3424 ms. It was reverted.

### Safe-prefix masking

The first prototype applied a full causal mask on every tile. Reusing
`safe_tile_end` avoids the `[64,64]` mask for the safe prefix.

- Correctness: passed
- Latency: 0.3354 ms
- Improvement over the initial 0.3396 ms: 1.3%

This optimization is retained.

### FP8 probability storage

Storing `P` as FP8 in LDS and converting it back to BF16 before PV halves the
dominant cross-partition payload.

- Correctness error ratio: 0 at `atol=0.015`, `rtol=0.01`
- Maximum absolute error in the small reference case: 0.0078125
- Dual-barrier latency: 0.3014 ms
- Improvement over BF16 handoff: approximately 10%

This is retained as an optional experimental mode, not the default, because
the accuracy study is limited.

### Single alternating barrier

One alternating-phase mbarrier per slot replaces the separate empty and ready
barriers and removes per-iteration phase arithmetic.

- BF16 handoff: 0.3343 ms
- FP8 handoff: 0.2908 ms
- Correctness: passed
- VGPR spills: 0

The best FP8 result is still 48.0% higher latency than the matched B64
baseline (0.1964 ms).

### One-time final-state barrier

Moving `L/M` to a one-time post-loop handoff used 458 VGPRs and remained
spill-free and correct, but latency was 0.2914 ms. It was reverted.

### Optimized context sweep

- Context 2048:
  - Matched B64: 0.062586 ms, 10.220 logical TB/s
  - Optimized FP8 specialization: 0.086633 ms, 7.383 logical TB/s
  - Latency regression: 38.4%
- Context 32768:
  - Matched B64: 0.729475 ms, 13.296 logical TB/s
  - Optimized FP8 specialization: 1.104118 ms, 8.785 logical TB/s
  - Latency regression: 51.4%

### TDM/LDS producer and monolithic WMMA consumer

A second design dedicated the worker partition to block-table lookup and
TDM HBM-to-LDS transfers. The default partition retained QK, online softmax,
PV, and the FP32 accumulator. This is the desired memory/compute split: it
removes the probability/state LDS handoff and allows TDM/LDS work for tile
`n+1` to overlap the WMMA-heavy compute for tile `n`.

`warp_specialize` adds worker waves; it does not reassign one of the four
default waves to a different SIMD role. The B64 launch therefore still has
eight waves. Its actual compiled resources were:

- 512 VGPRs
- 85 VGPR spills
- 344 bytes scratch
- 46 scratch load/store instructions
- 149,540 bytes LDS

It failed the static spill gate and was not launched.

B32 reduced the compiled footprint to 482 VGPRs with zero spills, zero
scratch instructions, and 148,516 bytes LDS. B16 used 492 VGPRs and was also
spill-free. Both smaller shapes normally use only two compute waves, however,
whereas warp specialization requires a four-wave default group. A four-wave
B32 QK layout must partition the 64-column score tile across waves instead of
partitioning only the head rows.

The B32 candidate was correct for one KV tile (maximum absolute error
0.000244), but its loop-carried online-softmax state was incorrect once QK
was partitioned across waves. At two tiles, maximum/mean absolute error was
0.050781/0.009174 and correlation with the reference was 0.0215. Replacing
TDM-completion barriers with explicit `async_wait(0)`, serializing the
pipeline to one LDS slot, and trying both 2x2 and N-only QK wave mappings did
not change the failure. This rules out LDS-buffer reuse and asynchronous
completion as the cause.

No correct, spill-free configuration remained, so this mode was removed
rather than retained behind an experimental environment variable. It was not
performance-benchmarked.

## Interpretation

The experiment solved the original register-allocation problem:

- Per-wave VGPR use fell from 899 to 460.
- Two waves per SIMD fit without VGPR spills.
- The PV accumulator remains partition-private.

However, the handoff is too expensive:

- LDS use increases by 91,956 bytes (61.5%).
- Every tile writes and rereads an 8 KiB probability tile.
- `alpha`, `L`, and `M` add more LDS traffic.
- Each tile performs ready/empty mbarrier synchronization.
- The baseline keeps probability and softmax state in registers.
- The producer cannot release a tile until QK and softmax finish, while the
  consumer cannot begin PV until the complete probability tile is visible.

The intended QK/PV overlap does not recover the LDS and synchronization cost.
The increasing long-context regression confirms that this cost recurs in the
hot loop.

There is also a more fundamental limitation: QK and PV both issue WMMA on
every SIMD. Persistent wave roles add independent wave scheduling, but do not
add another WMMA/XDL issue unit. QK WMMA and PV WMMA therefore contend for the
same execution resource rather than overlapping at full throughput. Only the
producer's softmax/VALU and memory work can overlap PV WMMA, and the existing
coexecution scheduler already captures part of that opportunity without an
LDS handoff.

## Conclusion and next steps

This implementation should not be enabled in production.

The result is still useful: true warp specialization avoids accumulator
versioning and is spill-free, so register pressure is no longer the blocker.
The blockers are cross-partition communication and shared WMMA/XDL
contention.

Potential follow-ups must reduce or remove the full `P` LDS round trip. Small
optimizations such as writing `L/M` only for the final tile will not recover a
43-78% regression. A viable next design likely needs one of:

- A hardware/compiler-supported register handoff between wave groups
- A fused ownership scheme where the wave group producing a probability
  subtile also performs its corresponding PV subtile
- Much smaller probability subtiles with streaming producer/consumer
  synchronization, provided barrier frequency does not dominate
- A different decomposition that overlaps TDM/VALU work while keeping QK,
  softmax, and PV in one register-owning partition

