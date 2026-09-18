#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Minimal gfx1250 MLA decode kernel for experiments.

Specialization of `_mla_decode_fwd_kernel` from
`aiter/ops/triton/_gluon_kernels/gfx1250/attention/mla.py`, stripped to exactly
the configuration we benchmark so the hot loop is readable and easy to modify:

    fp8 query x fp8 KV cache        (no bf16 / nvfp4 / scaled-WMMA paths)
    shuffled KV cache               (no gather path)
    4-stage TDM pipeline            (no 2-stage path)
    1 segment per sequence          (no split-K, no partials, no reduce kernel)
    decode only, 1 token per seq    (no prefill, no BLOCK_Q > 1)
    128 query heads / 1 KV head     (BLOCK_M == NUM_QUERIES_PER_KV, no head blocks)

Those constraints collapse a lot of the production kernel. In particular every
query mask is statically true, the causal mask reduces to a length check on the
final tile, and the epilogue is a single normalize-and-store.

Run directly to check against a torch reference and benchmark:

    gpu-lock python op_tests/op_benchmarks/triton/mla_min.py
    gpu-lock python op_tests/op_benchmarks/triton/mla_min.py --batch_size 1024
"""

from __future__ import annotations

import argparse

import torch
import triton
import triton.experimental.gluon.language as gl
import triton.language as tl
from triton.experimental import gluon

E4M3 = torch.float8_e4m3fn

# ---------------------------------------------------------------------------
# Fixed shape of the configuration under test
# ---------------------------------------------------------------------------
# Declared as gl.constexpr so the kernel body can reference them directly;
# host code reads them through `.value`.
KV_LORA_RANK = gl.constexpr(512)
QK_ROPE_HEAD_DIM = gl.constexpr(64)
HEAD_SIZE = gl.constexpr(KV_LORA_RANK.value + QK_ROPE_HEAD_DIM.value)
NUM_QUERY_HEADS = gl.constexpr(128)
BLOCK_SIZE = gl.constexpr(64)  # KV page size, also the tile size
BLOCK_M = gl.constexpr(NUM_QUERY_HEADS.value)
NUM_STAGES = gl.constexpr(4)
RCP_LN2 = gl.constexpr(1.4426950408889634)
NUM_WARPS = 4

# KV cache shuffle granularity; see shuffle_kv_buffer in test_mla.py.
K_WIDTH = gl.constexpr(16)
# Dot-operand k_width for the fp8 4-stage path.
DOT_K_WIDTH = 32

# ---------------------------------------------------------------------------
# Layouts
# ---------------------------------------------------------------------------
QK_WMMA: gl.constexpr = gl.amd.AMDWMMALayout(
    version=3, transposed=True, warp_bases=[(1, 0), (2, 0)], reg_bases=[],
    instr_shape=[16, 16, 64],
)
PV_WMMA: gl.constexpr = gl.amd.AMDWMMALayout(
    version=3, transposed=True, warp_bases=[(0, 1), (0, 2)], reg_bases=[],
    instr_shape=[16, 16, 64],
)
Q_DOT: gl.constexpr = gl.DotOperandLayout(0, QK_WMMA, DOT_K_WIDTH)
K_DOT: gl.constexpr = gl.DotOperandLayout(1, QK_WMMA, DOT_K_WIDTH)
P_DOT: gl.constexpr = gl.DotOperandLayout(0, PV_WMMA, DOT_K_WIDTH)
V_DOT: gl.constexpr = gl.DotOperandLayout(1, PV_WMMA, DOT_K_WIDTH)

Q_LORA_SMEM: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
    [[KV_LORA_RANK.value, 8]], [BLOCK_M.value, KV_LORA_RANK.value], [1, 0]
)
Q_ROPE_SMEM: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
    [[QK_ROPE_HEAD_DIM.value, 8]], [BLOCK_M.value, QK_ROPE_HEAD_DIM.value], [1, 0]
)
KV_SMEM: gl.constexpr = gl.SwizzledSharedLayout(
    vec=1, per_phase=1, max_phase=1, order=[1, 0]
)
Q_LORA_LOAD: gl.constexpr = gl.BlockedLayout(
    size_per_thread=[1, 8], threads_per_warp=[1, 32], warps_per_cta=[NUM_WARPS, 1],
    order=[1, 0],
)
Q_ROPE_LOAD: gl.constexpr = gl.BlockedLayout(
    size_per_thread=[1, 8], threads_per_warp=[4, 8], warps_per_cta=[NUM_WARPS, 1],
    order=[1, 0],
)


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
@gluon.jit
def _unshuffle_lora(smem, buffer_id):
    """LDS view of one KV page as [BLOCK_SIZE, KV_LORA_RANK]."""
    return (
        smem.index(buffer_id)
        .reshape((1, BLOCK_SIZE // 16, KV_LORA_RANK // (2 * K_WIDTH), 2, 16, K_WIDTH))
        .permute((0, 1, 4, 2, 3, 5))
        .reshape((BLOCK_SIZE, KV_LORA_RANK))
    )


@gluon.jit
def _unshuffle_rope(smem, buffer_id):
    return (
        smem.index(buffer_id)
        .reshape(
            (1, BLOCK_SIZE // 16, QK_ROPE_HEAD_DIM // (2 * K_WIDTH), 2, 16, K_WIDTH)
        )
        .permute((0, 1, 4, 2, 3, 5))
        .reshape((BLOCK_SIZE, QK_ROPE_HEAD_DIM))
    )


@gluon.jit
def _load_page(desc, smem, block_idx, buffer_id):
    gl.amd.gfx1250.tdm.async_load(desc, [block_idx, 0], smem.index(buffer_id))


@gluon.jit
def _process_tile(
    q_lora, q_rope, kv_lora_smem, k_rope_smem, buffer_id, tile_idx,
    qk_factor, seq_len, L, M, acc,
    wait_lora: gl.constexpr, wait_rope: gl.constexpr, IS_LAST: gl.constexpr,
):
    # --- S = Q @ K^T over the 576-deep reduction, in two WMMA groups --------
    S = gl.zeros([BLOCK_M, BLOCK_SIZE], dtype=tl.float32, layout=QK_WMMA)

    gl.amd.gfx1250.tdm.async_wait(wait_lora)
    k_lora = _unshuffle_lora(kv_lora_smem, buffer_id).permute((1, 0)).load(layout=K_DOT)
    S = gl.amd.gfx1250.wmma(q_lora, k_lora, S)

    gl.amd.gfx1250.tdm.async_wait(wait_rope)
    k_rope = _unshuffle_rope(k_rope_smem, buffer_id).permute((1, 0)).load(layout=K_DOT)
    S = gl.amd.gfx1250.wmma(q_rope, k_rope, S) * qk_factor

    if IS_LAST:
        # Only the final tile can run past the end of the sequence. With one
        # decode token, the causal bound is exactly seq_len.
        pos = tile_idx * BLOCK_SIZE + gl.arange(
            0, BLOCK_SIZE, layout=gl.SliceLayout(0, QK_WMMA)
        )
        S = gl.where(pos[None, :] < seq_len, S, float("-inf"))

    # --- online softmax over the 64 KV positions of this tile ---------------
    m_ij = gl.maximum(M, gl.max(S, axis=1))
    p = gl.exp2(S - m_ij[:, None])
    alpha = gl.exp2(M - m_ij)
    M = m_ij
    L = L * alpha + gl.sum(p, 1)
    acc = acc * gl.convert_layout(alpha[:, None], layout=PV_WMMA)

    # --- acc += P @ V, V being the same LDS page read untransposed ----------
    v = _unshuffle_lora(kv_lora_smem, buffer_id).load(layout=V_DOT)
    p = gl.convert_layout(p.to(v.dtype), P_DOT)
    acc = gl.amd.gfx1250.wmma(p, v, acc)
    return L, M, acc


@gluon.jit
def mla_decode_min_kernel(
    out_ptr,  # [num_seqs, NUM_QUERY_HEADS, KV_LORA_RANK]
    query_ptr,  # [num_seqs, NUM_QUERY_HEADS, HEAD_SIZE] fp8
    kv_buffer_ptr,  # [num_blocks, 1, BLOCK_SIZE, HEAD_SIZE] fp8, shuffled
    block_tables_ptr,  # [num_seqs, max_blocks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    q_descale_ptr,
    kv_descale_ptr,
    SCALE: gl.constexpr,
    block_tables_stride: gl.int64,
    query_stride_0: gl.int64,
    query_stride_1: gl.int64,
    out_stride_0: gl.int64,
    out_stride_1: gl.int64,
    kv_page_stride: gl.int32,
    num_pages: gl.int32,
):
    seq_idx = gl.program_id(0)
    seq_len = gl.load(seq_lens_ptr + seq_idx)
    num_tiles = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE

    qk_factor: gl.float32 = SCALE * RCP_LN2
    qk_factor = qk_factor * gl.load(q_descale_ptr) * gl.load(kv_descale_ptr)
    out_factor: gl.float32 = gl.load(kv_descale_ptr)

    # ---- Q, loaded once and kept in registers for the whole loop -----------
    # Row m of the block is query head m of this sequence: BLOCK_M equals
    # NUM_QUERY_HEADS, so no masking is required anywhere.
    offs_m_l = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, Q_LORA_LOAD))
    offs_d_l = gl.arange(0, KV_LORA_RANK, layout=gl.SliceLayout(0, Q_LORA_LOAD))
    offs_m_r = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, Q_ROPE_LOAD))
    offs_d_r = gl.arange(0, QK_ROPE_HEAD_DIM, layout=gl.SliceLayout(0, Q_ROPE_LOAD))

    q_base = query_ptr + seq_idx * query_stride_0
    q_lora_smem = gl.allocate_shared_memory(
        query_ptr.type.element_ty, [BLOCK_M, KV_LORA_RANK], Q_LORA_SMEM
    )
    q_lora_smem.store(
        gl.load(q_base + offs_m_l[:, None] * query_stride_1 + offs_d_l[None, :])
    )
    q_lora = q_lora_smem.load(layout=Q_DOT)

    q_rope_smem = gl.allocate_shared_memory(
        query_ptr.type.element_ty, [BLOCK_M, QK_ROPE_HEAD_DIM], Q_ROPE_SMEM
    )
    q_rope_smem.store(
        gl.load(
            q_base
            + offs_m_r[:, None] * query_stride_1
            + (KV_LORA_RANK + offs_d_r)[None, :]
        )
    )
    q_rope = q_rope_smem.load(layout=Q_DOT)

    # ---- KV ring buffers ---------------------------------------------------
    kv_lora_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=kv_buffer_ptr,
        shape=(num_pages, BLOCK_SIZE * KV_LORA_RANK),
        strides=(kv_page_stride, 1),
        block_shape=(gl.constexpr(1), BLOCK_SIZE * KV_LORA_RANK),
        layout=KV_SMEM,
    )
    k_rope_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=kv_buffer_ptr + BLOCK_SIZE * KV_LORA_RANK,
        shape=(num_pages, BLOCK_SIZE * QK_ROPE_HEAD_DIM),
        strides=(kv_page_stride, 1),
        block_shape=(gl.constexpr(1), BLOCK_SIZE * QK_ROPE_HEAD_DIM),
        layout=KV_SMEM,
    )
    kv_lora_smem = gl.allocate_shared_memory(
        kv_lora_desc.dtype, [NUM_STAGES] + kv_lora_desc.block_shape, KV_SMEM
    )
    k_rope_smem = gl.allocate_shared_memory(
        k_rope_desc.dtype, [NUM_STAGES] + k_rope_desc.block_shape, KV_SMEM
    )

    table = block_tables_ptr + seq_idx * block_tables_stride

    # ---- accumulators ------------------------------------------------------
    M = gl.full([BLOCK_M], float("-inf"), gl.float32, gl.SliceLayout(1, QK_WMMA))
    L = gl.full([BLOCK_M], 1.0, gl.float32, gl.SliceLayout(1, QK_WMMA))
    acc = gl.zeros([BLOCK_M, KV_LORA_RANK], dtype=gl.float32, layout=PV_WMMA)

    # ---- prologue: prime 3 of the 4 slots ----------------------------------
    for prime in gl.static_range(3):
        pg = gl.load(table + gl.minimum(prime, num_tiles - 1))
        _load_page(kv_lora_desc, kv_lora_smem, pg, prime)
        _load_page(k_rope_desc, k_rope_smem, pg, prime)

    # ---- steady state: load 3 tiles ahead, consume the oldest slot ---------
    slot: gl.int32 = 0
    for tile in range(0, num_tiles - 3):
        pg = gl.load(table + gl.minimum(tile + 3, num_tiles - 1))
        fill_slot = (slot + 3) % NUM_STAGES
        _load_page(kv_lora_desc, kv_lora_smem, pg, fill_slot)
        _load_page(k_rope_desc, k_rope_smem, pg, fill_slot)
        L, M, acc = _process_tile(
            q_lora, q_rope, kv_lora_smem, k_rope_smem, slot, tile,
            qk_factor, seq_len, L, M, acc,
            wait_lora=7, wait_rope=6, IS_LAST=False,
        )
        slot = (slot + 1) % NUM_STAGES

    # ---- drain: three tiles with no new loads in flight --------------------
    L, M, acc = _process_tile(
        q_lora, q_rope, kv_lora_smem, k_rope_smem, slot, num_tiles - 3,
        qk_factor, seq_len, L, M, acc,
        wait_lora=5, wait_rope=4, IS_LAST=False,
    )
    slot = (slot + 1) % NUM_STAGES
    L, M, acc = _process_tile(
        q_lora, q_rope, kv_lora_smem, k_rope_smem, slot, num_tiles - 2,
        qk_factor, seq_len, L, M, acc,
        wait_lora=3, wait_rope=2, IS_LAST=False,
    )
    slot = (slot + 1) % NUM_STAGES
    L, M, acc = _process_tile(
        q_lora, q_rope, kv_lora_smem, k_rope_smem, slot, num_tiles - 1,
        qk_factor, seq_len, L, M, acc,
        wait_lora=1, wait_rope=0, IS_LAST=True,
    )

    # ---- epilogue ----------------------------------------------------------
    acc = acc * gl.convert_layout(1.0 / L[:, None], layout=PV_WMMA) * out_factor

    offs_m_o = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, PV_WMMA))
    offs_d_o = gl.arange(0, KV_LORA_RANK, layout=gl.SliceLayout(0, PV_WMMA))
    o_offs = (
        seq_idx * out_stride_0
        + offs_m_o[:, None] * out_stride_1
        + offs_d_o[None, :]
    )
    gl.store(out_ptr + o_offs, acc.to(out_ptr.dtype.element_ty))


# ---------------------------------------------------------------------------
# Host wrapper
# ---------------------------------------------------------------------------
def mla_decode_min(q, kv_buffer, block_tables, seq_lens, q_descale, kv_descale,
                   softmax_scale, out=None, out_dtype=torch.bfloat16):
    # Caller must ensure every sequence spans more than three KV pages
    # (seq_len > 192): the drain phase peels three tiles unconditionally.
    # Checking it here would force a device sync on every launch.
    num_seqs = q.shape[0]
    if out is None:
        out = torch.empty(
            (num_seqs, NUM_QUERY_HEADS.value, KV_LORA_RANK.value),
            dtype=out_dtype, device=q.device,
        )
    return out, mla_decode_min_kernel[(num_seqs,)](
        out, q, kv_buffer, block_tables, seq_lens, q_descale, kv_descale,
        SCALE=softmax_scale,
        block_tables_stride=block_tables.stride(0),
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        out_stride_0=out.stride(0),
        out_stride_1=out.stride(1),
        kv_page_stride=kv_buffer.stride(1),
        num_pages=kv_buffer.shape[0],
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES.value,
        waves_per_eu=1,
    )


# ---------------------------------------------------------------------------
# Driver: same data generation as bench_mla.py, checked against the production
# kernel so this file can be trusted as a starting point for experiments.
# ---------------------------------------------------------------------------
def _make_inputs(batch_size, ctx_lens, seed=0):
    import random

    torch.manual_seed(0)
    random.seed(seed)

    seq_lens = torch.empty(batch_size, dtype=torch.int32, device="cuda")
    for i in range(batch_size):
        seq_lens[i] = max(random.normalvariate(ctx_lens, ctx_lens / 2), ctx_lens)

    max_seqlen = int(seq_lens.max())
    blocks_per_seq = (max_seqlen + BLOCK_SIZE.value - 1) // BLOCK_SIZE.value
    num_pages = batch_size * blocks_per_seq
    block_tables = torch.randperm(
        num_pages, dtype=torch.int32, device="cuda"
    ).reshape(batch_size, blocks_per_seq)

    kv = torch.randn(
        (num_pages, BLOCK_SIZE.value, 1, HEAD_SIZE.value), dtype=torch.bfloat16, device="cuda"
    ).to(E4M3)
    q = torch.randn(
        (batch_size, NUM_QUERY_HEADS.value, HEAD_SIZE.value), dtype=torch.bfloat16, device="cuda"
    ).to(E4M3)
    q_descale = torch.rand((1,), dtype=torch.float32, device="cuda")
    kv_descale = torch.rand((1,), dtype=torch.float32, device="cuda")
    return q, kv, block_tables, seq_lens, q_descale, kv_descale


def _reference(q, kv, block_tables, seq_lens, q_descale, kv_descale, scale):
    """Dense fp32 attention over the gathered KV pages."""
    out = torch.empty(
        (q.shape[0], NUM_QUERY_HEADS.value, KV_LORA_RANK.value), dtype=torch.float32, device="cuda"
    )
    qf = q.to(torch.float32) * q_descale
    for i in range(q.shape[0]):
        n = int(seq_lens[i])
        pages = block_tables[i, : (n + BLOCK_SIZE.value - 1) // BLOCK_SIZE.value]
        k = kv[pages].reshape(-1, HEAD_SIZE.value)[:n].to(torch.float32) * kv_descale
        s = (qf[i] @ k.T) * scale
        p = torch.softmax(s, dim=-1)
        out[i] = p @ k[:, :KV_LORA_RANK.value]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--ctx_lens", type=int, default=16384)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--check", action="store_true", help="compare against production")
    p.add_argument("--ref", action="store_true", help="compare against torch reference")
    p.add_argument("--compare", action="store_true", help="also time the production kernel")
    args = p.parse_args()

    q, kv, block_tables, seq_lens, q_descale, kv_descale = _make_inputs(
        args.batch_size, args.ctx_lens
    )
    scale = 1.0 / (HEAD_SIZE.value**0.5)

    from op_tests.triton_tests.attention.test_mla import shuffle_kv_buffer

    kv_shuf = shuffle_kv_buffer(kv, KV_LORA_RANK.value)

    out, kernel = mla_decode_min(
        q, kv_shuf, block_tables, seq_lens, q_descale, kv_descale, scale
    )
    torch.cuda.synchronize()
    print(
        f"vgpr={kernel.n_regs} spill={kernel.n_spills} "
        f"lds={kernel.metadata.shared} B"
    )

    if args.ref:
        ref = _reference(q, kv, block_tables, seq_lens, q_descale, kv_descale, scale)
        d = (out.to(torch.float32) - ref).abs()
        print(f"vs torch reference: max abs {d.max():.4f}  mean abs {d.mean():.5f}")

    if args.check:
        from aiter.ops.triton.attention.mla import mla_decode_fwd

        cu = torch.arange(
            args.batch_size + 1, dtype=torch.int32, device="cuda"
        )
        prod_out = torch.empty(
            (args.batch_size, NUM_QUERY_HEADS.value, KV_LORA_RANK.value),
            dtype=torch.bfloat16,
            device="cuda",
        )
        got = mla_decode_fwd(
            q, kv_shuf, prod_out, cu_seqlens_q=cu, seqused_k=seq_lens,
            max_seqlen_kv=int(seq_lens.max()), block_tables=block_tables,
            softmax_scale=scale, kv_lora_rank=KV_LORA_RANK.value,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM.value, causal=True,
            q_descale=q_descale, kv_descale=kv_descale, out_scale=None,
            shuffled_kv_cache=True, skip_reduce=False,
        )
        ref = got if isinstance(got, torch.Tensor) else got[0]
        d = (out.to(torch.float32) - ref.to(torch.float32)).abs()
        print(f"vs production kernel: max abs {d.max():.5f}  mean abs {d.mean():.6f}")

    # Byte accounting identical to bench_mla.py so TB/s is comparable.
    mem = (
        q.numel() * q.element_size()
        + int(seq_lens.sum()) * HEAD_SIZE.value * 2 * kv.element_size()
        + out.numel() * q.element_size()
    ) * 1e-12

    def time_it(fn):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        t0, t1 = torch.cuda.Event(True), torch.cuda.Event(True)
        t0.record()
        for _ in range(args.iters):
            fn()
        t1.record()
        torch.cuda.synchronize()
        return t0.elapsed_time(t1) / args.iters * 1e3

    assert int(seq_lens.min()) > 3 * BLOCK_SIZE.value, "needs seq_len > 192"
    us = time_it(
        lambda: mla_decode_min(
            q, kv_shuf, block_tables, seq_lens, q_descale, kv_descale, scale, out=out
        )
    )
    print(f"mla_min    batch {args.batch_size}: {us:8.2f} us   {mem / (us * 1e-6):.3f} TB/s")

    if args.compare:
        from aiter.ops.triton.attention.mla import mla_decode_fwd

        cu = torch.arange(args.batch_size + 1, dtype=torch.int32, device="cuda")
        po = torch.empty(
            (args.batch_size, NUM_QUERY_HEADS.value, KV_LORA_RANK.value),
            dtype=torch.bfloat16, device="cuda",
        )
        pus = time_it(
            lambda: mla_decode_fwd(
                q, kv_shuf, po, cu_seqlens_q=cu, seqused_k=seq_lens,
                max_seqlen_kv=int(seq_lens.max()), block_tables=block_tables,
                softmax_scale=scale, kv_lora_rank=KV_LORA_RANK.value,
                qk_rope_head_dim=QK_ROPE_HEAD_DIM.value, causal=True,
                q_descale=q_descale, kv_descale=kv_descale, out_scale=None,
                shuffled_kv_cache=True, skip_reduce=False,
            )
        )
        print(
            f"production batch {args.batch_size}: {pus:8.2f} us   "
            f"{mem / (pus * 1e-6):.3f} TB/s   (mla_min is {pus / us - 1:+.1%})"
        )


if __name__ == "__main__":
    main()
