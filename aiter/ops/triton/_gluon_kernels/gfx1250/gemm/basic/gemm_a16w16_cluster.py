# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from functools import lru_cache

import triton.experimental.gluon.language as gl
from triton.experimental import gluon
from triton.experimental.gluon.language.amd import gfx1250

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr


@lru_cache(maxsize=128)
def cluster_splitk_layouts(bm, bn, bk, splits, peers):
    split_bits = [[1 << i, 0, 0] for i in range(splits.bit_length() - 1)]
    peer_bits = [1 << i for i in range(peers.bit_length() - 1)]
    a_bases = split_bits + [[0, bit, 0] for bit in peer_bits]
    b_bases = split_bits + [[0, 0, 0] for _ in peer_bits]
    a = gl.PaddedSharedLayout.with_identity_for(
        [[bk, 8]], [splits, peers * bm, bk], [2, 1, 0], a_bases
    )
    b = gl.PaddedSharedLayout.with_identity_for(
        [[bk, 8]], [splits, bn, bk], [2, 1, 0], b_bases
    )
    wmma = gl.amd.AMDWMMALayout(
        3, True, [[0, 0, 1], [0, 1, 0]], [], [16, 16, 32], cga_layout=a_bases, rank=3
    )
    if peers == 1:
        reduce_bases = [[1 << i] for i in range(splits.bit_length() - 1)]
        reduce_vec = 2 if splits == 16 else 4
        reducer = gl.BlockedLayout([reduce_vec], [32], [4], [0], reduce_bases)
    else:
        reduce_bases = [[0, 1 << i] for i in range(splits.bit_length() - 1)]
        reduce_bases += [[bit, 0] for bit in peer_bits]
        reducer = gl.BlockedLayout([1, 4], [1, 32], [1, 4], [1, 0], reduce_bases)
    return a, b, wmma, reducer


@gluon.jit
def _cluster_barrier(DRAIN: gl.constexpr):
    if DRAIN:
        # All waves must complete their partial stores before the CTA signals.
        gl.inline_asm_elementwise(
            "s_wait_storecnt 0\ns_mov_b32 $0, 0",
            "=s,~{memory}",
            [],
            gl.int32,
            is_pure=False,
            pack=1,
        )
    gl.barrier()
    gfx1250.cluster.arrive()
    gfx1250.cluster.wait()
    gl.inline_asm_elementwise(
        "s_mov_b32 $0, 0", "=s,~{memory}", [], gl.int32, is_pure=False, pack=1
    )


@gluon.jit
def _issue(a_desc, b_desc, a_buf, b_buf, slot, k_offset):
    a = gfx1250.tdm.update_tensor_descriptor(a_desc, add_offsets=[0, 0, k_offset])
    b = gfx1250.tdm.update_tensor_descriptor(b_desc, add_offsets=[0, 0, k_offset])
    gfx1250.tdm.async_load_fused(
        [(a, a_buf.index(slot), 3), (b, b_buf.index(slot), 12)]
    )


@gluon.jit
def _wmma(a_buf, b_buf, slot, acc, WMMA: gl.constexpr):
    a = gl.amd.cdna4.async_copy.load_shared_relaxed(
        a_buf.index(slot), gl.DotOperandLayout(0, WMMA, 8)
    )
    b = gl.amd.cdna4.async_copy.load_shared_relaxed(
        b_buf.index(slot).permute([0, 2, 1]), gl.DotOperandLayout(1, WMMA, 8)
    )
    return gfx1250.wmma(a, b, acc)


@gluon.jit
def _cluster_splitk_tile(
    A,
    B,
    C,
    Scratch,
    a_buf,
    b_buf,
    tile,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    BM: gl.constexpr,
    BN: gl.constexpr,
    BK: gl.constexpr,
    SPLITS: gl.constexpr,
    PEERS: gl.constexpr,
    A_SHARED: gl.constexpr,
    B_SHARED: gl.constexpr,
    WMMA: gl.constexpr,
    REDUCER: gl.constexpr,
):
    TILE_M: gl.constexpr = PEERS * BM
    TILES_M: gl.constexpr = M // TILE_M
    K_PART: gl.constexpr = K // SPLITS
    K_STEPS: gl.constexpr = K_PART // BK
    row = tile % TILES_M * TILE_M
    col = tile // TILES_M * BN
    a_desc = gfx1250.tdm.make_tensor_descriptor(
        A + row * K,
        shape=(SPLITS, TILE_M if PEERS == 1 else M - row, K_PART),
        strides=(K_PART, K, 1),
        block_shape=(SPLITS, TILE_M, BK),
        layout=A_SHARED,
    )
    b_desc = gfx1250.tdm.make_tensor_descriptor(
        B + col * K,
        shape=(SPLITS, BN if PEERS == 1 else N - col, K_PART),
        strides=(K_PART, K, 1),
        block_shape=(SPLITS, BN, BK),
        layout=B_SHARED,
    )
    if PEERS > 1:
        # Peers rendezvous before issuing the first multicast B load.
        _cluster_barrier(False)
    acc = gl.zeros((SPLITS, TILE_M, BN), gl.float32, WMMA)
    _issue(a_desc, b_desc, a_buf, b_buf, 0, 0)
    _issue(a_desc, b_desc, a_buf, b_buf, 1, BK)
    for step in range(2, K_STEPS):
        _issue(a_desc, b_desc, a_buf, b_buf, step % 3, step * BK)
        gfx1250.tdm.async_wait(2)
        acc = _wmma(a_buf, b_buf, (step - 2) % 3, acc, WMMA)
    gfx1250.tdm.async_wait(1)
    acc = _wmma(a_buf, b_buf, (K_STEPS - 2) % 3, acc, WMMA)
    gfx1250.tdm.async_wait(0)
    acc = _wmma(a_buf, b_buf, (K_STEPS - 1) % 3, acc, WMMA)

    if SPLITS == 1:
        rows = gl.arange(0, TILE_M, layout=gl.SliceLayout(0, gl.SliceLayout(2, WMMA)))
        cols = gl.arange(0, BN, layout=gl.SliceLayout(0, gl.SliceLayout(1, WMMA)))
        gl.store(
            C + (row + rows[None, :, None]) * N + col + cols[None, None, :],
            acc.to(C.type.element_ty),
        )
    else:
        scratch = Scratch + tile * SPLITS * TILE_M * BN
        split = gl.arange(0, SPLITS, layout=gl.SliceLayout(1, gl.SliceLayout(2, WMMA)))
        rows = gl.arange(0, TILE_M, layout=gl.SliceLayout(0, gl.SliceLayout(2, WMMA)))
        cols = gl.arange(0, BN, layout=gl.SliceLayout(0, gl.SliceLayout(1, WMMA)))
        offset = (
            split[:, None, None] * TILE_M * BN
            + rows[None, :, None] * BN
            + cols[None, None, :]
        )
        gfx1250.buffer_store(acc, scratch, offset)
        _cluster_barrier(True)

        # Each workgroup owns a different slice of the final tile.
        if PEERS == 1:
            elem = gl.arange(0, BM * BN, layout=REDUCER)
            total = gl.full((BM * BN,), 0, gl.float32, REDUCER)
        else:
            peer = gl.arange(0, PEERS, layout=gl.SliceLayout(1, REDUCER))
            lane = gl.arange(0, BM * BN, layout=gl.SliceLayout(0, REDUCER))
            elem = peer[:, None] * BM * BN + lane[None, :]
            total = gl.full((PEERS, BM * BN), 0, gl.float32, REDUCER)
        for i in gl.static_range(SPLITS):
            if SPLITS == 16:
                # Reuse one buffer descriptor for all sixteen partials.
                total += gfx1250.buffer_load(scratch, elem + i * TILE_M * BN)
            else:
                total += gfx1250.buffer_load(scratch + i * TILE_M * BN, elem)
        gl.store(
            C + (row + elem // BN) * N + col + elem % BN, total.to(C.type.element_ty)
        )


@gluon.jit(
    repr=make_kernel_repr(
        "_gemm_a16w16_gfx1250_cluster", ["BM", "BN", "BK", "SPLITS", "PEERS", "GRID"]
    )
)
def _gemm_a16w16_cluster_kernel(
    A,
    B,
    C,
    Scratch,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    BM: gl.constexpr,
    BN: gl.constexpr,
    BK: gl.constexpr,
    SPLITS: gl.constexpr,
    PEERS: gl.constexpr,
    GRID: gl.constexpr,
    A_SHARED: gl.constexpr,
    B_SHARED: gl.constexpr,
    WMMA: gl.constexpr,
    REDUCER: gl.constexpr,
):
    TILES: gl.constexpr = M // (PEERS * BM) * (N // BN)
    gl.static_assert(M % (PEERS * BM) == 0 and N % BN == 0 and K % (SPLITS * BK) == 0)
    gl.static_assert(K // (SPLITS * BK) >= 3)
    a_buf = gl.allocate_shared_memory(
        A.type.element_ty, [3, SPLITS, PEERS * BM, BK], A_SHARED
    )
    b_buf = gl.allocate_shared_memory(B.type.element_ty, [3, SPLITS, BN, BK], B_SHARED)
    start = gl.program_id(0)
    if PEERS > 1 or (SPLITS == 16 and GRID == TILES):
        _cluster_splitk_tile(
            A,
            B,
            C,
            Scratch,
            a_buf,
            b_buf,
            start,
            M,
            N,
            K,
            BM,
            BN,
            BK,
            SPLITS,
            PEERS,
            A_SHARED,
            B_SHARED,
            WMMA,
            REDUCER,
        )
    else:
        for tile in range(start, TILES, GRID):
            _cluster_splitk_tile(
                A,
                B,
                C,
                Scratch,
                a_buf,
                b_buf,
                tile,
                M,
                N,
                K,
                BM,
                BN,
                BK,
                SPLITS,
                PEERS,
                A_SHARED,
                B_SHARED,
                WMMA,
                REDUCER,
            )
