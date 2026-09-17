# The kernels in this file are adapted from vLLM:
# https://github.com/vllm-project/vllm/blob/main/vllm/attention/ops/triton_unified_attention.py

import torch
import triton.experimental.gluon.language as gl
import triton.language as tl
from triton.experimental import gluon

from aiter.ops.triton._gluon_kernels.gfx1250.attention.mla import (
    MLAConfig,
    MLAProgram,
    cdiv_fn,
)
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils.common_utils import strip_annotate
from aiter.ops.triton.utils.types import e4m3_dtype

float8_info = torch.finfo(e4m3_dtype)


@gluon.jit
def prefetch_kv_lora_chunk0(
    pgm: MLAProgram,
    wait_lora: gl.constexpr,
    buffer_id,
):
    chunk_k: gl.constexpr = 128
    gl.amd.gfx1250.tdm.async_wait(wait_lora)
    return pgm.lds_unshuffle_kv_lora(buffer_id).slice(
        0, chunk_k, dim=0
    ).load(layout=pgm.cfg.K_DOT_LAYOUT)


@gluon.jit
def compute_qk_tile_fp8(
    pgm: MLAProgram,
    buffer_id,
    tile_idx,
    qk_factor,
    kv_lora_chunk0_prefetched,
    wait_lora: gl.constexpr,
    wait_rope: gl.constexpr,
    IS_LAST: gl.constexpr,
):
    S = gl.zeros(
        [pgm.cfg.BLOCK_M, pgm.cfg.TILE_SIZE],
        dtype=tl.float32,
        layout=pgm.cfg.QK_WMMA_UNPACKED_LAYOUT,
    )
    chunk_k: gl.constexpr = 128
    tl.static_assert(pgm.cfg.KV_LORA_RANK == (4 * chunk_k))

    q_lora_chunk0 = gl.amd.slice(pgm.q_lora, [pgm.cfg.BLOCK_M, chunk_k], [0, 0])
    kv_lora_chunk0 = kv_lora_chunk0_prefetched
    kv_lora_chunk0 = kv_lora_chunk0.to(q_lora_chunk0.dtype)

    q_lora_chunk1 = gl.amd.slice(pgm.q_lora, [pgm.cfg.BLOCK_M, chunk_k], [0, 128])
    gl.amd.gfx1250.tdm.async_wait(wait_lora)
    kv_lora_chunk1 = pgm.lds_unshuffle_kv_lora(buffer_id).slice(
        128, chunk_k, dim=0
    ).load(layout=pgm.cfg.K_DOT_LAYOUT)
    kv_lora_chunk1 = kv_lora_chunk1.to(q_lora_chunk1.dtype)
    S = gl.amd.gfx1250.wmma(q_lora_chunk0, kv_lora_chunk0, S)

    q_lora_chunk2 = gl.amd.slice(pgm.q_lora, [pgm.cfg.BLOCK_M, chunk_k], [0, 256])
    gl.amd.gfx1250.tdm.async_wait(wait_lora)
    kv_lora_chunk2 = pgm.lds_unshuffle_kv_lora(buffer_id).slice(
        256, chunk_k, dim=0
    ).load(layout=pgm.cfg.K_DOT_LAYOUT)
    kv_lora_chunk2 = kv_lora_chunk2.to(q_lora_chunk2.dtype)
    S = gl.amd.gfx1250.wmma(q_lora_chunk1, kv_lora_chunk1, S)

    q_lora_chunk3 = gl.amd.slice(pgm.q_lora, [pgm.cfg.BLOCK_M, chunk_k], [0, 384])
    gl.amd.gfx1250.tdm.async_wait(wait_lora)
    kv_lora_chunk3 = pgm.lds_unshuffle_kv_lora(buffer_id).slice(
        384, chunk_k, dim=0
    ).load(layout=pgm.cfg.K_DOT_LAYOUT)
    kv_lora_chunk3 = kv_lora_chunk3.to(q_lora_chunk3.dtype)
    S = gl.amd.gfx1250.wmma(q_lora_chunk2, kv_lora_chunk2, S)
    S = gl.amd.gfx1250.wmma(q_lora_chunk3, kv_lora_chunk3, S)

    k_rope = pgm.tdm_shared_load_k_rope(wait_rope, buffer_id)
    S = pgm.compute_qk_rope(k_rope, None, None, S) * qk_factor

    if IS_LAST:
        seq_offset = tile_idx * pgm.cfg.TILE_SIZE + gl.arange(
            0,
            pgm.cfg.TILE_SIZE,
            layout=gl.SliceLayout(0, pgm.cfg.QK_WMMA_UNPACKED_LAYOUT),
        )
        seq_mask = seq_offset[None, :] < pgm.context_len + pgm.query_pos_qk + 1
        S = gl.where(seq_mask, S, float("-inf"))
    return S


@gluon.jit
def finish_tile_fp8(pgm: MLAProgram, S, L, M, acc, buffer_id):
    p, alpha, M = pgm.softmax_part0(S, M)
    p, L, acc = pgm.softmax_part1(p, L, acc, alpha)
    kv_lora_trans = pgm.lds_unshuffle_kv_lora_trans(buffer_id).load(
        layout=pgm.cfg.V_DOT_LAYOUT
    )
    acc = pgm.compute_pkv_lora_trans(p, kv_lora_trans, None, acc)
    return L, M, acc


@gluon.jit
def process_tile_fp8(
    pgm: MLAProgram,
    L,
    M,
    acc,
    buffer_id,
    tile_idx,
    qk_factor,
    kv_lora_chunk0_prefetched,
    wait_lora: gl.constexpr,
    wait_rope: gl.constexpr,
    IS_LAST: gl.constexpr,
):
    S = compute_qk_tile_fp8(
        pgm,
        buffer_id,
        tile_idx,
        qk_factor,
        kv_lora_chunk0_prefetched,
        wait_lora,
        wait_rope,
        IS_LAST,
    )
    return finish_tile_fp8(pgm, S, L, M, acc, buffer_id)


_mla_decode_fwd_kernel_prefetch_repr = make_kernel_repr(
    "_mla_decode_fwd_kernel_prefetch",
    [
        "num_query_heads",
        "num_kv_heads",
        "TILE_SIZE",
        "KV_LORA_RANK",
        "QK_ROPE_HEAD_DIM",
        "BLOCK_Q",
        "BLOCK_M",
        "NUM_SEGMENTS_PER_SEQ",
        "num_warps",
        "num_stages",
        "ALL_DECODE",
        "SHUFFLED_KV_CACHE",
        "QUERY_DTYPE",
        "KV_CACHE_DTYPE",
    ],
)


@gluon.jit(repr=_mla_decode_fwd_kernel_prefetch_repr)
def _mla_decode_fwd_kernel_prefetch(
    segm_output_ptr,
    segm_max_ptr,
    segm_expsum_ptr,
    query_ptr,
    query_scales_ptr,
    kv_buffer_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    SCALE: gl.constexpr,
    q_scale_ptr,
    kv_scale_ptr,
    out_scale_ptr,
    num_query_heads: gl.constexpr,
    num_kv_heads: gl.constexpr,
    block_tables_stride: gl.int64,
    query_stride_0: gl.int64,
    query_stride_1: gl.int64,
    query_scales_stride_0: gl.int64,
    query_scales_stride_1: gl.int64,
    KV_LORA_RANK: gl.constexpr,
    QK_ROPE_HEAD_DIM: gl.constexpr,
    stride_kv_buffer_0: gl.int32,
    stride_kv_buffer_1: gl.int32,
    stride_kv_buffer_2: gl.int32,
    stride_kv_buffer_3: gl.int32,
    query_start_len_ptr,
    num_tokens_per_seq: gl.int32,
    num_blocks: gl.int32,
    TILE_SIZE: gl.constexpr,
    BLOCK_Q: gl.constexpr,
    BLOCK_M: gl.constexpr,
    NUM_SEGMENTS_PER_SEQ: gl.constexpr,
    WARP_SIZE: gl.constexpr,
    num_warps: gl.constexpr,
    num_stages: gl.constexpr,
    SHUFFLED_KV_CACHE: gl.constexpr = True,
    ALL_DECODE: gl.constexpr = False,
    K_WIDTH: gl.constexpr = 16,
    SCALE_K_WIDTH_LORA: gl.constexpr = 16,
    SCALE_K_WIDTH_ROPE: gl.constexpr = 16,
    QUERY_DTYPE: gl.constexpr = "fp8",
    KV_CACHE_DTYPE: gl.constexpr = "fp8",
    BLOCK_SCALES_SIZE: gl.constexpr = 16,
    NUM_HEAD_BLOCKS: gl.constexpr = 1,
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    assert SHUFFLED_KV_CACHE
    assert num_stages == 4
    assert QUERY_DTYPE == "fp8"
    assert KV_CACHE_DTYPE == "fp8"
    assert query_scales_ptr is None
    assert query_scales_stride_0 == 0
    assert query_scales_stride_1 == 0

    cfg = MLAConfig(
        KV_LORA_RANK,
        QK_ROPE_HEAD_DIM,
        TILE_SIZE,
        1,
        NUM_SEGMENTS_PER_SEQ,
        BLOCK_M,
        BLOCK_Q,
        num_query_heads,
        num_kv_heads,
        num_warps,
        WARP_SIZE,
        num_stages,
        SCALE,
        False,
        False,
        SHUFFLED_KV_CACHE,
        QUERY_DTYPE,
        KV_CACHE_DTYPE,
        ALL_DECODE,
        K_WIDTH,
        SCALE_K_WIDTH_LORA,
        SCALE_K_WIDTH_ROPE,
        BLOCK_SCALES_SIZE,
    )

    q_block_global_idx = gl.program_id(0)
    kv_head_idx = gl.program_id(1)
    segm_idx = gl.program_id(2)

    num_token_blocks_per_seq = cdiv_fn(num_tokens_per_seq, BLOCK_Q)
    num_q_blocks_per_seq = num_token_blocks_per_seq * NUM_HEAD_BLOCKS

    if cfg.ALL_DECODE:
        seq_idx = q_block_global_idx // NUM_HEAD_BLOCKS
    else:
        seq_idx = q_block_global_idx // num_q_blocks_per_seq
    q_block_local_idx = q_block_global_idx - seq_idx * num_q_blocks_per_seq
    q_start_idx = gl.load(query_start_len_ptr + seq_idx)

    token_q_block_local_idx = q_block_local_idx // NUM_HEAD_BLOCKS
    head_block_idx = q_block_local_idx % NUM_HEAD_BLOCKS
    head_offset = head_block_idx * BLOCK_M

    seq_len = gl.load(seq_lens_ptr + seq_idx)
    num_segments = NUM_SEGMENTS_PER_SEQ
    tiles_per_segment = cdiv_fn(seq_len, num_segments * TILE_SIZE)
    if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
        return

    qk_factor: gl.float32 = cfg.QK_SCALE
    out_factor: gl.float32 = 1.0
    if q_scale_ptr is not None:
        qk_factor = qk_factor * gl.load(q_scale_ptr)
    if kv_scale_ptr is not None:
        kv_scale = gl.load(kv_scale_ptr)
        qk_factor = qk_factor * kv_scale
        out_factor = kv_scale
    if out_scale_ptr is not None:
        out_factor = out_factor / tl.load(out_scale_ptr)

    context_len = seq_len - num_tokens_per_seq
    block_tables_ptr_shifted = block_tables_ptr + seq_idx * block_tables_stride

    q_lora_shared = gl.allocate_shared_memory(
        query_ptr.type.element_ty,
        shape=[BLOCK_M, KV_LORA_RANK],
        layout=cfg.Q_LORA_SHARED_LAYOUT,
    )
    q_rope_shared = gl.allocate_shared_memory(
        query_ptr.type.element_ty,
        shape=[BLOCK_M, QK_ROPE_HEAD_DIM],
        layout=cfg.Q_ROPE_SHARED_LAYOUT,
    )

    offs_q_m_lora = gl.arange(
        0, BLOCK_M, layout=gl.SliceLayout(1, cfg.Q_LORA_LOAD_LAYOUT)
    )
    offs_q_d_lora = gl.arange(
        0, KV_LORA_RANK, layout=gl.SliceLayout(0, cfg.Q_LORA_LOAD_LAYOUT)
    )
    offs_q_m_rope = gl.arange(
        0, BLOCK_M, layout=gl.SliceLayout(1, cfg.Q_ROPE_LOAD_LAYOUT)
    )
    offs_q_d_rope = gl.arange(
        0, QK_ROPE_HEAD_DIM, layout=gl.SliceLayout(0, cfg.Q_ROPE_LOAD_LAYOUT)
    )

    query_pos_lora = (
        token_q_block_local_idx * BLOCK_Q + offs_q_m_lora // cfg.NUM_QUERIES_PER_KV
    )
    query_offset_0_lora = q_start_idx + query_pos_lora
    query_offset_1_lora = (
        kv_head_idx * cfg.NUM_QUERIES_PER_KV
        + head_offset
        + offs_q_m_lora % cfg.NUM_QUERIES_PER_KV
    )
    query_offset_lora = (
        query_offset_0_lora[:, None] * query_stride_0
        + query_offset_1_lora[:, None] * query_stride_1
    )
    query_mask_0_lora = query_pos_lora < num_tokens_per_seq
    query_mask_1_lora = query_offset_1_lora < num_query_heads

    Q_lora_load = gl.load(
        query_ptr + query_offset_lora + offs_q_d_lora[None, :],
        mask=query_mask_0_lora[:, None] & query_mask_1_lora[:, None],
        other=0.0,
    )
    q_lora_shared.store(Q_lora_load)
    Q_lora = q_lora_shared.load(layout=cfg.Q_DOT_LAYOUT)

    query_pos_rope = (
        token_q_block_local_idx * BLOCK_Q + offs_q_m_rope // cfg.NUM_QUERIES_PER_KV
    )
    query_offset_0_rope = q_start_idx + query_pos_rope
    query_offset_1_rope = (
        kv_head_idx * cfg.NUM_QUERIES_PER_KV
        + head_offset
        + offs_q_m_rope % cfg.NUM_QUERIES_PER_KV
    )
    query_offset_rope = (
        query_offset_0_rope[:, None] * query_stride_0
        + query_offset_1_rope[:, None] * query_stride_1
    )
    query_mask_0_rope = query_pos_rope < num_tokens_per_seq
    query_mask_1_rope = query_offset_1_rope < num_query_heads

    Q_rope_load = gl.load(
        query_ptr + query_offset_rope + (KV_LORA_RANK + offs_q_d_rope)[None, :],
        mask=query_mask_0_rope[:, None] & query_mask_1_rope[:, None],
        other=0.0,
    )
    q_rope_shared.store(Q_rope_load)
    Q_rope = q_rope_shared.load(layout=cfg.Q_DOT_LAYOUT)

    offs_q_m_qk = gl.arange(
        0, BLOCK_M, layout=gl.SliceLayout(1, cfg.QK_WMMA_UNPACKED_LAYOUT)
    )
    query_pos_qk = (
        token_q_block_local_idx * BLOCK_Q + offs_q_m_qk // cfg.NUM_QUERIES_PER_KV
    )
    query_offset_0_qk = q_start_idx + query_pos_qk
    query_offset_1_qk = (
        kv_head_idx * cfg.NUM_QUERIES_PER_KV
        + head_offset
        + offs_q_m_qk % cfg.NUM_QUERIES_PER_KV
    )
    query_mask_0_qk = query_pos_qk < num_tokens_per_seq
    query_mask_1_qk = query_offset_1_qk < num_query_heads

    query_offset_0_pv = gl.convert_layout(
        query_offset_0_qk, layout=gl.SliceLayout(1, cfg.PV_WMMA_LAYOUT)
    )
    query_offset_1_pv = gl.convert_layout(
        query_offset_1_qk, layout=gl.SliceLayout(1, cfg.PV_WMMA_LAYOUT)
    )
    query_mask_0_pv = gl.convert_layout(
        query_mask_0_qk, layout=gl.SliceLayout(1, cfg.PV_WMMA_LAYOUT)
    )
    query_mask_1_pv = gl.convert_layout(
        query_mask_1_qk, layout=gl.SliceLayout(1, cfg.PV_WMMA_LAYOUT)
    )

    max_seq_prefix_len = (
        context_len
        + token_q_block_local_idx * BLOCK_Q
        + (BLOCK_M - 1) // cfg.NUM_QUERIES_PER_KV
        + 1
    )
    max_seq_prefix_len = gl.minimum(max_seq_prefix_len, seq_len)

    pgm: MLAProgram = MLAProgram.initialize(
        cfg,
        Q_lora,
        Q_rope,
        kv_buffer_ptr,
        segm_output_ptr,
        segm_max_ptr,
        segm_expsum_ptr,
        max_seq_prefix_len,
        token_q_block_local_idx,
        num_q_blocks_per_seq,
        context_len,
        kv_head_idx,
        num_blocks,
        query_pos_qk,
        query_offset_0_qk,
        query_offset_1_qk,
        query_mask_0_qk,
        query_mask_1_qk,
        query_offset_0_pv,
        query_offset_1_pv,
        query_mask_0_pv,
        query_mask_1_pv,
        segm_idx,
        tiles_per_segment,
        stride_kv_buffer_0,
        stride_kv_buffer_1,
        stride_kv_buffer_2,
        stride_kv_buffer_3,
    )

    L, M, acc = pgm.allocate_accumulator()

    j_hbm_start: gl.int32 = segm_idx * tiles_per_segment
    num_tiles_this_seg: gl.int32 = pgm.tile_end - pgm.tile_start
    j_hbm: gl.int32 = 0
    slot_iter: gl.int32 = 0

    j_hbm, physical_block_idx = pgm.load_physical_block_idx(
        j_hbm, block_tables_ptr_shifted, j_hbm_start
    )
    row_offsets = pgm.get_kv_buffer_row_offsets(physical_block_idx)
    pgm.tdm_load_global_to_shared_kv_lora(row_offsets, 0)
    pgm.tdm_load_global_to_shared_k_rope(row_offsets, 0)

    if num_tiles_this_seg > 2:
        j_hbm, next_physical_block_idx = pgm.load_physical_block_idx_with_mod(
            j_hbm, block_tables_ptr_shifted, j_hbm_start, num_tiles_this_seg
        )
        row_offsets = pgm.get_kv_buffer_row_offsets(next_physical_block_idx)
        pgm.tdm_load_global_to_shared_kv_lora(row_offsets, 1)
        pgm.tdm_load_global_to_shared_k_rope(row_offsets, 1)

        j_hbm, next_next_physical_block_idx = pgm.load_physical_block_idx_with_mod(
            j_hbm, block_tables_ptr_shifted, j_hbm_start, num_tiles_this_seg
        )
        row_offsets = pgm.get_kv_buffer_row_offsets(next_next_physical_block_idx)
        pgm.tdm_load_global_to_shared_kv_lora(row_offsets, 2)
        pgm.tdm_load_global_to_shared_k_rope(row_offsets, 2)

        j_hbm, future_physical_block_idx = pgm.load_physical_block_idx_with_mod(
            j_hbm, block_tables_ptr_shifted, j_hbm_start, num_tiles_this_seg
        )
        read_buffer_id = slot_iter % 4
        prefetched_chunk0 = prefetch_kv_lora_chunk0(
            pgm, wait_lora=7, buffer_id=read_buffer_id
        )
        for tile_idx in range(pgm.tile_start, pgm.tile_end - 3):
            read_buffer_id = slot_iter % 4
            next_read_buffer_id = (slot_iter + 1) % 4
            fill_buffer_id = (slot_iter + 3) % 4
            row_offsets = pgm.get_kv_buffer_row_offsets(future_physical_block_idx)
            pgm.tdm_load_global_to_shared_kv_lora(row_offsets, fill_buffer_id)
            pgm.tdm_load_global_to_shared_k_rope(row_offsets, fill_buffer_id)
            next_prefetched_chunk0 = prefetch_kv_lora_chunk0(
                pgm,
                wait_lora=7,
                buffer_id=next_read_buffer_id,
            )
            L, M, acc = process_tile_fp8(
                pgm,
                L,
                M,
                acc,
                read_buffer_id,
                tile_idx,
                qk_factor,
                prefetched_chunk0,
                wait_lora=7,
                wait_rope=6,
                IS_LAST=False,
            )

            j_hbm, future_physical_block_idx = pgm.load_physical_block_idx_with_mod(
                j_hbm,
                block_tables_ptr_shifted,
                j_hbm_start,
                num_tiles_this_seg,
            )
            slot_iter = slot_iter + 1
            prefetched_chunk0 = next_prefetched_chunk0

        read_buffer_id = slot_iter % 4
        L, M, acc = process_tile_fp8(
            pgm,
            L,
            M,
            acc,
            read_buffer_id,
            pgm.tile_end - 3,
            qk_factor,
            prefetched_chunk0,
            wait_lora=5,
            wait_rope=4,
            IS_LAST=False,
        )
        slot_iter = slot_iter + 1
        read_buffer_id = slot_iter % 4
        prefetched_chunk0 = prefetch_kv_lora_chunk0(
            pgm, wait_lora=3, buffer_id=read_buffer_id
        )
        L, M, acc = process_tile_fp8(
            pgm,
            L,
            M,
            acc,
            read_buffer_id,
            pgm.tile_end - 2,
            qk_factor,
            prefetched_chunk0,
            wait_lora=3,
            wait_rope=2,
            IS_LAST=False,
        )
        slot_iter = slot_iter + 1
        read_buffer_id = slot_iter % 4
        prefetched_chunk0 = prefetch_kv_lora_chunk0(
            pgm, wait_lora=1, buffer_id=read_buffer_id
        )
        L, M, acc = process_tile_fp8(
            pgm,
            L,
            M,
            acc,
            read_buffer_id,
            pgm.tile_end - 1,
            qk_factor,
            prefetched_chunk0,
            wait_lora=1,
            wait_rope=0,
            IS_LAST=True,
        )
    elif num_tiles_this_seg > 1:
        j_hbm, next_physical_block_idx = pgm.load_physical_block_idx_with_mod(
            j_hbm, block_tables_ptr_shifted, j_hbm_start, num_tiles_this_seg
        )
        row_offsets = pgm.get_kv_buffer_row_offsets(next_physical_block_idx)
        pgm.tdm_load_global_to_shared_kv_lora(row_offsets, 1)
        pgm.tdm_load_global_to_shared_k_rope(row_offsets, 1)
        prefetched_chunk0 = prefetch_kv_lora_chunk0(pgm, wait_lora=3, buffer_id=0)
        L, M, acc = process_tile_fp8(
            pgm,
            L,
            M,
            acc,
            0,
            pgm.tile_end - 2,
            qk_factor,
            prefetched_chunk0,
            wait_lora=3,
            wait_rope=2,
            IS_LAST=False,
        )
        prefetched_chunk0 = prefetch_kv_lora_chunk0(pgm, wait_lora=1, buffer_id=1)
        L, M, acc = process_tile_fp8(
            pgm,
            L,
            M,
            acc,
            1,
            pgm.tile_end - 1,
            qk_factor,
            prefetched_chunk0,
            wait_lora=1,
            wait_rope=0,
            IS_LAST=True,
        )
    else:
        prefetched_chunk0 = prefetch_kv_lora_chunk0(pgm, wait_lora=1, buffer_id=0)
        L, M, acc = process_tile_fp8(
            pgm,
            L,
            M,
            acc,
            0,
            pgm.tile_end - 1,
            qk_factor,
            prefetched_chunk0,
            wait_lora=1,
            wait_rope=0,
            IS_LAST=True,
        )

    if cfg.NUM_SEGMENTS_PER_SEQ == 1:
        one_over_L = gl.convert_layout(1.0 / L[:, None], layout=cfg.PV_WMMA_LAYOUT)
        acc *= one_over_L
    acc *= out_factor
    if cfg.NUM_SEGMENTS_PER_SEQ == 1 and segm_output_ptr.type.element_ty.is_fp8():
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)
    pgm.store_output_3D(acc, M, L, segm_idx)
