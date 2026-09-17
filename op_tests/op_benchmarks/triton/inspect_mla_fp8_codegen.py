#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Compile the gfx1250 FP8 MLA decode kernel AOT and report codegen stats.

This script is intended for codegen iteration only (no benchmark run).
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
from dataclasses import dataclass

import torch

from aiter.ops.triton.utils.types import e4m3_dtype


@dataclass
class LoopRegion:
    start: int
    end: int
    label: str
    size: int


def _load_kernel(module_path: str, symbol: str):
    mod = importlib.import_module(module_path)
    return getattr(mod, symbol)


def _find_largest_backward_loop(asm_lines: list[str]) -> LoopRegion | None:
    label_to_line: dict[str, int] = {}
    label_pat = re.compile(r"^(\.LBB\d+_\d+):")
    branch_pat = re.compile(r"\bs_(?:cbranch_[a-z0-9_]+|branch)\s+(\.LBB\d+_\d+)\b")

    for idx, line in enumerate(asm_lines):
        m = label_pat.match(line.strip())
        if m:
            label_to_line[m.group(1)] = idx

    best: LoopRegion | None = None
    for idx, line in enumerate(asm_lines):
        m = branch_pat.search(line)
        if not m:
            continue
        label = m.group(1)
        if label not in label_to_line:
            continue
        start = label_to_line[label]
        if start >= idx:
            continue
        size = idx - start + 1
        candidate = LoopRegion(start=start, end=idx, label=label, size=size)
        if best is None or candidate.size > best.size:
            best = candidate
    return best


def _count(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text, flags=re.MULTILINE))


def _extract_amdgcn_numeric(asm_text: str, key: str) -> int | None:
    m = re.search(rf"^\s*\.{re.escape(key)}:\s*([0-9]+)\s*$", asm_text, flags=re.MULTILINE)
    if not m:
        return None
    return int(m.group(1))


def _collect_counts(asm_text: str) -> tuple[dict[str, int], dict[str, int], LoopRegion | None]:
    lines = asm_text.splitlines()
    loop = _find_largest_backward_loop(lines)
    loop_text = "\n".join(lines[loop.start : loop.end + 1]) if loop else ""

    pats = {
        "s_barrier_signal": r"\bs_barrier_signal\b",
        "s_barrier_wait": r"\bs_barrier_wait\b",
        "s_wait_dscnt": r"\bs_wait_dscnt\b",
        "s_wait_dscnt_0": r"\bs_wait_dscnt\s+0x0\b",
        "s_wait_tensorcnt": r"\bs_wait_tensorcnt\b",
        "tensor_load_to_lds": r"\btensor_load_to_lds\b",
        "ds_load": r"\bds_load(?:_[a-z0-9_]+)?\b",
        "v_wmma": r"\bv_wmma_[a-z0-9_]+\b",
        "s_set_vgpr_msb": r"\bs_set_vgpr_msb\b",
        "scratch_load": r"\bscratch_load\b",
        "scratch_store": r"\bscratch_store\b",
    }
    total_counts = {name: _count(pat, asm_text) for name, pat in pats.items()}
    loop_counts = {name: _count(pat, loop_text) for name, pat in pats.items()}
    return total_counts, loop_counts, loop


def _build_inputs(
    batch_size: int,
    block_size: int,
    max_ctx_len: int,
    num_query_heads: int,
    num_kv_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    num_blocks: int,
):
    head_dim = kv_lora_rank + qk_rope_head_dim
    max_num_blocks_per_seq = (max_ctx_len + block_size - 1) // block_size

    # decode_qlen=1 configuration.
    q = torch.empty((batch_size, num_query_heads, head_dim), dtype=e4m3_dtype, device="cuda")
    kv = torch.empty(
        (num_blocks, num_kv_heads, block_size, head_dim),
        dtype=e4m3_dtype,
        device="cuda",
    )
    out = torch.empty(
        (batch_size, num_query_heads, 4, kv_lora_rank),
        dtype=torch.float32,
        device="cuda",
    )
    segm_max = torch.empty((batch_size, num_query_heads, 4), dtype=torch.float32, device="cuda")
    segm_expsum = torch.empty_like(segm_max)
    block_tables = torch.zeros(
        (batch_size, max_num_blocks_per_seq), dtype=torch.int32, device="cuda"
    )
    seq_lens = torch.full((batch_size,), max_ctx_len, dtype=torch.int32, device="cuda")
    query_start_len = torch.arange(batch_size + 1, dtype=torch.int32, device="cuda")
    fp_scale = torch.ones((1,), dtype=torch.float32, device="cuda")

    return {
        "q": q,
        "kv": kv,
        "out": out,
        "segm_max": segm_max,
        "segm_expsum": segm_expsum,
        "block_tables": block_tables,
        "seq_lens": seq_lens,
        "query_start_len": query_start_len,
        "fp_scale": fp_scale,
    }


def main():
    parser = argparse.ArgumentParser(description="AOT compile and inspect MLA FP8 decode kernel")
    parser.add_argument(
        "--kernel-module",
        default="aiter.ops.triton._gluon_kernels.gfx1250.attention.mla",
        help="Python module containing the kernel symbol",
    )
    parser.add_argument("--kernel-symbol", default="_mla_decode_fwd_kernel")
    parser.add_argument("--dump-dir", default="/tmp/mladump")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    os.makedirs(args.dump_dir, exist_ok=True)
    os.environ["TRITON_DUMP_DIR"] = args.dump_dir

    kernel = _load_kernel(args.kernel_module, args.kernel_symbol)
    tensors = _build_inputs(
        batch_size=256,
        block_size=64,
        max_ctx_len=16384,
        num_query_heads=128,
        num_kv_heads=1,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        num_blocks=65536,
    )

    compiled = kernel.warmup(
        segm_output_ptr=tensors["out"],
        segm_max_ptr=tensors["segm_max"],
        segm_expsum_ptr=tensors["segm_expsum"],
        query_ptr=tensors["q"],
        query_scales_ptr=None,
        kv_buffer_ptr=tensors["kv"],
        block_tables_ptr=tensors["block_tables"],
        seq_lens_ptr=tensors["seq_lens"],
        SCALE=1.0 / (576.0**0.5),
        q_scale_ptr=tensors["fp_scale"],
        kv_scale_ptr=tensors["fp_scale"],
        out_scale_ptr=None,
        num_query_heads=128,
        num_kv_heads=1,
        block_tables_stride=tensors["block_tables"].stride(0),
        query_stride_0=tensors["q"].stride(0),
        query_stride_1=tensors["q"].stride(1),
        query_scales_stride_0=0,
        query_scales_stride_1=0,
        KV_LORA_RANK=512,
        QK_ROPE_HEAD_DIM=64,
        stride_kv_buffer_0=tensors["kv"].stride(0),
        stride_kv_buffer_1=tensors["kv"].stride(1),
        stride_kv_buffer_2=tensors["kv"].stride(2),
        stride_kv_buffer_3=tensors["kv"].stride(3),
        query_start_len_ptr=tensors["query_start_len"],
        num_tokens_per_seq=1,
        num_blocks=65536,
        TILE_SIZE=64,
        BLOCK_Q=1,
        BLOCK_M=128,
        NUM_SEGMENTS_PER_SEQ=4,
        WARP_SIZE=32,
        num_warps=4,
        num_stages=4,
        SHUFFLED_KV_CACHE=True,
        ALL_DECODE=True,
        K_WIDTH=16,
        SCALE_K_WIDTH_LORA=4,
        SCALE_K_WIDTH_ROPE=4,
        QUERY_DTYPE="fp8",
        KV_CACHE_DTYPE="fp8",
        BLOCK_SCALES_SIZE=16,
        NUM_HEAD_BLOCKS=1,
        waves_per_eu=1,
        grid=(256, 1, 4),
    )

    asm_text = compiled.asm["amdgcn"]
    total_counts, loop_counts, loop = _collect_counts(asm_text)

    metadata = compiled.metadata._asdict()
    vgpr_count = metadata.get("vgpr_count")
    vgpr_spill_count = metadata.get("vgpr_spill_count")
    sgpr_count = metadata.get("sgpr_count")
    sgpr_spill_count = metadata.get("sgpr_spill_count")
    private_segment_fixed_size = metadata.get("private_segment_fixed_size")

    if vgpr_count is None:
        vgpr_count = _extract_amdgcn_numeric(asm_text, "vgpr_count")
    if vgpr_spill_count is None:
        vgpr_spill_count = _extract_amdgcn_numeric(asm_text, "vgpr_spill_count")
    if sgpr_count is None:
        sgpr_count = _extract_amdgcn_numeric(asm_text, "sgpr_count")
    if sgpr_spill_count is None:
        sgpr_spill_count = _extract_amdgcn_numeric(asm_text, "sgpr_spill_count")
    if private_segment_fixed_size is None:
        private_segment_fixed_size = _extract_amdgcn_numeric(
            asm_text, "private_segment_fixed_size"
        )

    report = {
        "kernel_name": compiled.name,
        "kernel_module": args.kernel_module,
        "kernel_symbol": args.kernel_symbol,
        "metadata": {
            "arch": metadata.get("arch"),
            "shared": metadata.get("shared"),
            "num_warps": metadata.get("num_warps"),
            "num_stages": metadata.get("num_stages"),
            "vgpr_count": vgpr_count,
            "vgpr_spill_count": vgpr_spill_count,
            "sgpr_count": sgpr_count,
            "sgpr_spill_count": sgpr_spill_count,
            "private_segment_fixed_size": private_segment_fixed_size,
        },
        "largest_backward_loop": (
            {
                "label": loop.label,
                "line_start": loop.start,
                "line_end": loop.end,
                "size_lines": loop.size,
            }
            if loop
            else None
        ),
        "total_counts": total_counts,
        "largest_loop_counts": loop_counts,
    }

    print(json.dumps(report, indent=2, sort_keys=True))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
