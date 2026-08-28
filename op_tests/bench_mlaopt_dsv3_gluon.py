# SPDX-License-Identifier: MIT
# Quick DSv3 Gluon MLA decode A/B benchmark (main vs mlaOpt).

import argparse
import subprocess
import sys

import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.triton.attention.mla import mla_decode_fwd as gluon_mla_decode_fwd
from aiter.test_common import run_perftest

torch.set_default_device("cuda")

KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM
PAGE_SIZE = 64


def _pack_rope_split2_kv(tensor, nope_dim, rope_dim):
    pages, page_size, nhead_kv, head_dim = tensor.shape
    packed = torch.cat(
        (
            tensor[..., :nope_dim].reshape(pages, page_size * nope_dim),
            tensor[..., nope_dim:].reshape(pages, page_size * rope_dim),
        ),
        dim=-1,
    )
    return packed.reshape(pages, page_size, nhead_kv, head_dim).contiguous()


def _make_case(batch, ctx_len, nhead, decode_qlen=1, seed=20260513):
    torch.manual_seed(seed + batch + ctx_len + nhead)
    num_pages_per_batch = (ctx_len + PAGE_SIZE - 1) // PAGE_SIZE
    total_pages = batch * num_pages_per_batch

    qo_indptr = torch.zeros(batch + 1, dtype=torch.int32, device="cuda")
    seq_lens_qo = torch.full((batch,), decode_qlen, dtype=torch.int32, device="cuda")
    qo_indptr[1:] = torch.cumsum(seq_lens_qo, dim=0)
    total_q = int(qo_indptr[-1].item())

    kv_token = torch.full(
        (total_pages, PAGE_SIZE, 1, QK_HEAD_DIM),
        0.25,
        dtype=dtypes.fp8,
        device="cuda",
    )
    kv_buffer = _pack_rope_split2_kv(kv_token, KV_LORA_RANK, QK_ROPE_HEAD_DIM)
    assert kv_buffer.shape[2] == 1, "EP decode benchmark expects num_kv_heads=1"
    block_tables = (
        torch.arange(total_pages, dtype=torch.int32, device="cuda")
        .view(batch, num_pages_per_batch)
        .contiguous()
    )
    q = torch.full(
        (total_q, nhead, QK_HEAD_DIM), 0.25, dtype=dtypes.fp8, device="cuda"
    )
    out = torch.zeros((total_q, nhead, KV_LORA_RANK), dtype=torch.bfloat16, device="cuda")
    seq_lens_kv = torch.full((batch,), ctx_len, dtype=torch.int32, device="cuda")
    q_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    kv_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    sm_scale = 1.0 / (QK_HEAD_DIM**0.5)

    def run_fn():
        gluon_mla_decode_fwd(
            q,
            kv_buffer,
            out,
            cu_seqlens_q=qo_indptr,
            seqused_k=seq_lens_kv,
            max_seqlen_kv=ctx_len,
            block_tables=block_tables,
            softmax_scale=sm_scale,
            kv_lora_rank=KV_LORA_RANK,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            causal=True,
            q_descale=q_scale,
            kv_descale=kv_scale,
            shuffled_kv_cache=False,
            kv_rope_split2=True,
        )

    total_kv = batch * ctx_len
    nbytes = (
        total_kv * QK_HEAD_DIM
        + total_q * nhead * QK_HEAD_DIM
        + total_q * nhead * KV_LORA_RANK * 2
    )
    return run_fn, nbytes


def git_rev():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            text=True,
            cwd=aiter.__path__[0] + "/..",
        ).strip()
    except Exception:
        return "unknown"


def git_short():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            cwd=aiter.__path__[0] + "/..",
        ).strip()
    except Exception:
        return "unknown"


def bench_one(batch, ctx_len, nhead, decode_qlen, iters, warmup):
    run_fn, nbytes = _make_case(batch, ctx_len, nhead, decode_qlen)
    _, us = run_perftest(run_fn, num_iters=iters, num_warmup=warmup)
    return us, nbytes / us / 1e6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    gfx = get_gfx()
    if gfx != "gfx1250":
        print(f"skip: need gfx1250, got {gfx}", file=sys.stderr)
        return

    shapes = [
        (64, 512, 128, 1),
        (64, 8192, 128, 1),
        (128, 512, 128, 1),
        (128, 8192, 128, 1),
    ]

    print(f"branch={git_rev()} sha={git_short()} gfx={gfx}")
    print(f"{'batch':>5} {'ctx':>6} {'nhead':>5} {'qlen':>4} {'us':>10} {'TB/s':>8}")
    for batch, ctx_len, nhead, decode_qlen in shapes:
        us, tbs = bench_one(
            batch, ctx_len, nhead, decode_qlen, args.iters, args.warmup
        )
        print(
            f"{batch:5d} {ctx_len:6d} {nhead:5d} {decode_qlen:4d} {us:10.3f} {tbs:8.3f}"
        )


if __name__ == "__main__":
    main()
