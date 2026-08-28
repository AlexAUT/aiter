# SPDX-License-Identifier: MIT
"""Sweep EP fp8 decode configs: layout, TDM, warps, shuffled pipelined path."""

import argparse

import torch

from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.triton.attention.mla import mla_decode_fwd as gluon_mla_decode_fwd
from aiter.test_common import checkAllclose, run_perftest

torch.set_default_device("cuda")

KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM
PAGE_SIZE = 64


def pack_rope_split2_kv(tensor, nope_dim, rope_dim):
    pages, page_size, nhead_kv, head_dim = tensor.shape
    packed = torch.cat(
        (
            tensor[..., :nope_dim].reshape(pages, page_size * nope_dim),
            tensor[..., nope_dim:].reshape(pages, page_size * rope_dim),
        ),
        dim=-1,
    )
    return packed.reshape(pages, page_size, nhead_kv, head_dim).contiguous()


def shuffle_kv_buffer(kv_buffer, kv_lora_rank):
    from op_tests.triton_tests.attention.test_mla import shuffle_kv_buffer as _shuffle

    num_blocks, block_size, num_kv_heads, head_size = kv_buffer.shape
    shuffled = _shuffle(kv_buffer, kv_lora_rank)
    return shuffled.view(num_blocks, num_kv_heads, block_size, head_size).contiguous()


def _ref_mla(q, kv, block_tables, seq_lens, sm_scale, kv_lora_rank):
    """Token-major fp32 reference."""
    batch = seq_lens.numel()
    nhead = q.shape[1]
    out = torch.zeros(q.shape[0], nhead, kv_lora_rank, dtype=torch.float32, device=q.device)
    for b in range(batch):
        ctx = int(seq_lens[b].item())
        pages = block_tables[b]
        for h in range(nhead):
            q_vec = q[b, h].to(torch.float32)
            acc = torch.zeros(kv_lora_rank, dtype=torch.float32, device=q.device)
            m_i = float("-inf")
            l_i = 0.0
            for pos in range(ctx):
                page = pos // PAGE_SIZE
                tok = pos % PAGE_SIZE
                phys = int(pages[page].item())
                k = kv[phys, tok, 0].to(torch.float32)
                k_nope = k[:kv_lora_rank]
                k_rope = k[kv_lora_rank:]
                q_nope = q_vec[:kv_lora_rank]
                q_rope = q_vec[kv_lora_rank:]
                score = sm_scale * (q_nope @ k_nope + q_rope @ k_rope)
                m_ij = max(m_i, score.item())
                alpha = pow(2.0, m_i - m_ij) if m_i > float("-inf") else 0.0
                p = pow(2.0, score.item() - m_ij)
                acc = acc * alpha + p * k_nope
                l_i = l_i * alpha + p
                m_i = m_ij
            out[b, h] = acc / l_i if l_i > 0 else 0.0
    return out


def _make_case(batch, ctx_len, nhead, *, rope_split2=False, shuffled=False, seed=1):
    torch.manual_seed(seed + batch + ctx_len + nhead)
    num_pages_per_batch = (ctx_len + PAGE_SIZE - 1) // PAGE_SIZE
    total_pages = batch * num_pages_per_batch

    qo_indptr = torch.zeros(batch + 1, dtype=torch.int32, device="cuda")
    qo_indptr[1:] = torch.arange(1, batch + 1, dtype=torch.int32, device="cuda")
    block_tables = (
        torch.arange(total_pages, dtype=torch.int32, device="cuda")
        .view(batch, num_pages_per_batch)
        .contiguous()
    )
    kv_token = torch.randn(
        (total_pages, PAGE_SIZE, 1, QK_HEAD_DIM), dtype=torch.bfloat16, device="cuda"
    ).to(dtypes.fp8)
    if rope_split2:
        kv_buffer = pack_rope_split2_kv(kv_token, KV_LORA_RANK, QK_ROPE_HEAD_DIM)
    elif shuffled:
        kv_bf16 = kv_token.to(torch.bfloat16)
        kv_buffer = shuffle_kv_buffer(kv_bf16, KV_LORA_RANK).to(dtypes.fp8)
    else:
        kv_buffer = kv_token.contiguous()
    q = torch.randn((batch, nhead, QK_HEAD_DIM), dtype=torch.bfloat16, device="cuda").to(
        dtypes.fp8
    )
    out = torch.zeros((batch, nhead, KV_LORA_RANK), dtype=torch.bfloat16, device="cuda")
    seq_lens_kv = torch.full((batch,), ctx_len, dtype=torch.int32, device="cuda")
    sm_scale = 1.0 / (QK_HEAD_DIM**0.5)

    def run_fn(**kwargs):
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
            q_descale=torch.tensor([1.0], device="cuda"),
            kv_descale=torch.tensor([1.0], device="cuda"),
            shuffled_kv_cache=shuffled,
            kv_rope_split2=rope_split2,
            **kwargs,
        )

    total_kv = batch * ctx_len
    nbytes = (
        total_kv * QK_HEAD_DIM
        + batch * nhead * QK_HEAD_DIM
        + batch * nhead * KV_LORA_RANK * 2
    )
    return run_fn, out, q, kv_buffer, block_tables, seq_lens_kv, sm_scale, nbytes


def _bench(label, run_fn, nbytes, num_iters, num_warmup):
    _, us = run_perftest(run_fn, num_iters=num_iters, num_warmup=num_warmup)
    return label, us, nbytes / us / 1e6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--ctx", type=int, default=8192)
    parser.add_argument("--nhead", type=int, default=128)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    print(f"gfx={get_gfx()} batch={args.batch} ctx={args.ctx} nhead={args.nhead}")
    configs = [
        ("baseline_token_major", dict(rope_split2=False, shuffled=False), dict(kv_rope_split2=False)),
        ("rope_split2_default", dict(rope_split2=True, shuffled=False), dict()),
        ("rope_split2_tdm", dict(rope_split2=True, shuffled=False), dict(ep_use_tdm=True)),
        ("asm_head_split", dict(rope_split2=True, shuffled=False), dict(ep_asm_head_split=True)),
        ("shuffled_pipelined", dict(rope_split2=False, shuffled=True), dict(kv_rope_split2=False)),
    ]

    rows = []
    for label, case_kw, launch_kw in configs:
        try:
            run_fn, out, q, kv, bt, sl, sm, nbytes = _make_case(
                args.batch, args.ctx, args.nhead, **case_kw
            )
            tag, us, tbs = _bench(
                label, lambda: run_fn(**launch_kw), nbytes, args.iters, args.warmup
            )
            rows.append((tag, us, tbs))
            if args.check and not case_kw.get("shuffled"):
                run_fn(**launch_kw)
                ref = _ref_mla(q, kv, bt, sl, sm, KV_LORA_RANK)
                err = checkAllclose(
                    ref, out.float(), rtol=0.08, atol=0.08, msg=f"check {label}"
                )
                print(f"  check {label}: err={err}")
        except Exception as exc:
            print(f"  {label}: FAILED {exc}")
            rows.append((label, float("nan"), float("nan")))

    print(f"{'config':<24} {'us':>10} {'TB/s':>8}")
    for label, us, tbs in rows:
        print(f"{label:<24} {us:>10.3f} {tbs:>8.3f}")


if __name__ == "__main__":
    main()
