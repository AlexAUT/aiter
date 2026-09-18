"""Per-iteration cycle breakdown of the MLA hot loop from a decoded ATT trace.

Instruction records are [timestamp, type, stall, duration, code_index].
Iterations are delimited by executions of the loop's backward branch.
"""

import glob
import json
import re
import sys
from collections import Counter, defaultdict


def load_code(ui):
    entries = json.load(open(f"{ui}/code.json"))["code"]
    text = {}
    for e in entries:
        text[e[2]] = e[0]
    return text


def hot_loop(exec_count):
    """Decoded branch targets are byte offsets, not labels, so recover the loop
    from the trace itself: the longest contiguous run of heavily executed
    instruction indices. Its last index is the back-edge."""
    if not exec_count:
        return None
    peak = max(exec_count.values())
    hot = sorted(i for i, c in exec_count.items() if c >= peak * 0.5)
    best = cur_lo = prev = hot[0]
    best_hi = hot[0]
    for i in hot[1:]:
        if i - prev > 8:  # tolerate a few rarely-taken slots
            if prev - cur_lo > best_hi - best:
                best, best_hi = cur_lo, prev
            cur_lo = i
        prev = i
    if prev - cur_lo > best_hi - best:
        best, best_hi = cur_lo, prev
    return best, best_hi


def category(ins):
    ins = ins.strip()
    for pfx, name in (
        ("v_wmma", "WMMA (matrix)"),
        ("ds_load", "LDS read"),
        ("ds_store", "LDS write"),
        ("s_wait_dscnt", "wait: LDS"),
        ("s_wait_tensorcnt", "wait: TDM"),
        ("s_wait_loadcnt", "wait: vmem"),
        ("s_wait_kmcnt", "wait: scalar mem"),
        ("s_wait_alu", "wait: ALU dep"),
        ("s_wait_xcnt", "wait: xcnt"),
        ("s_barrier", "barrier"),
        ("tensor_load", "TDM issue"),
        ("v_", "VALU"),
        ("s_", "SALU"),
        ("global_", "global mem"),
        ("buffer_", "global mem"),
    ):
        if ins.startswith(pfx):
            return name
    return "other"


def main(ui, label):
    text = load_code(ui)
    waves = sorted(glob.glob(f"{ui}/se0_*_wv*.json"))

    exec_count = Counter()
    for wf in waves:
        for r in json.load(open(wf))["wave"]["instructions"]:
            exec_count[r[4]] += 1
    loop = hot_loop(exec_count)
    if loop is None:
        sys.exit("no loop found")
    lo, hi = loop
    print(f"hot loop: instruction index {lo}..{hi} "
          f"({hi - lo + 1} instructions, executed ~{exec_count[hi]}x)")

    per_iter_cycles = []
    cat_cycles = Counter()
    cat_stall = Counter()
    top_stall = defaultdict(int)
    top_count = Counter()

    for wf in waves:
        w = json.load(open(wf))["wave"]
        insts = w["instructions"]
        branch_ts = [r[0] for r in insts if r[4] == hi]
        for a, b in zip(branch_ts, branch_ts[1:]):
            if b - a < 100000:  # drop outliers from wave (de)scheduling
                per_iter_cycles.append(b - a)
        for ts, _t, stall, dur, idx in insts:
            if lo <= idx <= hi:
                c = category(text.get(idx, ""))
                cat_cycles[c] += dur
                cat_stall[c] += stall
                if stall:
                    top_stall[idx] += stall
                top_count[c] += 1

    n_iter = len(per_iter_cycles)
    med = sorted(per_iter_cycles)[n_iter // 2] if n_iter else 0
    total = sum(cat_cycles.values())

    print(f"\n=== {label} ===")
    print(f"waves traced: {len(waves)}   loop iterations observed: {n_iter}")
    if n_iter:
        s = sorted(per_iter_cycles)
        print(f"cycles / iteration: median {med}  p10 {s[n_iter // 10]}  "
              f"p90 {s[9 * n_iter // 10]}  min {s[0]}")
    print(f"\n{'category':18s} {'cycles':>12s} {'share':>7s} {'of which stall':>15s} {'count':>8s}")
    for c, v in cat_cycles.most_common():
        print(f"{c:18s} {v:>12,} {100 * v / total:>6.1f}% "
              f"{cat_stall[c]:>15,} {top_count[c]:>8,}")
    print(f"{'TOTAL':18s} {total:>12,}")

    print("\ntop stalling instructions:")
    for idx, st in sorted(top_stall.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {st:>11,} cycles  {text.get(idx, '?').strip()[:72]}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
