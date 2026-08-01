# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""PERF 12 bake-off — DRAM ND-SHARDED weights vs the shipped DRAM-INTERLEAVED weight stream.

ONE test, two variants of the SAME binary (the shard width is read off the tensor, so the
`interleaved` variant is the shipped kernel byte for byte):

    MOE_WSHARD=0   weights DRAM interleaved  -- the honest baseline (what the op ships today)
    MOE_WSHARD=1   weights DRAM ND-sharded   -- shard [TILE, N*TILE] per `weight_memory_configs`
    MOE_WSHARD=g   gate/up sharded, W_down left interleaved (which half carries the win)
    MOE_WSHARD=d   W_down sharded, gate/up left interleaved

Case list and grid come from the environment exactly like the Perf-2 harness:

    MOE_SWIGLU_GRID=11x8 MOE_WSHARD=1 MOE_R2_CASES="7168,5120,256,bf16_rm" \
      scripts/run_safe_pytest.sh --profile <this file>

`MOE_WSHARD_CHECK=1` additionally runs the interleaved reference on the SAME inputs and asserts the
two agree — the correctness gate on the sharded read path (a mis-coalesced run would return the
wrong tiles, and this op's output is a matmul over them, so PCC collapses).

Nothing here imports or edits the shipped op's kernels; it only chooses the weights' PLACEMENT,
which is a caller-side decision the op already reads (`nd_shard_n_tiles`).
"""

import os

import pytest

import ttnn

from ttnn.operations.moe_fused_swiglu import moe_fused_swiglu
from ttnn.operations.moe_fused_swiglu.moe_fused_swiglu_program_descriptor import weight_memory_configs

TILE = 32
HIDDEN = 2048
BFP4_TILE = 576
NUM_GLOBAL_EXPERTS, NUM_LOCAL_EXPERTS, LOCAL_EXPERT_ID, GLOBAL_EXPERT_ID = 256, 8, 3, 137

_FORMATS = {"bf16_rm": (ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT), "bfp8_tile": (ttnn.bfloat8_b, ttnn.TILE_LAYOUT)}

#: The Perf-2 guard set, verbatim, so a number here is comparable with every earlier round's.
GUARD_SET = (
    "7168,5120,128,bf16_rm;"
    "7168,5120,256,bf16_rm;"
    "7168,5120,512,bf16_rm;"
    "7168,1024,256,bf16_rm;"
    "6144,5120,256,bf16_rm;"
    "7168,5120,5120,bf16_rm;"
    "7168,5120,128,bfp8_tile;"
    "7168,5120,256,bfp8_tile;"
    "7168,5120,512,bfp8_tile;"
    "7168,1024,256,bfp8_tile;"
    "6144,5120,256,bfp8_tile;"
    "7168,5120,5120,bfp8_tile"
)

_DEFAULT = "7168,5120,256,bf16_rm"


def _cases():
    spec = os.environ.get("MOE_R2_CASES", _DEFAULT)
    if spec == "guard":
        spec = GUARD_SET
    out = []
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        emb, capacity, count, fmt = part.split(",")
        out.append((int(emb), int(capacity), int(count), fmt.strip()))
    return out


def _weight_mem_configs(device, emb, which):
    """(gate_up, down) memory configs for this variant. `None` == DRAM interleaved."""
    if which == "0":
        return None, None
    gu_mc, dn_mc = weight_memory_configs(device, emb, HIDDEN)
    if which == "g":
        return gu_mc, None
    if which == "d":
        return None, dn_mc
    return gu_mc, dn_mc


def _build(emb, capacity, count, input_format, device, which):
    import torch  # lazy: ttnn/ forbids a global torch import

    torch.manual_seed(42)
    x = torch.randn((1, 1, capacity, emb), dtype=torch.float32)
    if count < capacity:
        x[:, :, count:, :] = 100.0  # hostile sentinel in the phantom rows
    dt, lay = _FORMATS[input_format]
    tt_x = ttnn.from_torch(
        x.to(torch.bfloat16), dtype=dt, layout=lay, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    gu_mc, dn_mc = _weight_mem_configs(device, emb, which)
    torch.manual_seed(7)
    shapes = ((emb, HIDDEN), (emb, HIDDEN), (HIDDEN, emb))
    mcs = (gu_mc, gu_mc, dn_mc)
    tt_w = [
        ttnn.from_torch(
            torch.randn(s, dtype=torch.bfloat16),
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=mc or ttnn.DRAM_MEMORY_CONFIG,
        )
        for s, mc in zip(shapes, mcs)
    ]
    counts = torch.zeros(NUM_GLOBAL_EXPERTS, dtype=torch.int32)
    counts[GLOBAL_EXPERT_ID] = count
    idx = torch.tensor([(11 + 37 * i) % NUM_GLOBAL_EXPERTS for i in range(NUM_LOCAL_EXPERTS)], dtype=torch.int32)
    idx[LOCAL_EXPERT_ID] = GLOBAL_EXPERT_ID
    to_dev = lambda t: ttnn.from_torch(  # noqa: E731
        t, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    return tt_x, tt_w, to_dev(counts), to_dev(idx)


@pytest.mark.parametrize("case", _cases(), ids=lambda c: f"{c[3]}_e{c[0]}_c{c[1]}_n{c[2]}")
def test_wshard(device, case):
    import torch  # lazy: ttnn/ forbids a global torch import

    emb, capacity, count, input_format = case
    which = os.environ.get("MOE_WSHARD", "1")
    tt_x, tt_w, tt_counts, tt_idx = _build(emb, capacity, count, input_format, device, which)
    out = moe_fused_swiglu(tt_x, tt_w[0], tt_w[1], tt_w[2], tt_counts, tt_idx, LOCAL_EXPERT_ID)
    assert list(out.shape) == [1, 1, capacity, emb]

    if os.environ.get("MOE_WSHARD_CHECK") == "1" and which != "0":
        # Same weights, interleaved placement: the reference the sharded read must reproduce. Built
        # from the identical torch seeds, so any disagreement is the read path and nothing else.
        rx, rw, rc, ri = _build(emb, capacity, count, input_format, device, "0")
        ref = moe_fused_swiglu(rx, rw[0], rw[1], rw[2], rc, ri, LOCAL_EXPERT_ID)
        a = ttnn.to_torch(out).float()[0, 0, :count, :]
        b = ttnn.to_torch(ref).float()[0, 0, :count, :]
        if count == 0:
            # `count == 0` defines NO output row, so there is nothing to compare — the assertion is
            # that the dispatch happened at all (m_blocks == 0 on every core, no CB traffic).
            print("[wshard-check] count=0: no defined rows, dispatch-only")
            return
        pcc = torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()
        maxdiff = (a - b).abs().max().item()
        print(f"[wshard-check] pcc_vs_interleaved={pcc:.9f} maxabsdiff={maxdiff:.6g}")
        assert pcc > 0.99999, f"sharded read diverges from interleaved: pcc={pcc}"

    print(f"[wshard] variant={which} {input_format} emb={emb} cap={capacity} count={count}")
