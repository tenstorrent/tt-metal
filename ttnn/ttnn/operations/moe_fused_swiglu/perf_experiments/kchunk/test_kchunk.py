# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""PERF 14 bake-off — K-CHUNKED gate/up vs the shipped whole-K single-K-block matmul.

Variants are values of `MOE_SWIGLU_GU_KCHUNKS` (and `MOE_SWIGLU_WG_KPUB`), read by the op at
descriptor-build time, so each variant is ONE process:

    MOE_SWIGLU_GU_KCHUNKS=1                     the shipped shape (baseline, byte-identical)
    MOE_SWIGLU_GU_KCHUNKS=N                     the chunk's matmul as N K-blocks, L1-accumulating
    MOE_SWIGLU_GU_KCHUNKS=N MOE_SWIGLU_WG_KPUB=1  ... plus per-K-sub-block weight publication

Weights are DRAM ND-sharded (the shipped placement) so a number here is comparable with the
Perf-12 baselines. Same seeds and same guard set as `perf_experiments/wshard/test_wshard.py`.

CORRECTNESS IS THE GATE, and it is a TORCH reference, not a self-comparison: the K-blocking changes
the accumulation ORDER and the accumulator's FORMAT handling, so comparing two K-chunk settings
against each other could agree on the same wrong answer. The weights are read BACK off the device
(`ttnn.to_torch`) so the bfp4 quantisation is shared and the only difference left is the kernel.

    MOE_SWIGLU_GRID=11x8 MOE_SWIGLU_GU_KCHUNKS=4 MOE_KC_CASES="7168,5120,256,bf16_rm" \
      scripts/run_safe_pytest.sh --profile <this file>
"""

import os

import pytest

import ttnn

from ttnn.operations.moe_fused_swiglu import moe_fused_swiglu
from ttnn.operations.moe_fused_swiglu.moe_fused_swiglu_program_descriptor import weight_memory_configs

TILE = 32
HIDDEN = 2048
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
    spec = os.environ.get("MOE_KC_CASES", _DEFAULT)
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


def _build(emb, capacity, count, input_format, device):
    """The shipped input shape: ND-sharded bfp4 weights, hostile sentinel in the phantom rows."""
    import torch  # lazy: ttnn/ forbids a global torch import

    torch.manual_seed(42)
    x = torch.randn((1, 1, capacity, emb), dtype=torch.float32)
    if count < capacity:
        x[:, :, count:, :] = 100.0
    dt, lay = _FORMATS[input_format]
    tt_x = ttnn.from_torch(
        x.to(torch.bfloat16), dtype=dt, layout=lay, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    gu_mc, dn_mc = weight_memory_configs(device, emb, HIDDEN)
    torch.manual_seed(7)
    shapes = ((emb, HIDDEN), (emb, HIDDEN), (HIDDEN, emb))
    mcs = (gu_mc, gu_mc, dn_mc)
    tt_w = [
        ttnn.from_torch(
            torch.randn(s, dtype=torch.bfloat16),
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=mc,
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
def test_kchunk(device, case):
    import torch  # lazy: ttnn/ forbids a global torch import

    emb, capacity, count, input_format = case
    tt_x, tt_w, tt_counts, tt_idx = _build(emb, capacity, count, input_format, device)
    out = moe_fused_swiglu(tt_x, tt_w[0], tt_w[1], tt_w[2], tt_counts, tt_idx, LOCAL_EXPERT_ID)
    assert list(out.shape) == [1, 1, capacity, emb]

    kc = os.environ.get("MOE_SWIGLU_GU_KCHUNKS", "1")
    kpub = os.environ.get("MOE_SWIGLU_WG_KPUB", "0")
    if count == 0:
        print(f"[kchunk] kc={kc} kpub={kpub} count=0: no defined rows, dispatch-only")
        return

    # ---- the torch reference, on the DEVICE's own bfp4 weights ----
    xr = ttnn.to_torch(tt_x).float()[0, 0, :count, :]
    wg = ttnn.to_torch(tt_w[0]).float()
    wu = ttnn.to_torch(tt_w[1]).float()
    wd = ttnn.to_torch(tt_w[2]).float()
    h = torch.nn.functional.silu(xr @ wg) * (xr @ wu)
    ref = h @ wd

    got = ttnn.to_torch(out).float()[0, 0, :count, :]
    pcc = torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1].item()
    rel = ((got - ref).abs().max() / ref.abs().max()).item()
    print(f"[kchunk] kc={kc} kpub={kpub} {input_format} e{emb} c{capacity} n{count} pcc={pcc:.6f} maxrel={rel:.4g}")
    # The op is bfp4 weights x bfp8 activations at LoFi, so the reference agreement is ~0.999.
    # This gate is deliberately loose enough to pass the SHIPPED kernel and tight enough that a
    # broken accumulation (wrong K window, unsupported L1-acc format) cannot slip through.
    assert pcc > 0.99, f"kc={kc} diverges from the torch reference: pcc={pcc}"
