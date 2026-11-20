# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bake-off: split the ROW_MAJOR bank-run fill between the reader (NCRISC, NoC0) and the writer (BRISC, NoC1)
(perf experiment, round 17, `brf_noc_split`).

Re-run (from the repo root): ttnn/ttnn/operations/conv2d_3x3/perf_experiments/brf_noc_split/run.sh <label> 35,33,41 base,split
(add BRF_ORDER=alt to alternate the variant order per case)
(runs/<label>.txt = device ns per case|variant, runs/<label>.jsonl = PCC / bit-identity / blocking,
runs/<label>_dev.csv.gz = the device profile for core_spans.py).

Variants:
  base   op_base/ (unmodified copy of the working-tree op)
  split  op_copy/ (the recommended split fill; integration.patch = diff -ru op_base op_copy)
  <name> op_variants/ with the knobs in VARIANTS (the copy the variant menu was measured on)

Env: BRF_CASES, BRF_VARIANTS (run in that order), BRF_ORDER=alt (reverse the order on every other case), BRF_LOG,
BRF_ITERS (measured calls, default 1). Each variant's calls are signposted `cmp|<case>|<variant>|<i>`. Every variant's
output is compared bit-exactly with the first variant's output of the same case and its PCC vs torch is recorded.
Precision contract fixed: bf16, HiFi4, fp32 DEST, no approx, bias.
"""

import importlib
import json
import os
import time
from pathlib import Path

import pytest
import ttnn
from tracy import signpost

HERE = Path(__file__).parent
_BASE = "ttnn.operations.conv2d_3x3.perf_experiments.brf_noc_split."
MODS = {}
for _k in ("op_base", "op_copy", "op_variants"):
    MODS[_k] = (
        importlib.import_module(_BASE + _k + ".conv2d_3x3"),
        importlib.import_module(_BASE + _k + ".conv2d_3x3_program_descriptor"),
    )
    assert Path(MODS[_k][1].KERNEL_DIR).parent.name == _k, MODS[_k][1].KERNEL_DIR

# FLUX.2 bh2x4 cases (test_flux_conv3x3_compare.py FLUX_CMP_CASES index -> (H, W, C_in, C_out)); padding (0, 0, 1, 1).
FLUX = {
    31: (66, 128, 32, 128),
    32: (130, 256, 32, 128),
    33: (514, 1024, 128, 3),
    34: (66, 128, 128, 512),
    35: (1026, 2048, 128, 3),
    36: (514, 1024, 32, 128),
    37: (258, 512, 64, 256),
    38: (130, 256, 128, 512),
    39: (514, 1024, 64, 128),
    40: (258, 512, 128, 256),
    41: (1026, 2048, 32, 128),
    42: (514, 1024, 64, 256),
    43: (258, 512, 128, 512),
    44: (1026, 2048, 64, 128),
    45: (514, 1024, 128, 256),
    46: (1026, 2048, 64, 256),
    47: (514, 1024, 128, 512),
}
PAD = (0, 0, 1, 1)
BAND_ROWS = 32

# Module constants of each package's descriptor, reset before every variant.
KNOB_DEFAULTS = {
    "op_base": {},
    # op_copy: the recommended split (dynamic front/back claims, half the staging each) = the integration patch.
    "op_copy": dict(BRF_SPLIT=True, BRF_NSTG=4, BRF_CHUNK_BYTES=16384),
    # op_variants: the knob-laden copy the menu was measured on.
    "op_variants": dict(
        BRF_SPLIT=True,
        BRF_SPLIT_P=2,
        BRF_SPLIT_W=1,
        BRF_SPLIT_STAGE="half",
        BRF_DEBUG=False,
        BRF_NSTG=4,
        BRF_DYN=False,
        BRF_CHUNK_BYTES=16384,
        BRF_STORE_SERVE=False,
    ),
}
VARIANTS = {
    "base": ("op_base", {}),
    "split": ("op_copy", {}),  # recommended
    "nosplit": ("op_copy", dict(BRF_SPLIT=False)),
    # Menu (op_variants): static chunk parity (writer takes c % P >= P - W) and the dynamic claims.
    "static": ("op_variants", {}),  # alternating chunks, half the staging each
    "same": ("op_variants", dict(BRF_SPLIT_STAGE="same")),  # reader keeps its slots, writer takes the rest
    "w13": ("op_variants", dict(BRF_SPLIT_P=3, BRF_SPLIT_W=1)),
    "w23": ("op_variants", dict(BRF_SPLIT_P=3, BRF_SPLIT_W=2)),
    "dyn": ("op_variants", dict(BRF_DYN=True)),  # = op_copy's split
    "dynpoll": ("op_variants", dict(BRF_DYN=True, BRF_STORE_SERVE=True)),
    "dynfill": ("op_variants", dict(BRF_DYN=True, BRF_SPLIT_STAGE="fill")),
    "dynsmall": ("op_variants", dict(BRF_DYN=True, BRF_CHUNK_BYTES=8192)),
    "dyndbg": ("op_variants", dict(BRF_DYN=True, BRF_DEBUG=True)),
    # Controls: the unsplit fill with fewer staging slots (fewer chunks in flight).
    "n3": ("op_variants", dict(BRF_SPLIT=False, BRF_NSTG=3)),
    "n2": ("op_variants", dict(BRF_SPLIT=False, BRF_NSTG=2)),
}


def _bands(h_out):
    if h_out <= 3 * BAND_ROWS:
        return [(0, h_out)]
    mid = h_out // 2 - BAND_ROWS // 2
    return [(0, BAND_ROWS), (mid, mid + BAND_ROWS), (h_out - BAND_ROWS, h_out)]


def _ref_rows(x_hwc, weight, bias, pad, r0, r1):
    import torch

    top, bottom, left, right = pad
    h = x_hwc.shape[0]
    first = r0 - top
    last = (r1 - 1) - top + 3
    rows = x_hwc[max(first, 0) : min(last, h)].float().permute(2, 0, 1).unsqueeze(0)
    rows = torch.nn.functional.pad(rows, (left, right, max(0, -first), max(0, last - h)))
    return torch.nn.functional.conv2d(rows, weight.float(), bias.float())[0].permute(1, 2, 0)


def _pcc_bands(out, x_hwc, weight, bias, pad, h_out, w_out, c_out):
    import torch

    out = out.reshape(-1, out.shape[-1])[: h_out * w_out, :c_out].reshape(h_out, w_out, c_out)
    got, ref = [], []
    for r0, r1 in _bands(h_out):
        got.append(out[r0:r1].double().flatten())
        ref.append(_ref_rows(x_hwc, weight, bias, pad, r0, r1).double().flatten())
    got, ref = torch.cat(got), torch.cat(ref)
    return torch.corrcoef(torch.stack([got, ref]))[0, 1].item()


def _set_knobs(pkg, knobs):
    for p, defaults in KNOB_DEFAULTS.items():
        _, pd = MODS[p]
        for k, v in {**defaults, **(knobs if p == pkg else {})}.items():
            setattr(pd, k, v)
        for name in dir(pd):
            f = getattr(pd, name)
            if hasattr(f, "cache_clear"):
                f.cache_clear()


def _in_collect_pass():
    import sys

    plugin = sys.modules.get("tests.plugins.up_front_collect")
    return plugin is not None and getattr(plugin, "_PASS", "real") == "collect"


def _cases():
    return [int(c) for c in os.environ.get("BRF_CASES", "35,33,41").split(",") if c.strip()]


def _variants():
    return [v for v in os.environ.get("BRF_VARIANTS", "base,split").split(",") if v]


@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
@pytest.mark.parametrize("idx", _cases())
def test_brf_split(device, idx):
    import hashlib

    import torch

    H, W, c_in, c_out = FLUX[idx]
    h_out, w_out = H - 2, W
    torch.manual_seed(0)
    x = torch.randn(H, W, c_in, dtype=torch.bfloat16)
    weight = (torch.randn(c_out, c_in, 3, 3) / (9 * c_in) ** 0.5).to(torch.bfloat16)
    bias = (0.1 * torch.randn(c_out)).to(torch.bfloat16)
    tt_in = ttnn.from_torch(
        x.reshape(1, 1, H * W, c_in),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
    )
    variants = _variants()
    log = os.environ.get("BRF_LOG", "brf")
    if os.environ.get("BRF_ORDER") == "alt" and _cases().index(idx) % 2 == 1:
        variants = variants[::-1]
    iters = int(os.environ.get("BRF_ITERS", "1"))
    ref_out = None
    for vname in variants:
        pkg, knobs = VARIANTS[vname]
        op, pd = MODS[pkg]
        _set_knobs(pkg, knobs)
        device.clear_program_cache()
        prep = dict(
            stride=1,
            padding=PAD,
            dilation=1,
            input_dtype=ttnn.bfloat16,
            input_layout=ttnn.ROW_MAJOR_LAYOUT,
            weight_dtype=ttnn.bfloat16,
            input_memory_config=tt_in.memory_config(),
            device=device,
        )
        tt_w = op.prepare_conv2d_weights(weight, **prep)
        tt_b = op.prepare_conv2d_bias(bias, **prep)

        def run():
            return op.conv2d_3x3(
                tt_in, tt_w, input_height=H, input_width=W, bias=tt_b, padding=PAD, stride=1, compute_kernel_config=cfg
            )

        rec = dict(case=idx, variant=vname, knobs=dict(knobs), log=log, t=time.time())
        try:
            out = run()
            ttnn.synchronize_device(device)
            ttnn.ReadDeviceProfiler(device)
            for i in range(iters):
                ttnn.deallocate(out)
                signpost(header=f"cmp|{idx}|{vname}|{i}")
                out = run()
                ttnn.synchronize_device(device)
                ttnn.ReadDeviceProfiler(device)
            signpost(header=f"cmpend|{idx}|{vname}")
            if _in_collect_pass():
                ttnn.deallocate(out)
                continue
            got = ttnn.to_torch(out)
            ttnn.deallocate(out)
            rec["status"] = "ok"
            b = getattr(op, "LAST_BLOCKING", None)
            if b is not None:
                rec["blocking"] = dict(
                    bm=b.block_m_tiles,
                    bk=b.block_k_tiles,
                    m_blocks=b.max_m_blocks_per_core,
                    k_blocks=b.num_k_blocks,
                    depth_rows=b.im2col_rm_depth_rows,
                    resident=bool(getattr(b, "weights_resident", False)),
                    footprint=b.footprint_bytes,
                )
            valid = got.reshape(-1, got.shape[-1])[: h_out * w_out, :c_out].contiguous()
            rec["sha1"] = hashlib.sha1(valid.to(torch.bfloat16).view(torch.int16).numpy().tobytes()).hexdigest()
            gotf = got.float()
            rec["pcc"] = _pcc_bands(gotf, x, weight, bias, PAD, h_out, w_out, c_out)
            if ref_out is None:
                ref_out, rec["ref"] = valid, True
            gi = valid.to(torch.bfloat16).view(torch.int16).to(torch.int32)
            ri = ref_out.to(torch.bfloat16).view(torch.int16).to(torch.int32)
            ulp = (gi - ri).abs()
            rec["bit_identical_to_first"] = bool(torch.equal(gi, ri))
            rec["frac_diff_vs_first"] = float((ulp > 0).float().mean())
            rec["max_ulp_vs_first"] = int(ulp.max())
        except Exception as e:
            rec["status"] = "error"
            rec["error"] = str(e).splitlines()[0][:400] if str(e) else type(e).__name__
        ttnn.deallocate(tt_w)
        ttnn.deallocate(tt_b)
        print("BRF:", json.dumps(rec))
        with open(HERE / "runs" / f"{log}.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
    _set_knobs(None, {})
    ttnn.deallocate(tt_in)
