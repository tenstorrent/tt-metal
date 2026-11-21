# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bake-off: the writer's DRAM output store over both NoCs (perf experiment, round 19, `store_noc_split`).

Re-run (from the repo root):
    ttnn/ttnn/operations/conv2d_3x3/perf_experiments/store_noc_split/run.sh <label> 41,36 base,dyn
(results: runs/<label>.txt = device ns per case|variant, runs/<label>.jsonl = PCC / bit-identity / blocking).

Variants (one dispatch per measured call, one warm-up call first): see VARIANTS.
  base   op_base/ (unmodified copy of the working-tree op: every output tile on NoC1)
  final  op_final/ (the recommendation: routed store, no knobs; integration.patch = op_base -> op_final)
  off    op_copy/ with STORE_SPLIT 0: must equal base
  dyn*   op_copy/, dynamic: NoC1 while its unsent-write backlog < D1, else NoC0 while < D0
  s*     op_copy/, static: tile i on NoC0 when i % DEN < NUM

Env: RFS_CASES (comma list of FLUX bh2x4 idx), RFS_VARIANTS (comma list, run in that order), RFS_ORDER=alt (reverse the
variant order on every other case), RFS_LOG (run-log name), RFS_ITERS (measured calls, default 1). Each variant's calls
are signposted `cmp|<case>|<variant>|<i>` (cmp_windows.py sums them). Every variant's output is compared bit-exactly
with the first variant's output of the same case (sha1 of the output bytes, ulp diff) and its PCC vs torch is recorded
(output-row bands, as the FLUX harness). Precision contract fixed: bf16, HiFi4, fp32 DEST, no approx, bias.
"""

import hashlib
import importlib
import json
import os
import time
from pathlib import Path

import pytest
import ttnn
from tracy import signpost

HERE = Path(__file__).parent
_BASE = "ttnn.operations.conv2d_3x3.perf_experiments.store_noc_split."
MODS = {}
LAST_BLOCKING = {}


def _recording(pkg, create):
    def wrapped(*args, **kwargs):
        LAST_BLOCKING[pkg] = kwargs.get("blocking")
        return create(*args, **kwargs)

    return wrapped


for _k in ("op_base", "op_copy", "op_final"):
    MODS[_k] = (
        importlib.import_module(_BASE + _k + ".conv2d_3x3"),
        importlib.import_module(_BASE + _k + ".conv2d_3x3_program_descriptor"),
    )
    assert Path(MODS[_k][1].KERNEL_DIR).parent.name == _k, MODS[_k][1].KERNEL_DIR
    # The blocking each call ran with (for the record).
    MODS[_k][0].create_program_descriptor = _recording(_k, MODS[_k][0].create_program_descriptor)

# Cases: (H, W, C_in, C_out, stride, padding (top, bottom, left, right), TILE input).
# "<n>": FLUX.2 bh2x4 case n of test_flux_conv3x3_compare.py (ROW_MAJOR input, padding (0, 0, 1, 1));
# "t<n>": the same on TILE input (FLUX_CMP_BH_LAYOUT=TILE); "x<n>": its 1x1 (single-device, TILE input) case n;
# "r<n>": extra ROW_MAJOR regimes (not FLUX cases).
_BH = {
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
FLUX = {str(k): (*v, 1, (0, 0, 1, 1), False) for k, v in _BH.items()}
FLUX.update({f"t{k}": (*v, 1, (0, 0, 1, 1), True) for k, v in _BH.items()})
FLUX.update(
    {
        "x10": (256, 256, 256, 512, 1, (1, 1, 1, 1), True),
        "x12": (512, 512, 256, 256, 1, (1, 1, 1, 1), True),
        "x15": (512, 512, 256, 256, 2, (0, 1, 0, 1), True),
        "x20": (512, 512, 256, 512, 1, (1, 1, 1, 1), True),
        "x22": (1024, 1024, 256, 256, 1, (1, 1, 1, 1), True),
        "r1": (258, 1024, 128, 512, 1, (0, 0, 1, 1), False),
        "r2": (130, 512, 128, 512, 1, (0, 0, 1, 1), False),
        "r3": (258, 512, 256, 512, 1, (0, 0, 1, 1), False),
    }
)
BAND_ROWS = 32

KNOB_DEFAULTS = dict(
    STORE_EXP=False,
    STORE_FOLD=False,
    STORE_OUT_FREE=False,
    STORE_DIAG=False,
    STORE_HK=1,
    STORE_HT=1,
    STORE_FIRST_BLOCKS=0,
    STORE_SKIP_BLOCKS=0,
    STORE_N1Y=1,
    STORE_SPLIT=0,
    STORE_PIPE=0,
    STORE_D1=2,
    STORE_D0=2,
    STORE_NUM=1,
    STORE_DEN=2,
)


def _v(split=0, pipe=0, d1=2, d0=2, num=1, den=2):
    return dict(
        STORE_EXP=True, STORE_SPLIT=split, STORE_PIPE=pipe, STORE_D1=d1, STORE_D0=d0, STORE_NUM=num, STORE_DEN=den
    )


VARIANTS = {
    "base": ("op_base", {}),
    "final": ("op_final", {}),
    "off": ("op_copy", {}),
    # flush per row group (the base structure)
    "dyn": ("op_copy", _v(split=1, d1=2, d0=2)),
    "dyn1": ("op_copy", _v(split=1, d1=1, d0=1)),
    "s50": ("op_copy", _v(split=2, num=1, den=2)),
    "s25": ("op_copy", _v(split=2, num=1, den=4)),
    "s100": ("op_copy", _v(split=2, num=1, den=1)),
    # row-group pipeline (pop a group once its writes left L1)
    "pipe": ("op_copy", _v(split=0, pipe=1)),
    "route": ("op_copy", _v(split=3, pipe=1)),
    "proute": ("op_copy", _v(split=4, pipe=1, d1=4, d0=4)),
    "proute2": ("op_copy", _v(split=4, pipe=1, d1=2, d0=2)),
    "rk2": ("op_copy", dict(_v(split=3, pipe=1), STORE_HK=2)),
    "rk4": ("op_copy", dict(_v(split=3, pipe=1), STORE_HK=4)),
    "rx2": ("op_copy", dict(_v(split=3, pipe=1), STORE_HK=2, STORE_N1Y=0)),
    "rx4": ("op_copy", dict(_v(split=3, pipe=1), STORE_HK=4, STORE_N1Y=0)),
    "rt3": ("op_copy", dict(_v(split=3, pipe=1), STORE_HT=3)),
    "rt5": ("op_copy", dict(_v(split=3, pipe=1), STORE_HT=5)),
    "rt8": ("op_copy", dict(_v(split=3, pipe=1), STORE_HT=8)),
    "sp2": ("op_copy", _v(split=5, pipe=1, d1=2, d0=4)),
    "sp4": ("op_copy", _v(split=5, pipe=1, d1=4, d0=4)),
    "sp6": ("op_copy", _v(split=5, pipe=1, d1=6, d0=4)),
    "rf1": ("op_copy", dict(_v(split=3, pipe=1), STORE_FIRST_BLOCKS=1)),
    "rf2": ("op_copy", dict(_v(split=3, pipe=1), STORE_FIRST_BLOCKS=2)),
    "rf1n": ("op_copy", dict(_v(split=3, pipe=0), STORE_FIRST_BLOCKS=1)),
    "s0": ("op_copy", _v(split=0, pipe=0)),
    "rs1": ("op_copy", dict(_v(split=3, pipe=0), STORE_SKIP_BLOCKS=1)),
    "free": ("op_copy", dict(STORE_OUT_FREE=True)),
    "r0free": ("op_copy", dict(_v(split=3, pipe=0), STORE_OUT_FREE=True)),
    "route0": ("op_copy", _v(split=3, pipe=0)),
    "diag": ("op_copy", dict(_v(split=1, pipe=1, d1=2, d0=4), STORE_DIAG=True)),
    "pdyn4": ("op_copy", _v(split=1, pipe=1, d1=4, d0=4)),
    "pdyn6": ("op_copy", _v(split=1, pipe=1, d1=6, d0=4)),
    "pdyn2": ("op_copy", _v(split=1, pipe=1, d1=2, d0=4)),
    "pdyn3": ("op_copy", _v(split=1, pipe=1, d1=3, d0=8)),
}


def _bands(h_out):
    if h_out <= 3 * BAND_ROWS:
        return [(0, h_out)]
    mid = h_out // 2 - BAND_ROWS // 2
    return [(0, BAND_ROWS), (mid, mid + BAND_ROWS), (h_out - BAND_ROWS, h_out)]


def _ref_rows(x_hwc, weight, bias, stride, pad, r0, r1):
    import torch

    top, bottom, left, right = pad
    h = x_hwc.shape[0]
    first = r0 * stride - top
    last = (r1 - 1) * stride - top + 3
    rows = x_hwc[max(first, 0) : min(last, h)].float().permute(2, 0, 1).unsqueeze(0)
    rows = torch.nn.functional.pad(rows, (left, right, max(0, -first), max(0, last - h)))
    return torch.nn.functional.conv2d(rows, weight.float(), bias.float(), stride=stride)[0].permute(1, 2, 0)


def _pcc_bands(out, x_hwc, weight, bias, stride, pad, h_out, w_out, c_out):
    import torch

    out = out.reshape(-1, out.shape[-1])[: h_out * w_out, :c_out].reshape(h_out, w_out, c_out)
    got, ref = [], []
    for r0, r1 in _bands(h_out):
        got.append(out[r0:r1].double().flatten())
        ref.append(_ref_rows(x_hwc, weight, bias, stride, pad, r0, r1).double().flatten())
    got, ref = torch.cat(got), torch.cat(ref)
    return torch.corrcoef(torch.stack([got, ref]))[0, 1].item()


def _set_knobs(knobs):
    for pkg in [k for k in MODS if k.startswith("op_copy")]:
        _, pd = MODS[pkg]
        for k, v in {**KNOB_DEFAULTS, **knobs}.items():
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
    return [c.strip() for c in os.environ.get("RFS_CASES", "43").split(",") if c.strip()]


def _variants():
    return [v for v in os.environ.get("RFS_VARIANTS", "base,split").split(",") if v]


@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
@pytest.mark.parametrize("idx", _cases())
def test_rfs(device, idx):
    import torch

    H, W, c_in, c_out, stride, pad, tile = FLUX[idx]
    h_out = (H + pad[0] + pad[1] - 3) // stride + 1
    w_out = (W + pad[2] + pad[3] - 3) // stride + 1
    layout = ttnn.TILE_LAYOUT if tile else ttnn.ROW_MAJOR_LAYOUT
    pad_arg = pad if len(set(pad)) > 1 else pad[0]
    torch.manual_seed(0)
    x = torch.randn(H, W, c_in, dtype=torch.bfloat16)
    weight = (torch.randn(c_out, c_in, 3, 3) / (9 * c_in) ** 0.5).to(torch.bfloat16)
    bias = (0.1 * torch.randn(c_out)).to(torch.bfloat16)
    tt_in = ttnn.from_torch(
        x.reshape(1, 1, H * W, c_in),
        dtype=ttnn.bfloat16,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
    )
    variants = _variants()
    log = os.environ.get("RFS_LOG", "rfs")
    if os.environ.get("RFS_ORDER") == "alt" and _cases().index(idx) % 2 == 1:
        variants = variants[::-1]
    iters = int(os.environ.get("RFS_ITERS", "1"))
    ref_out = None
    for vname in variants:
        pkg, knobs = VARIANTS[vname]
        op, pd = MODS[pkg]
        _set_knobs(knobs if pkg.startswith("op_copy") else {})
        device.clear_program_cache()
        prep = dict(
            stride=stride,
            padding=pad_arg,
            dilation=1,
            input_dtype=ttnn.bfloat16,
            input_layout=layout,
            weight_dtype=ttnn.bfloat16,
            input_memory_config=tt_in.memory_config(),
            device=device,
        )
        tt_w = op.prepare_conv2d_weights(weight, **prep)
        tt_b = op.prepare_conv2d_bias(bias, **prep)

        def run():
            return op.conv2d_3x3(
                tt_in,
                tt_w,
                input_height=H,
                input_width=W,
                bias=tt_b,
                padding=pad_arg,
                stride=stride,
                compute_kernel_config=cfg,
            )

        rec = dict(case=idx, variant=vname, knobs={k: v for k, v in knobs.items()}, log=log, t=time.time())
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
            got = ttnn.to_torch(out).float()
            ttnn.deallocate(out)
            rec["status"] = "ok"
            b = LAST_BLOCKING.get(pkg)
            if b is not None:
                rec["blocking"] = dict(
                    ng=getattr(b, "n_groups", 1),
                    cores=b.num_m_groups * getattr(b, "n_groups", 1),
                    core_m=b.core_m_tiles,
                    bm=b.block_m_tiles,
                    bk=b.block_k_tiles,
                    bn=b.block_n_tiles,
                    m_blocks=b.max_m_blocks_per_core,
                    k_blocks=b.num_k_blocks,
                    resident=b.weights_resident,
                    footprint=b.footprint_bytes,
                    depth=b.im2col_rm_depth_rows,
                    ring_px=getattr(b, "rm_ring_px", 0),
                    split=getattr(b, "rm_split", False),
                    out_cb=getattr(b, "out_depth_row_groups", 2),
                    stage=(getattr(b, "rm_stage_n", 0), getattr(b, "rm_stage_px", 0)),
                )
            rec["pcc"] = _pcc_bands(got, x, weight, bias, stride, pad, h_out, w_out, c_out)
            valid = got.reshape(-1, got.shape[-1])[: h_out * w_out, :c_out].to(torch.bfloat16).contiguous()
            rec["sha1"] = hashlib.sha1(valid.view(torch.int16).numpy().tobytes()).hexdigest()[:16]
            if ref_out is None:
                ref_out, rec["ref"] = valid, True
            gi = valid.view(torch.int16).to(torch.int32)
            ri = ref_out.view(torch.int16).to(torch.int32)
            ulp = (gi - ri).abs()
            rec["bit_identical_to_first"] = bool(torch.equal(gi, ri))
            rec["n_diff_vs_first"] = int((ulp > 0).sum())
            rec["max_ulp_vs_first"] = int(ulp.max())
        except Exception as e:
            rec["status"] = "error"
            rec["error"] = str(e).splitlines()[0][:400] if str(e) else type(e).__name__
        ttnn.deallocate(tt_w)
        ttnn.deallocate(tt_b)
        print("RFS:", json.dumps(rec))
        if not _in_collect_pass():
            with open(HERE / "runs" / f"{log}.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
    _set_knobs({})
    ttnn.deallocate(tt_in)
