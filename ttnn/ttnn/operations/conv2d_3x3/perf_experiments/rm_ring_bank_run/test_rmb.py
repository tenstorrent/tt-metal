# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bake-off: ROW_MAJOR ring slab across M-blocks filled with bank-run DRAM reads, the next block's pixels filled under
compute (perf experiment, round 17, `rm_ring_bank_run`).

Re-run (from the repo root, after `export TT_METAL_HOME=$PWD PYTHONPATH=$PWD`):
    ttnn/ttnn/operations/conv2d_3x3/perf_experiments/rm_ring_bank_run/run.sh <label> 41,35 base,ring
(results: runs/<label>.txt = device ns per case|variant, runs/<label>.jsonl = PCC / bit-identity / blocking).

Variants (one dispatch per measured call, a warm-up call first):
  base   op_base/ (unmodified copy of the working-tree op)
  <name> op_copy/ with the knobs in VARIANTS (module constants of op_copy's descriptor, reset before every case)

Env: RMB_CASES (comma list of FLUX bh2x4 idx), RMB_VARIANTS (comma list, run in that order), RMB_ORDER=alt (reverse
the variant order on every other case), RMB_LOG (run-log name), RMB_ITERS (measured calls, default 1). Each variant's
calls are signposted `cmp|<case>|<variant>|<i>` (cmp_windows.py sums them). Every variant's output is compared
bit-exactly with the first variant's output of the same case (sha1 of the bytes) and its PCC vs torch is recorded
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
_BASE = "ttnn.operations.conv2d_3x3.perf_experiments.rm_ring_bank_run."
MODS = {}
for _k in ("op_base", "op_copy"):
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

KNOB_DEFAULTS = dict(
    RM_RING=True,
    RM_RING_OVERLAP=True,
    RM_RING_NSTG=2,
    RM_RING_CHUNK_BYTES=8192,
    RM_RING_MIN_BM_FRAC=0.0,
    RM_RING_NARROW_N_TILES=2,
)
# Menu variants measured during the experiment whose knobs were removed from op_copy afterwards (numbers in README.md):
# band-stream fill order (flat), non-resident ring blocking (P3 -1.6%, P1 +75%), ring fill ablations (diagnostic).
VARIANTS = {
    "base": ("op_base", {}),
    "ring": ("op_copy", {}),
    "noring": ("op_copy", dict(RM_RING=False)),
    "serial": ("op_copy", dict(RM_RING_OVERLAP=False)),
    "s4c16": ("op_copy", dict(RM_RING_NSTG=4, RM_RING_CHUNK_BYTES=16384)),
    "s3c8": ("op_copy", dict(RM_RING_NSTG=3, RM_RING_CHUNK_BYTES=8192)),
    "s4c8": ("op_copy", dict(RM_RING_NSTG=4, RM_RING_CHUNK_BYTES=8192)),
    "s2c16": ("op_copy", dict(RM_RING_NSTG=2, RM_RING_CHUNK_BYTES=16384)),
    "f50": ("op_copy", dict(RM_RING_MIN_BM_FRAC=0.5)),
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


def _set_knobs(knobs):
    _, pd = MODS["op_copy"]
    for k, v in {**KNOB_DEFAULTS, **knobs}.items():
        assert hasattr(pd, k), k
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
    return [int(c) for c in os.environ.get("RMB_CASES", "41,35").split(",") if c.strip()]


def _variants():
    return [v for v in os.environ.get("RMB_VARIANTS", "base,ring").split(",") if v]


_LAST = {}


def _recorder(op):
    orig = op.select_blocking
    if getattr(orig, "_rmb_wrapped", False):
        return

    def wrapped(*a, **k):
        b = orig(*a, **k)
        _LAST["b"] = b
        return b

    wrapped._rmb_wrapped = True
    wrapped.cache_clear = orig.cache_clear
    op.select_blocking = wrapped


@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
@pytest.mark.parametrize("idx", _cases())
def test_rmb(device, idx):
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
    log = os.environ.get("RMB_LOG", "rmb")
    if os.environ.get("RMB_ORDER") == "alt" and _cases().index(idx) % 2 == 1:
        variants = variants[::-1]
    iters = int(os.environ.get("RMB_ITERS", "1"))
    ref_sha = None
    # RMB_DEADLINE (epoch s): variants that would start later are skipped, so a run queued behind the device lock
    # never outlives the caller's time cap mid-device.
    deadline = float(os.environ.get("RMB_DEADLINE", "inf"))
    for vname in variants:
        if not _in_collect_pass() and time.time() > deadline:
            print("RMB:", json.dumps(dict(case=idx, variant=vname, status="skipped_deadline")))
            continue
        pkg, knobs = VARIANTS[vname]
        op, pd = MODS[pkg]
        _recorder(op)
        _set_knobs(knobs if pkg == "op_copy" else {})
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
            b = _LAST.get("b")
            if b is not None:
                rec["blocking"] = dict(
                    bm=b.block_m_tiles,
                    bk=b.block_k_tiles,
                    m_blocks=b.max_m_blocks_per_core,
                    k_blocks=b.num_k_blocks,
                    resident=b.weights_resident,
                    rm_ring_px=getattr(b, "rm_ring_px", 0),
                    stage=(getattr(b, "rm_stage_n", 0), getattr(b, "rm_stage_px", 0)),
                    im2col_depth=b.im2col_rm_depth_rows,
                    footprint=b.footprint_bytes,
                )
            valid = got.reshape(-1, got.shape[-1])[: h_out * w_out, :c_out].contiguous()
            sha = hashlib.sha1(valid.view(torch.int16).numpy().tobytes()).hexdigest()
            rec["sha1"] = sha
            if ref_sha is None:
                ref_sha, ref_valid = sha, valid
                rec["ref"] = True
            rec["bit_identical_to_first"] = sha == ref_sha
            if sha != ref_sha:
                ulp = (valid.view(torch.int16).to(torch.int32) - ref_valid.view(torch.int16).to(torch.int32)).abs()
                rec["frac_diff"] = float((ulp > 0).float().mean())
                rec["max_ulp"] = int(ulp.max())
            rec["pcc"] = _pcc_bands(got.float(), x, weight, bias, PAD, h_out, w_out, c_out)
        except Exception as e:
            rec["status"] = "error"
            rec["error"] = str(e).splitlines()[0][:400] if str(e) else type(e).__name__
        ttnn.deallocate(tt_w)
        ttnn.deallocate(tt_b)
        print("RMB:", json.dumps(rec))
        with open(HERE / "runs" / f"{log}.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
    _set_knobs({})
    ttnn.deallocate(tt_in)
